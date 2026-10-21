use super::{
    error::{DeviceIndexOutOfRange, DeviceUnavailable, OutOfDeviceMemory},
    DeviceEngine, DeviceEngineBuffer, DeviceEngineKernel, DeviceId, DeviceInfo, DeviceLost,
    DeviceOptions, Features, KernelDesc, KernelKey,
};

use anyhow::{Error, Result};
use ash::vk::Handle;
use crossbeam_channel::{Receiver, Sender};
use dashmap::DashMap;
use parking_lot::Mutex;
use std::{
    mem::MaybeUninit,
    ops::Range,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, CommandBuffer as UnsafeCommandBuffer,
        CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsage, CopyBufferInfo,
        RecordingCommandBuffer as UnsafeCommandBufferBuilder,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        layout::{DescriptorSetLayout, DescriptorType},
        sys::RawDescriptorSet,
        WriteDescriptorSet,
    },
    device::{
        Device, DeviceCreateInfo, DeviceOwned, Queue, QueueCreateInfo, QueueFlags, QueueGuard,
    },
    instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
            DebugUtilsMessengerCallback, DebugUtilsMessengerCreateInfo,
        },
        Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions,
    },
    library::VulkanLibrary,
    memory::allocator::{
        AllocationCreateInfo, GenericMemoryAllocatorCreateInfo, MemoryTypeFilter,
        StandardMemoryAllocator,
    },
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    shader::{ShaderModule, ShaderStages},
    sync::semaphore::Semaphore,
    VulkanObject,
};

pub struct Engine {
    info: Arc<DeviceInfo>,
    semaphore: Arc<Semaphore>,
    epoch: AtomicU64,
    pending: Arc<AtomicU64>,
    frame_outer: Mutex<FrameOuter>,
    host_buffer_sender: Sender<HostBuffer>,
    host_buffer_receiver: Receiver<HostBuffer>,
    kernels: DashMap<KernelKey, KernelInner>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    queue: Arc<Queue>,
    engine_exited: Arc<AtomicBool>,
    worker_exited: Arc<AtomicBool>,
    _instance: Arc<Instance>,
}

impl Engine {
    unsafe fn transfer(
        &self,
        src: Subbuffer<[u8]>,
        dst: Subbuffer<[u8]>,
        host_buffer: &mut HostBuffer,
        dst_device_buffer: Option<&DeviceBuffer>,
    ) -> Result<()> {
        let mut frame_outer = self.frame_outer.lock();
        unsafe { frame_outer.transfer(&self.epoch, src, dst, host_buffer, dst_device_buffer) }
    }
    unsafe fn compute(
        &self,
        kernel_desc: &Arc<KernelDesc>,
        pipeline: &Arc<ComputePipeline>,
        groups: u32,
        buffers: &[Arc<DeviceBuffer>],
        push_consts: &[u8],
        debug_printf_panic: Option<Arc<AtomicBool>>,
    ) -> Result<()> {
        let mut frame_outer = self.frame_outer.lock();
        let new_descriptors: u32 = buffers.len().try_into().unwrap();
        if frame_outer.kernels >= Frame::MAX_KERNELS
            || frame_outer.descriptors + new_descriptors > Frame::MAX_DESCRIPTORS
        {
            loop {
                if frame_outer.empty.load(Ordering::SeqCst) {
                    break;
                }
                if self.worker_exited.load(Ordering::SeqCst) {
                    return Err(DeviceLost(self.id()).into());
                }
                std::hint::spin_loop();
            }
        }
        unsafe {
            frame_outer.compute(
                kernel_desc,
                &self.epoch,
                pipeline,
                groups,
                buffers,
                push_consts,
                debug_printf_panic,
            )
        }
    }
    fn wait_pending(&self, epoch: u64) -> Result<(), DeviceLost> {
        while self.pending.load(Ordering::SeqCst) < epoch {
            if self.worker_exited.load(Ordering::SeqCst) {
                return Err(DeviceLost(self.id()));
            }
            std::hint::spin_loop();
        }
        Ok(())
    }
    fn wait_epoch(&self, epoch: u64) -> Result<(), DeviceLost> {
        loop {
            let result = unsafe { wait_semaphore(self.queue.device(), &self.semaphore, epoch) };
            match result {
                ash::vk::Result::SUCCESS => return Ok(()),
                ash::vk::Result::TIMEOUT => (),
                _ => return Err(DeviceLost(self.id())),
            }
            if self.worker_exited.load(Ordering::SeqCst) {
                return Err(DeviceLost(self.id()));
            }
            std::hint::spin_loop();
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.engine_exited.store(true, Ordering::SeqCst);
        while !self.worker_exited.load(Ordering::SeqCst) {}
        let result = self.queue.with(|mut x| x.wait_idle());
        if !std::thread::panicking() {
            result.unwrap();
        }
    }
}

impl DeviceEngine for Engine {
    type DeviceBuffer = DeviceBuffer;
    type Kernel = Kernel;
    fn new(options: DeviceOptions) -> anyhow::Result<std::sync::Arc<Self>> {
        let DeviceOptions {
            index,
            optimal_features,
        } = options;
        let library = VulkanLibrary::new().map_err(|e| Error::new(DeviceUnavailable).context(e))?;
        let debug_printf = Arc::new(AtomicBool::default());
        let debug_printf2 = debug_printf.clone();
        let debug_create_info = DebugUtilsMessengerCreateInfo {
            message_severity: DebugUtilsMessageSeverity::INFO,
            message_type: DebugUtilsMessageType::VALIDATION,
            ..DebugUtilsMessengerCreateInfo::user_callback(unsafe {
                DebugUtilsMessengerCallback::new(move |_, _, msg| {
                    if debug_printf2.load(Ordering::SeqCst) {
                        return;
                    }
                    if msg.message_id_name
                        == Some("UNASSIGNED-khronos-validation-createinstance-status-message")
                        && msg.message.contains("Khronos Validation Layer Active:")
                        && msg.message.contains(
                            "Current Enables: VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT.",
                        )
                    {
                        debug_printf2.store(true, Ordering::SeqCst);
                    }
                })
            })
        };
        let instance_create_info = InstanceCreateInfo {
            enabled_extensions: InstanceExtensions {
                ext_debug_utils: true,
                ..Default::default()
            },
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            debug_utils_messengers: vec![debug_create_info],
            ..InstanceCreateInfo::application_from_cargo_toml()
        };
        let instance = Instance::new(library, instance_create_info)?;
        let debug_printf = debug_printf.load(Ordering::SeqCst);
        let mut physical_devices = instance.enumerate_physical_devices()?;
        let devices = physical_devices.len();
        let physical_device = if let Some(physical_device) = physical_devices.nth(index) {
            physical_device
        } else {
            return Err(DeviceIndexOutOfRange { index, devices }.into());
        };
        let name = physical_device.properties().device_name.clone();
        let optimal_device_extensions = vulkano::device::DeviceExtensions {
            khr_vulkan_memory_model: true,
            khr_shader_float_controls: true,
            khr_shader_float_controls2: true,
            ext_subgroup_size_control: true,
            ..vulkano::device::DeviceExtensions::empty()
        };
        let device_extensions = physical_device
            .supported_extensions()
            .intersection(&optimal_device_extensions);
        let optimal_device_features = vulkano::device::DeviceFeatures {
            shader_float_controls2: true,
            vulkan_memory_model: true,
            timeline_semaphore: true,
            subgroup_size_control: true,
            shader_int8: optimal_features.contains(Features::INT8),
            shader_int16: optimal_features.contains(Features::INT16),
            shader_int64: optimal_features.contains(Features::INT64),
            shader_float16: optimal_features.contains(Features::FLOAT16),
            shader_float64: optimal_features.contains(Features::FLOAT64),
            storage_buffer8_bit_access: optimal_features.contains(Features::BUFFER8),
            storage_buffer16_bit_access: optimal_features.contains(Features::BUFFER16),
            storage_push_constant8: optimal_features.contains(Features::PUSH_CONSTANT8),
            storage_push_constant16: optimal_features.contains(Features::PUSH_CONSTANT16),
            ..vulkano::device::DeviceFeatures::empty()
        };
        let device_features = physical_device
            .supported_features()
            .intersection(&optimal_device_features);
        let compute_family = physical_device
            .queue_family_properties()
            .iter()
            .position(|x| {
                x.queue_flags.contains(QueueFlags::COMPUTE)
                    && !x.queue_flags.contains(QueueFlags::GRAPHICS)
            })
            .or_else(|| {
                physical_device
                    .queue_family_properties()
                    .iter()
                    .position(|x| x.queue_flags.contains(QueueFlags::COMPUTE))
            })
            .map(|x| x as u32)
            .unwrap();
        let queue_create_infos = vec![QueueCreateInfo {
            queue_family_index: compute_family,
            queues: vec![1f32],
            ..Default::default()
        }];
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: device_features,
                queue_create_infos,
                ..Default::default()
            },
        )?;
        let queue = queues.next().unwrap();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new(
            device.clone(),
            GenericMemoryAllocatorCreateInfo {
                block_sizes: &device
                    .physical_device()
                    .memory_properties()
                    .memory_types
                    .iter()
                    .map(|memory_type| {
                        let heap = &device.physical_device().memory_properties().memory_heaps
                            [memory_type.heap_index as usize];
                        if heap.size >= DeviceBuffer::MAX_SIZE as u64 {
                            DeviceBuffer::MAX_SIZE as u64
                        } else {
                            64_000_000
                        }
                    })
                    .collect::<Vec<_>>(),
                dedicated_allocation: false,
                ..Default::default()
            },
        ));
        let (host_buffer_sender, host_buffer_receiver) = crossbeam_channel::bounded(2);
        for _ in 0..2 {
            let buffer_info = BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                ..Default::default()
            };
            let allocation_info = AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            };
            let inner = Buffer::new_slice(
                memory_allocator.clone(),
                buffer_info,
                allocation_info,
                DeviceBuffer::HOST_BUFFER_SIZE as u64,
            )?;
            host_buffer_sender
                .send(HostBuffer {
                    inner,
                    queue: queue.clone(),
                    epoch: 0,
                })
                .unwrap();
        }
        let kernels = DashMap::default();
        let properties = device.physical_device().properties();
        let (min_subgroup_threads, max_subgroup_threads) = if device_features.subgroup_size_control
        {
            (
                properties.min_subgroup_size.unwrap_or(1),
                properties.max_subgroup_size.unwrap_or(128),
            )
        } else {
            (1, 128)
        };

        let mut features = Features::empty();
        if device_features.shader_int8 {
            features = features.union(Features::INT8);
        }
        if device_features.shader_int16 {
            features = features.union(Features::INT16);
        }
        if device_features.shader_int64 {
            features = features.union(Features::INT64);
        }
        if device_features.shader_float16 {
            features = features.union(Features::FLOAT16);
        }
        if device_features.shader_float64 {
            features = features.union(Features::FLOAT64);
        }
        if device_features.storage_buffer8_bit_access {
            features = features.union(Features::BUFFER8);
        }
        if device_features.storage_buffer16_bit_access {
            features = features.union(Features::BUFFER16);
        }
        if device_features.storage_push_constant8 {
            features = features.union(Features::PUSH_CONSTANT8);
        }
        if device_features.storage_push_constant16 {
            features = features.union(Features::PUSH_CONSTANT16);
        }
        if let Some(subgroup_features) = properties.subgroup_supported_operations {
            use vulkano::device::physical::SubgroupFeatures;

            if subgroup_features.contains(SubgroupFeatures::BASIC) {
                features = features.union(Features::SUBGROUP_BASIC);
            }
            if subgroup_features.contains(SubgroupFeatures::VOTE) {
                features = features.union(Features::SUBGROUP_VOTE);
            }
            if subgroup_features.contains(SubgroupFeatures::ARITHMETIC) {
                features = features.union(Features::SUBGROUP_ARITHMETIC);
            }
            if subgroup_features.contains(SubgroupFeatures::BALLOT) {
                features = features.union(Features::SUBGROUP_BALLOT);
            }
            if subgroup_features.contains(SubgroupFeatures::SHUFFLE) {
                features = features.union(Features::SUBGROUP_SHUFFLE);
            }
            if subgroup_features.contains(SubgroupFeatures::SHUFFLE_RELATIVE) {
                features = features.union(Features::SUBGROUP_SHUFFLE_RELATIVE);
            }
            if subgroup_features.contains(SubgroupFeatures::CLUSTERED) {
                features = features.union(Features::SUBGROUP_CLUSTERED);
            }
            if subgroup_features.contains(SubgroupFeatures::QUAD) {
                features = features.union(Features::SUBGROUP_QUAD);
            }
        }
        let info = Arc::new(DeviceInfo {
            index,
            name,
            device_id: properties.device_id,
            vendor_id: properties.vendor_id,
            max_groups: properties.max_compute_work_group_count[0],
            max_threads: properties.max_compute_work_group_size[0],
            min_subgroup_threads,
            max_subgroup_threads,
            features,
            debug_printf,
        });
        let mut worker = Worker::new(queue.clone(), index)?;
        let semaphore = worker.semaphore.clone();
        let epoch = AtomicU64::default();
        let pending = worker.pending.clone();
        let frame_outer = Mutex::new(FrameOuter::new(
            worker.ready_frame.clone(),
            worker.empty.clone(),
        ));
        let engine_exited = worker.engine_exited.clone();
        let worker_exited = worker.worker_exited.clone();
        std::thread::spawn(move || worker.run());
        Ok(Arc::new(Self {
            info,
            semaphore,
            epoch,
            pending,
            frame_outer,
            host_buffer_sender,
            host_buffer_receiver,
            kernels,
            memory_allocator,
            engine_exited,
            worker_exited,
            queue,
            _instance: instance,
        }))
    }
    fn id(&self) -> DeviceId {
        let index = self.info.index;
        let handle = self.queue.device().handle().as_raw().try_into().unwrap();
        DeviceId { index, handle }
    }
    fn info(&self) -> &Arc<DeviceInfo> {
        &self.info
    }
    fn wait(&self) -> Result<(), DeviceLost> {
        let epoch = self.epoch.load(Ordering::SeqCst);
        self.wait_epoch(epoch)
    }
}

fn new_semaphore(device: &Arc<Device>) -> Result<Semaphore> {
    let mut semaphore = MaybeUninit::uninit();
    let mut semaphore_type_create_info = ash::vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(ash::vk::SemaphoreType::TIMELINE);
    let semaphore_create_info =
        ash::vk::SemaphoreCreateInfo::default().push_next(&mut semaphore_type_create_info);
    unsafe {
        (device.fns().v1_0.create_semaphore)(
            device.handle(),
            &semaphore_create_info,
            std::ptr::null(),
            semaphore.as_mut_ptr(),
        )
        .result()
        .unwrap();
        Ok(Semaphore::from_handle(
            device.clone(),
            semaphore.assume_init(),
            Default::default(),
        ))
    }
}

unsafe fn queue_submit(
    queue: &Queue,
    _guard: &mut QueueGuard,
    command_buffer: &UnsafeCommandBuffer,
    semaphore: &Semaphore,
    epoch: u64,
) -> Result<(), ash::vk::Result> {
    let command_buffers = &[command_buffer.handle()];
    let signal_semaphore_values = &[epoch];
    let mut semaphore_submit_info = ash::vk::TimelineSemaphoreSubmitInfo::default()
        .signal_semaphore_values(signal_semaphore_values);
    let signal_semaphores = &[semaphore.handle()];
    let submit_info = ash::vk::SubmitInfo::default()
        .command_buffers(command_buffers)
        .signal_semaphores(signal_semaphores)
        .push_next(&mut semaphore_submit_info);
    let device = queue.device();
    unsafe {
        (device.fns().v1_0.queue_submit)(
            queue.handle(),
            1,
            [submit_info].as_ptr() as _,
            ash::vk::Fence::null(),
        )
        .result()?;
    }
    Ok(())
}

unsafe fn wait_semaphore(device: &Device, semaphore: &Semaphore, value: u64) -> ash::vk::Result {
    let semaphores = &[semaphore.handle()];
    let values = &[value];
    let semaphore_wait_info = ash::vk::SemaphoreWaitInfo::default()
        .semaphores(semaphores)
        .values(values);
    unsafe { (device.fns().v1_2.wait_semaphores)(device.handle(), &semaphore_wait_info, 0) }
}

struct HostBuffer {
    inner: Subbuffer<[u8]>,
    queue: Arc<Queue>,
    epoch: u64,
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        let result = self.queue.with(|mut x| x.wait_idle());
        if !std::thread::panicking() {
            result.unwrap();
        }
    }
}

struct FrameOuter {
    frame: Arc<Mutex<Frame>>,
    empty: Arc<AtomicBool>,
    kernels: u32,
    descriptors: u32,
}

impl FrameOuter {
    fn new(frame: Arc<Mutex<Frame>>, empty: Arc<AtomicBool>) -> Self {
        Self {
            frame,
            empty,
            kernels: 0,
            descriptors: 0,
        }
    }
    unsafe fn transfer(
        &mut self,
        epoch: &AtomicU64,
        src: Subbuffer<[u8]>,
        dst: Subbuffer<[u8]>,
        host_buffer: &mut HostBuffer,
        dst_device_buffer: Option<&DeviceBuffer>,
    ) -> Result<()> {
        let mut frame = self.frame.lock();
        if frame.command_buffer_builder.is_none() {
            self.kernels = 0;
            self.descriptors = 0;
            unsafe {
                frame.begin()?;
            }
            epoch.store(frame.epoch, Ordering::SeqCst);
            self.empty.store(false, Ordering::SeqCst);
        }
        unsafe {
            frame.transfer(src, dst, host_buffer, dst_device_buffer);
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    unsafe fn compute(
        &mut self,
        kernel_desc: &Arc<KernelDesc>,
        epoch: &AtomicU64,
        pipeline: &Arc<ComputePipeline>,
        groups: u32,
        buffers: &[Arc<DeviceBuffer>],
        push_consts: &[u8],
        debug_printf_panic: Option<Arc<AtomicBool>>,
    ) -> Result<()> {
        let new_descriptors: u32 = buffers.len().try_into().unwrap();
        let mut frame = self.frame.lock();
        if frame.command_buffer_builder.is_none() {
            self.kernels = 0;
            self.descriptors = 0;
            unsafe {
                frame.begin()?;
            }
            epoch.store(frame.epoch, Ordering::SeqCst);
            self.empty.store(false, Ordering::SeqCst);
        }
        unsafe {
            frame.compute(
                kernel_desc,
                pipeline,
                groups,
                buffers,
                push_consts,
                debug_printf_panic,
            );
        }
        self.kernels += 1;
        self.descriptors += new_descriptors;
        Ok(())
    }
}

struct Frame {
    queue: Arc<Queue>,
    command_allocator: Arc<StandardCommandBufferAllocator>,
    command_buffer_builder: Option<UnsafeCommandBufferBuilder>,
    descriptor_allocator: Arc<StandardDescriptorSetAllocator>,
    descriptor_sets: Vec<RawDescriptorSet>,
    buffers: Vec<Subbuffer<[u8]>>,
    epoch: u64,
    debug_kernel_desc_panic: Option<(Arc<KernelDesc>, Arc<AtomicBool>)>,
}

// SAFETY: Each frame owns a private command allocator, used only by this frame.
// Frame access (including recording, ending and allocation) is exclusive: either
// behind its mutex or owned by the worker. No other frame shares its command pool.
unsafe impl Send for Frame {}

impl Frame {
    const MAX_KERNELS: u32 = 4;
    const MAX_DESCRIPTORS: u32 = 32;
    fn new(queue: Arc<Queue>) -> Result<Self> {
        let device = queue.device();
        let command_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffer_builder = None;
        let descriptor_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let buffers = Vec::new();
        let epoch = 0;
        Ok(Self {
            queue,
            command_allocator,
            command_buffer_builder,
            descriptor_allocator,
            descriptor_sets: Vec::new(),
            buffers,
            epoch,
            debug_kernel_desc_panic: None,
        })
    }
    unsafe fn begin(&mut self) -> Result<()> {
        self.descriptor_sets.clear();
        self.command_buffer_builder.replace({
            UnsafeCommandBufferBuilder::new(
                self.command_allocator.clone(),
                self.queue.queue_family_index(),
                CommandBufferLevel::Primary,
                CommandBufferBeginInfo {
                    usage: CommandBufferUsage::OneTimeSubmit,
                    ..Default::default()
                },
            )?
        });
        Ok(())
    }
    unsafe fn transfer(
        &mut self,
        src: Subbuffer<[u8]>,
        dst: Subbuffer<[u8]>,
        host_buffer: &mut HostBuffer,
        dst_device_buffer: Option<&DeviceBuffer>,
    ) {
        let builder = self.command_buffer_builder.as_mut().unwrap();
        unsafe {
            builder
                .copy_buffer(&CopyBufferInfo::buffers(src.clone(), dst.clone()))
                .unwrap();
        }
        self.buffers.extend_from_slice(&[src, dst]);
        host_buffer.epoch = self.epoch;
        if let Some(dst_device_buffer) = dst_device_buffer {
            dst_device_buffer.epoch.store(self.epoch, Ordering::SeqCst);
        }
    }
    unsafe fn compute(
        &mut self,
        kernel_desc: &Arc<KernelDesc>,
        pipeline: &Arc<ComputePipeline>,
        groups: u32,
        buffers: &[Arc<DeviceBuffer>],
        push_consts: &[u8],
        debug_printf_panic: Option<Arc<AtomicBool>>,
    ) {
        let builder = self.command_buffer_builder.as_mut().unwrap();
        unsafe {
            builder.bind_pipeline_compute(pipeline).unwrap();
        }
        let pipeline_layout = pipeline.layout();
        if !buffers.is_empty() {
            let descriptor_set_layout = pipeline_layout.set_layouts().first().unwrap();
            let writes: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(binding, x)| {
                    WriteDescriptorSet::buffer(binding as u32, x.inner.as_ref().unwrap().clone())
                })
                .collect();
            unsafe {
                let descriptor_set = RawDescriptorSet::new(
                    self.descriptor_allocator.clone(),
                    descriptor_set_layout,
                    0,
                )
                .unwrap();
                descriptor_set.update(&writes, &[]).unwrap();
                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        pipeline_layout,
                        0,
                        &[&descriptor_set],
                        &[],
                    )
                    .unwrap();
                self.descriptor_sets.push(descriptor_set);
            }
        }
        if !push_consts.is_empty() {
            unsafe {
                (builder.device().fns().v1_0.cmd_push_constants)(
                    builder.handle(),
                    pipeline_layout.handle(),
                    ash::vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_consts.len() as u32,
                    push_consts.as_ptr().cast(),
                );
            }
        }
        unsafe {
            builder.dispatch([groups, 1, 1]).unwrap();
        }
        self.buffers
            .extend(buffers.iter().map(|x| x.inner.as_ref().unwrap().clone()));
        for (buffer, slice_desc) in buffers.iter().zip(kernel_desc.slice_descs.iter()) {
            if slice_desc.mutable {
                buffer.epoch.store(self.epoch, Ordering::SeqCst);
            }
        }
        if let Some(debug_printf_panic) = debug_printf_panic {
            self.debug_kernel_desc_panic
                .replace((kernel_desc.clone(), debug_printf_panic));
        }
    }
    unsafe fn finish(&mut self) {
        self.buffers.clear();
        self.debug_kernel_desc_panic.take();
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        let result = self.queue.with(|mut x| x.wait_idle());
        if !std::thread::panicking() {
            result.unwrap();
        }
    }
}

struct Worker {
    queue: Arc<Queue>,
    index: usize,
    semaphore: Arc<Semaphore>,
    empty: Arc<AtomicBool>,
    pending: Arc<AtomicU64>,
    ready_frame: Arc<Mutex<Frame>>,
    pending_frame: Frame,
    engine_exited: Arc<AtomicBool>,
    worker_exited: Arc<AtomicBool>,
}

impl Worker {
    fn new(queue: Arc<Queue>, index: usize) -> Result<Self> {
        let semaphore = Arc::new(new_semaphore(queue.device())?);
        let empty = Arc::new(AtomicBool::new(true));
        let pending = Arc::new(AtomicU64::default());
        let mut ready_frame = Frame::new(queue.clone())?;
        ready_frame.epoch = 1;
        let ready_frame = Arc::new(Mutex::new(ready_frame));
        let pending_frame = Frame::new(queue.clone())?;
        let engine_exited = Arc::new(AtomicBool::default());
        let worker_exited = Arc::new(AtomicBool::default());
        Ok(Self {
            queue,
            index,
            semaphore,
            empty,
            pending,
            ready_frame,
            pending_frame,
            engine_exited,
            worker_exited,
        })
    }
    fn run(&mut self) {
        let id = DeviceId {
            index: self.index,
            handle: self.queue.device().handle().as_raw().try_into().unwrap(),
        };
        loop {
            while self.empty.load(Ordering::SeqCst) {
                if self.engine_exited.load(Ordering::SeqCst) {
                    return;
                }
                std::hint::spin_loop();
            }
            {
                let mut ready_frame = self.ready_frame.lock();
                self.pending_frame.epoch = ready_frame.epoch + 1;
                self.empty.store(true, Ordering::SeqCst);
                std::mem::swap(&mut *ready_frame, &mut self.pending_frame);
            }
            self.pending
                .store(self.pending_frame.epoch, Ordering::SeqCst);
            let command_buffer = unsafe {
                self.pending_frame
                    .command_buffer_builder
                    .take()
                    .unwrap()
                    .end()
            }
            .unwrap();
            let _messenger = if let Some((kernel_desc, panicked)) =
                self.pending_frame.debug_kernel_desc_panic.take()
            {
                Some(
                    unsafe {
                        DebugUtilsMessenger::new(
                            self.queue.device().instance().clone(),
                            DebugUtilsMessengerCreateInfo {
                                message_severity: DebugUtilsMessageSeverity::INFO,
                                message_type: DebugUtilsMessageType::VALIDATION,
                                ..DebugUtilsMessengerCreateInfo::user_callback(
                                    DebugUtilsMessengerCallback::new(move |_, _, msg| {
                                        if let Some(layer_prefix) = msg.message_id_name.as_ref() {
                                            if layer_prefix.contains("DEBUG-PRINTF") {
                                                eprintln!(
                                                    "[{id:?} {}] {}",
                                                    kernel_desc.name, msg.message
                                                );
                                                if msg.message.contains("[Rust panicked at ") {
                                                    panicked.store(true, Ordering::SeqCst);
                                                }
                                            }
                                        }
                                    }),
                                )
                            },
                        )
                    }
                    .unwrap(),
                )
            } else {
                None
            };
            self.queue.with(|mut guard| unsafe {
                queue_submit(
                    &self.queue,
                    &mut guard,
                    &command_buffer,
                    &self.semaphore,
                    self.pending_frame.epoch,
                )
                .unwrap();
            });
            loop {
                let result = unsafe {
                    wait_semaphore(
                        self.queue.device(),
                        &self.semaphore,
                        self.pending_frame.epoch,
                    )
                };
                match result {
                    ash::vk::Result::SUCCESS => break,
                    ash::vk::Result::TIMEOUT => std::hint::spin_loop(),
                    _ => result.result().unwrap(),
                }
            }
            unsafe {
                self.pending_frame.finish();
            }
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.queue.with(|mut guard| guard.wait_idle());
        self.worker_exited.store(true, Ordering::SeqCst);
    }
}

const fn aligned_floor(x: usize, align: usize) -> usize {
    if x % align != 0 {
        align * (x / align)
    } else {
        x
    }
}

const fn aligned_ceil(x: usize, align: usize) -> usize {
    if x % align != 0 {
        aligned_floor(x, align) + align
    } else {
        x
    }
}

#[derive(Clone)]
pub(super) struct DeviceBuffer {
    inner: Option<Subbuffer<[u8]>>,
    engine: Arc<Engine>,
    offset: usize,
    len: usize,
    epoch: Arc<AtomicU64>,
}

impl DeviceBuffer {
    const MAX_LEN: usize = i32::MAX as usize;
    const MAX_SIZE: usize = aligned_ceil(Self::MAX_LEN, Self::ALIGN);
    const ALIGN: usize = 256;
    const HOST_BUFFER_SIZE: usize = 32_000_000;
    fn host_visible(&self) -> bool {
        if let Some(inner) = self.inner.as_ref() {
            inner.mapped_slice().is_ok()
        } else {
            false
        }
    }
}

impl DeviceEngineBuffer for DeviceBuffer {
    type Engine = Engine;
    fn engine(&self) -> &Arc<Self::Engine> {
        &self.engine
    }
    unsafe fn uninit(engine: Arc<Engine>, len: usize) -> Result<Self> {
        use vulkano::{memory::allocator::MemoryAllocatorError, VulkanError};
        let inner = if len > 0 {
            let len = aligned_ceil(len, Self::ALIGN);
            let usage =
                BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC;
            let buffer_info = BufferCreateInfo {
                usage,
                size: len.try_into().unwrap(),
                ..Default::default()
            };
            let allocation_info = AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            };
            use vulkano::{
                buffer::sys::RawBuffer,
                memory::{
                    allocator::{AllocationType, MemoryAllocator},
                    DeviceAlignment,
                },
            };
            let device = engine.queue.device();
            let raw_buffer = RawBuffer::new(device.clone(), buffer_info)?;
            let align = DeviceAlignment::new(DeviceBuffer::ALIGN.try_into().unwrap()).unwrap();
            let mut requirements = *raw_buffer.memory_requirements();
            requirements.layout = requirements.layout.align_to(align).unwrap();
            requirements.prefers_dedicated_allocation = false;
            let memory_alloc = engine
                .memory_allocator
                .allocate(requirements, AllocationType::Unknown, allocation_info, None)
                .map_err(|e| {
                    if let MemoryAllocatorError::AllocateDeviceMemory(vulkano::Validated::Error(
                        VulkanError::OutOfDeviceMemory,
                    )) = e
                    {
                        Error::new(OutOfDeviceMemory(engine.id())).context(e)
                    } else {
                        e.into()
                    }
                })?;
            let memory_alloc = unsafe {
                vulkano::memory::ResourceMemory::from_allocation(
                    engine.memory_allocator.clone(),
                    memory_alloc,
                )
            };
            let buffer = raw_buffer
                .bind_memory(memory_alloc)
                .map_err(|(e, _, _)| e)?;
            Some(Subbuffer::new(Arc::new(buffer)))
        } else {
            None
        };
        Ok(Self {
            inner,
            engine,
            offset: 0,
            len,
            epoch: Arc::new(AtomicU64::new(0)),
        })
    }
    fn upload(&self, data: &[u8]) -> Result<()> {
        debug_assert_eq!(data.len(), self.len);
        if self.len == 0 {
            return Ok(());
        }
        let buffer = if let Some(buffer) = self.inner.as_ref() {
            buffer
                .clone()
                .slice(self.offset as u64..(self.offset + self.len) as u64)
        } else {
            return Ok(());
        };
        let engine = &self.engine;
        let buffer_epoch = self.epoch.load(Ordering::SeqCst);
        if self.host_visible() {
            engine.wait_epoch(buffer_epoch)?;
            buffer.write().unwrap().copy_from_slice(data);
            return Ok(());
        }
        let mut offset = 0;
        for chunk in data.chunks(Self::HOST_BUFFER_SIZE) {
            let mut host_buffer = engine.host_buffer_receiver.recv().unwrap();
            let size = chunk.len() as u64;
            let buffer_slice = buffer.clone().slice(offset..offset + size);
            let host_slice = host_buffer.inner.clone().slice(0..size);
            engine.wait_epoch(host_buffer.epoch)?;
            host_slice.write().unwrap().copy_from_slice(chunk);
            engine.wait_pending(buffer_epoch)?;
            unsafe {
                engine.transfer(host_slice, buffer_slice, &mut host_buffer, Some(self))?;
            }
            engine.host_buffer_sender.send(host_buffer).unwrap();
            offset += size;
        }
        Ok(())
    }
    fn download(&self, data: &mut [u8]) -> Result<()> {
        debug_assert_eq!(data.len(), self.len);
        if self.len == 0 {
            return Ok(());
        }
        let buffer = if let Some(buffer) = self.inner.as_ref() {
            buffer
                .clone()
                .slice(self.offset as u64..(self.offset + self.len) as u64)
        } else {
            return Ok(());
        };
        let engine = &self.engine;
        let buffer_epoch = self.epoch.load(Ordering::SeqCst);
        if self.host_visible() {
            engine.wait_epoch(buffer_epoch)?;
            data.copy_from_slice(&buffer.read().unwrap());
            return Ok(());
        }
        struct HostCopy<'a> {
            chunk: &'a mut [u8],
            host_buffer: HostBuffer,
            host_slice: Subbuffer<[u8]>,
        }
        let mut host_copy: Option<HostCopy> = None;
        let mut offset = 0;
        for chunk in data.chunks_mut(Self::HOST_BUFFER_SIZE).chain([[].as_mut()]) {
            let prev_host_copy = host_copy.take();
            if !chunk.is_empty() {
                let mut host_buffer = engine.host_buffer_receiver.recv().unwrap();
                engine.wait_epoch(host_buffer.epoch)?;
                let size = chunk.len() as u64;
                let buffer_slice = buffer.clone().slice(offset..offset + size);
                let host_slice = host_buffer.inner.clone().slice(0..size);
                engine.wait_pending(buffer_epoch)?;
                unsafe {
                    engine.transfer(buffer_slice, host_slice.clone(), &mut host_buffer, None)?;
                }
                host_copy.replace(HostCopy {
                    chunk,
                    host_buffer,
                    host_slice,
                });
                offset += size;
            }
            if let Some(prev_host_copy) = prev_host_copy {
                let HostCopy {
                    chunk,
                    host_buffer,
                    host_slice,
                } = prev_host_copy;
                engine.wait_epoch(host_buffer.epoch)?;
                chunk.copy_from_slice(&host_slice.read().unwrap());
                engine.host_buffer_sender.send(host_buffer).unwrap();
            }
        }
        Ok(())
    }
    fn transfer(&self, dst: &Self) -> Result<()> {
        debug_assert_eq!(dst.len, self.len);
        if self.len == 0 {
            return Ok(());
        }
        let (buffer1, buffer2) =
            if let Some((buffer1, buffer2)) = self.inner.as_ref().zip(dst.inner.as_ref()) {
                let buffer1 = buffer1
                    .clone()
                    .slice(self.offset as u64..(self.offset + self.len) as u64);
                let buffer2 = buffer2
                    .clone()
                    .slice(dst.offset as u64..(dst.offset + dst.len) as u64);
                (buffer1, buffer2)
            } else {
                return Ok(());
            };
        let engine1 = &self.engine;
        let buffer1_epoch = self.epoch.load(Ordering::SeqCst);
        let engine2 = &dst.engine;
        let buffer2_epoch = dst.epoch.load(Ordering::SeqCst);
        if self.host_visible() && dst.host_visible() {
            engine1.wait_epoch(buffer1_epoch)?;
            engine2.wait_epoch(buffer2_epoch)?;
            buffer2
                .write()
                .unwrap()
                .copy_from_slice(&buffer1.read().unwrap());
            return Ok(());
        } else if self.host_visible() {
            engine1.wait_epoch(buffer1_epoch)?;
            return dst.upload(&buffer1.read().unwrap());
        } else if dst.host_visible() {
            engine2.wait_epoch(buffer2_epoch)?;
            return self.download(&mut buffer2.write().unwrap());
        }
        struct HostCopy {
            host_buffer1: HostBuffer,
            host_slice1: Subbuffer<[u8]>,
            buffer_slice2: Subbuffer<[u8]>,
        }
        let mut host_copy: Option<HostCopy> = None;
        let mut offset = 0;
        loop {
            let size = buffer1
                .size()
                .checked_sub(offset)
                .unwrap_or_default()
                .min(Self::HOST_BUFFER_SIZE as u64);
            let prev_host_copy = host_copy.take();
            if size > 0 {
                let mut host_buffer1 = engine1.host_buffer_receiver.recv().unwrap();
                let buffer_slice1 = buffer1.clone().slice(offset..offset + size);
                let host_slice1 = host_buffer1.inner.clone().slice(0..size);
                engine1.wait_epoch(host_buffer1.epoch)?;
                engine1.wait_pending(buffer1_epoch)?;
                unsafe {
                    engine1.transfer(
                        buffer_slice1,
                        host_slice1.clone(),
                        &mut host_buffer1,
                        None,
                    )?;
                }
                let buffer_slice2 = buffer2.clone().slice(offset..offset + size);
                host_copy.replace(HostCopy {
                    host_buffer1,
                    host_slice1,
                    buffer_slice2,
                });
                offset += size;
            }
            if let Some(prev_host_copy) = prev_host_copy {
                let HostCopy {
                    host_buffer1,
                    host_slice1,
                    buffer_slice2,
                } = prev_host_copy;
                let size = buffer_slice2.size();
                let mut host_buffer2 = engine2.host_buffer_receiver.recv().unwrap();
                let host_slice2 = host_buffer2.inner.clone().slice(0..size);
                engine1.wait_epoch(host_buffer1.epoch)?;
                engine2.wait_epoch(host_buffer2.epoch)?;
                host_slice2
                    .write()
                    .unwrap()
                    .copy_from_slice(&host_slice1.read().unwrap());
                engine1.host_buffer_sender.send(host_buffer1).unwrap();
                engine2.wait_pending(buffer2_epoch)?;
                unsafe {
                    engine2.transfer(host_slice2, buffer_slice2, &mut host_buffer2, Some(dst))?;
                }
                engine2.host_buffer_sender.send(host_buffer2).unwrap();
            } else if size == 0 {
                break;
            }
        }
        Ok(())
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn len(&self) -> usize {
        self.len
    }
    fn slice(self: &Arc<Self>, range: Range<usize>) -> Option<Arc<Self>> {
        let Range { start, end } = range;
        if start > self.len {
            return None;
        }
        if end > self.len {
            return None;
        }
        let offset = self.offset.checked_add(start)?;
        let len = end.checked_sub(start)?;
        Some(Arc::new(Self {
            offset,
            len,
            ..Self::clone(self)
        }))
    }
}

#[derive(Clone)]
struct KernelInner {
    desc: Arc<KernelDesc>,
    compute_pipeline: Arc<ComputePipeline>,
}

fn prepare_float_controls(device: &Device, desc: &KernelDesc) -> Result<Vec<u32>> {
    prepare_float_controls_with_properties(
        &desc.spirv,
        &desc.name,
        device.enabled_features().shader_float_controls2,
        device.physical_device().properties(),
        cfg!(target_os = "macos"),
    )
}

fn prepare_float_controls_with_properties(
    spirv: &[u32],
    kernel_name: &str,
    controls2: bool,
    properties: &vulkano::device::DeviceProperties,
    macos: bool,
) -> Result<Vec<u32>> {
    use rspirv::spirv::{Capability, ExecutionMode, Op};
    use rspirv::{
        binary::Assemble,
        dr::{Instruction, Operand},
    };
    use vulkano::device::physical::ShaderFloatControlsIndependence;
    let mut module = crate::spirv::load_words(spirv)?;
    if module
        .capabilities
        .iter()
        .any(|inst| inst.operands[0].unwrap_capability() == Capability::FloatControls2)
        && !controls2
    {
        anyhow::bail!(
            "kernel `{}` requires Vulkan shaderFloatControls2",
            kernel_name
        );
    }
    for mode in &module.execution_modes {
        if mode.operands[1].unwrap_execution_mode() != ExecutionMode::FPFastMathDefault {
            continue;
        }
        let ty = mode.operands[2].unwrap_id_ref();
        let width = module
            .types_global_values
            .iter()
            .find(|inst| inst.class.opcode == Op::TypeFloat && inst.result_id == Some(ty))
            .ok_or_else(|| anyhow::anyhow!("invalid FPFastMathDefault target type"))?
            .operands[0]
            .unwrap_literal_bit32();
        // Our safe policies never assume away NaN/infinity, even when NSZ is set.
        // Vulkan requires this property when any of those fast-math bits is absent.
        let supported = match width {
            16 => properties.shader_signed_zero_inf_nan_preserve_float16,
            32 => properties.shader_signed_zero_inf_nan_preserve_float32,
            64 => properties.shader_signed_zero_inf_nan_preserve_float64,
            _ => None,
        };
        if supported != Some(true) {
            anyhow::bail!(
                "kernel `{}` requires shaderSignedZeroInfNanPreserveFloat{}",
                kernel_name,
                width
            );
        }
    }
    // Restricted independence requires matching modes even for unused widths.
    // Add companion modes before checking support for the whole group.
    for (independence, candidates) in [
        (
            properties.denorm_behavior_independence,
            [
                ExecutionMode::DenormPreserve,
                ExecutionMode::DenormFlushToZero,
            ],
        ),
        (
            properties.rounding_mode_independence,
            [
                ExecutionMode::RoundingModeRTE,
                ExecutionMode::RoundingModeRTZ,
            ],
        ),
    ] {
        let widths: &[u32] = match independence {
            Some(ShaderFloatControlsIndependence::Float32Only) => &[16, 64],
            Some(ShaderFloatControlsIndependence::None) => &[16, 32, 64],
            _ => continue,
        };
        let existing: Vec<_> = module
            .execution_modes
            .iter()
            .filter(|inst| {
                candidates.contains(&inst.operands[1].unwrap_execution_mode())
                    && widths.contains(&inst.operands[2].unwrap_literal_bit32())
            })
            .cloned()
            .collect();
        if let Some(first) = existing.first() {
            for width in widths {
                if !existing
                    .iter()
                    .any(|inst| inst.operands[2] == Operand::LiteralBit32(*width))
                {
                    module.execution_modes.push(Instruction::new(
                        Op::ExecutionMode,
                        None,
                        None,
                        vec![
                            first.operands[0].clone(),
                            first.operands[1].clone(),
                            Operand::LiteralBit32(*width),
                        ],
                    ));
                }
            }
        }
    }
    let mut unsupported = Vec::new();
    for inst in &module.execution_modes {
        let mode = inst.operands[1].unwrap_execution_mode();
        if !matches!(
            mode,
            ExecutionMode::DenormPreserve
                | ExecutionMode::DenormFlushToZero
                | ExecutionMode::RoundingModeRTE
                | ExecutionMode::RoundingModeRTZ
        ) {
            continue;
        }
        let width = inst.operands[2].unwrap_literal_bit32();
        let (preserve, flush, rte, rtz) = match width {
            16 => (
                properties.shader_denorm_preserve_float16,
                properties.shader_denorm_flush_to_zero_float16,
                properties.shader_rounding_mode_rte_float16,
                properties.shader_rounding_mode_rtz_float16,
            ),
            32 => (
                properties.shader_denorm_preserve_float32,
                properties.shader_denorm_flush_to_zero_float32,
                properties.shader_rounding_mode_rte_float32,
                properties.shader_rounding_mode_rtz_float32,
            ),
            64 => (
                properties.shader_denorm_preserve_float64,
                properties.shader_denorm_flush_to_zero_float64,
                properties.shader_rounding_mode_rte_float64,
                properties.shader_rounding_mode_rtz_float64,
            ),
            _ => (None, None, None, None),
        };
        let supported = match mode {
            ExecutionMode::DenormPreserve => preserve,
            ExecutionMode::DenormFlushToZero => flush,
            ExecutionMode::RoundingModeRTE => rte,
            _ => rtz,
        };
        if supported != Some(true) {
            if !macos {
                anyhow::bail!("kernel `{kernel_name}` requires {mode:?} for f{width}, which this device does not support");
            }
            unsupported.push((mode, width));
        }
    }
    // On macOS, use device defaults where explicit modes are unsupported.
    // Remove coupled modes together; keep the fast-math restrictions intact.
    module.execution_modes.retain(|inst| {
        let mode = inst.operands[1].unwrap_execution_mode();
        !unsupported.iter().any(|&(unsupported_mode, width)| {
            if mode != unsupported_mode {
                return false;
            }
            let independence = match mode {
                ExecutionMode::DenormPreserve | ExecutionMode::DenormFlushToZero => {
                    properties.denorm_behavior_independence
                }
                _ => properties.rounding_mode_independence,
            };
            let other_width = inst.operands[2].unwrap_literal_bit32();
            width == other_width
                || independence == Some(ShaderFloatControlsIndependence::None)
                || (independence == Some(ShaderFloatControlsIndependence::Float32Only)
                    && width != 32
                    && other_width != 32)
        })
    });
    if macos {
        module.capabilities.retain(|inst| {
            let mode = match inst.operands[0].unwrap_capability() {
                Capability::DenormPreserve => ExecutionMode::DenormPreserve,
                Capability::DenormFlushToZero => ExecutionMode::DenormFlushToZero,
                Capability::RoundingModeRTE => ExecutionMode::RoundingModeRTE,
                Capability::RoundingModeRTZ => ExecutionMode::RoundingModeRTZ,
                _ => return true,
            };
            module
                .execution_modes
                .iter()
                .any(|inst| inst.operands[1].unwrap_execution_mode() == mode)
        });
    }
    // Vulkano also validates these modes and mixed-width independence.
    Ok(module.assemble())
}

impl KernelInner {
    fn new(engine: &Arc<Engine>, desc: Arc<KernelDesc>) -> Result<Self> {
        use vulkano::{
            descriptor_set::layout::{DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo},
            pipeline::layout::{PipelineLayout, PipelineLayoutCreateInfo, PushConstantRange},
            pipeline::{compute::ComputePipelineCreateInfo, PipelineShaderStageCreateInfo},
        };
        let device = engine.queue.device();
        let spirv = prepare_float_controls(device, &desc)?;
        let push_consts_range = desc.push_consts_range();
        let push_constant_range = if push_consts_range > 0 {
            Some(PushConstantRange {
                stages: ShaderStages::COMPUTE,
                offset: 0,
                size: push_consts_range,
            })
        } else {
            None
        };
        let entry_point = "main";
        let shader_module = unsafe {
            ShaderModule::new(
                device.clone(),
                vulkano::shader::ShaderModuleCreateInfo::new(&spirv),
            )?
        };
        let bindings = (0..desc.slice_descs.len())
            .map(|binding| {
                let descriptor_set_layout_binding = DescriptorSetLayoutBinding {
                    descriptor_count: 1,
                    stages: ShaderStages::COMPUTE,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
                };
                (binding.try_into().unwrap(), descriptor_set_layout_binding)
            })
            .collect();
        let descriptor_set_layout_create_info = DescriptorSetLayoutCreateInfo {
            bindings,
            ..DescriptorSetLayoutCreateInfo::default()
        };
        let descriptor_set_layout =
            DescriptorSetLayout::new(device.clone(), descriptor_set_layout_create_info)?;
        let pipeline_layout_create_info = PipelineLayoutCreateInfo {
            set_layouts: vec![descriptor_set_layout],
            push_constant_ranges: push_constant_range.into_iter().collect(),
            ..PipelineLayoutCreateInfo::default()
        };
        let pipeline_layout = PipelineLayout::new(device.clone(), pipeline_layout_create_info)?;
        let cache = None;
        let compute_pipeline = ComputePipeline::new(
            device.clone(),
            cache,
            ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(shader_module.entry_point(entry_point).unwrap()),
                pipeline_layout,
            ),
        )?;
        Ok(Self {
            desc,
            compute_pipeline,
        })
    }
}

#[cfg(test)]
mod float_control_tests {
    use super::prepare_float_controls_with_properties;
    use rspirv::{
        binary::Assemble,
        dr::{Builder, Instruction, Operand},
        spirv::{Capability, ExecutionMode, ExecutionModel, Op},
    };
    use vulkano::device::{physical::ShaderFloatControlsIndependence, DeviceProperties};

    fn shader(mode: ExecutionMode, width: u32) -> Vec<u32> {
        let mut builder = Builder::new();
        let float = builder.type_float(width, None);
        let uint = builder.type_int(32, 0);
        let flags = builder.constant_bit32(uint, 0);
        let entry = builder.id();
        builder.capability(Capability::FloatControls2);
        builder.entry_point(ExecutionModel::GLCompute, entry, "main", []);
        builder.module_mut().execution_modes.push(Instruction::new(
            Op::ExecutionModeId,
            None,
            None,
            vec![
                Operand::IdRef(entry),
                Operand::ExecutionMode(ExecutionMode::FPFastMathDefault),
                Operand::IdRef(float),
                Operand::IdRef(flags),
            ],
        ));
        builder.capability(match mode {
            ExecutionMode::DenormPreserve => Capability::DenormPreserve,
            _ => Capability::DenormFlushToZero,
        });
        builder.capability(Capability::RoundingModeRTE);
        builder.execution_mode(entry, mode, [width]);
        builder.execution_mode(entry, ExecutionMode::RoundingModeRTE, [width]);
        builder.module().assemble()
    }

    fn properties() -> DeviceProperties {
        DeviceProperties {
            shader_signed_zero_inf_nan_preserve_float64: Some(true),
            shader_denorm_preserve_float64: Some(true),
            shader_denorm_preserve_float16: Some(true),
            shader_rounding_mode_rte_float64: Some(true),
            shader_rounding_mode_rte_float16: Some(true),
            denorm_behavior_independence: Some(ShaderFloatControlsIndependence::Float32Only),
            rounding_mode_independence: Some(ShaderFloatControlsIndependence::Float32Only),
            ..Default::default()
        }
    }

    #[test]
    fn rejects_unsupported_policies() {
        let props = properties();
        let strict = shader(ExecutionMode::DenormPreserve, 64);
        let error = prepare_float_controls_with_properties(&strict, "strict", false, &props, false)
            .unwrap_err();
        assert!(error.to_string().contains("shaderFloatControls2"));
        let fast = shader(ExecutionMode::DenormFlushToZero, 64);
        let error =
            prepare_float_controls_with_properties(&fast, "fast", true, &props, false).unwrap_err();
        assert!(error.to_string().contains("DenormFlushToZero"));
    }

    #[test]
    fn macos_keeps_arithmetic_restrictions_without_unsupported_modes() {
        let shader = shader(ExecutionMode::DenormPreserve, 32);
        // MoltenVK 1.4.2 supports RTE for f16/f32, but couples all widths and
        // does not support f64 RTE or either explicit subnormal mode.
        let props = DeviceProperties {
            shader_signed_zero_inf_nan_preserve_float32: Some(true),
            shader_rounding_mode_rte_float16: Some(true),
            shader_rounding_mode_rte_float32: Some(true),
            denorm_behavior_independence: Some(ShaderFloatControlsIndependence::None),
            rounding_mode_independence: Some(ShaderFloatControlsIndependence::None),
            ..Default::default()
        };
        assert!(
            prepare_float_controls_with_properties(&shader, "strict", true, &props, false).is_err()
        );
        let words =
            prepare_float_controls_with_properties(&shader, "strict", true, &props, true).unwrap();
        let module = crate::spirv::load_words(words).unwrap();
        let original = crate::spirv::load_words(&shader).unwrap();
        assert_eq!(
            module.execution_modes,
            original
                .execution_modes
                .into_iter()
                .filter(|inst| inst.operands[1]
                    == Operand::ExecutionMode(ExecutionMode::FPFastMathDefault))
                .collect::<Vec<_>>()
        );
        assert_eq!(module.types_global_values, original.types_global_values);
        assert_eq!(module.capabilities.len(), 1);
        assert_eq!(
            module.capabilities[0].operands[0],
            Operand::Capability(Capability::FloatControls2)
        );
        // macOS must still support the restrictions on arithmetic and special values.
        assert!(
            prepare_float_controls_with_properties(&shader, "strict", false, &props, true).is_err()
        );
        let props = DeviceProperties {
            shader_signed_zero_inf_nan_preserve_float32: Some(false),
            ..props
        };
        assert!(
            prepare_float_controls_with_properties(&shader, "strict", true, &props, true).is_err()
        );
    }

    #[test]
    fn macos_retains_supported_modes() {
        let shader = shader(ExecutionMode::DenormPreserve, 64);
        let props = properties();
        let strict =
            prepare_float_controls_with_properties(&shader, "strict", true, &props, false).unwrap();
        assert_eq!(
            prepare_float_controls_with_properties(&shader, "strict", true, &props, true).unwrap(),
            strict
        );
        let props = DeviceProperties {
            shader_denorm_preserve_float16: Some(false),
            ..props
        };
        let words =
            prepare_float_controls_with_properties(&shader, "strict", true, &props, true).unwrap();
        let module = crate::spirv::load_words(words).unwrap();
        // Unsupported f16 preservation also removes the coupled f64 mode,
        // without dropping the supported rounding modes.
        assert!(!module
            .execution_modes
            .iter()
            .any(|inst| inst.operands[1] == Operand::ExecutionMode(ExecutionMode::DenormPreserve)));
        assert_eq!(
            module
                .execution_modes
                .iter()
                .filter(|inst| inst.operands[1]
                    == Operand::ExecutionMode(ExecutionMode::RoundingModeRTE))
                .count(),
            2
        );
    }

    #[test]
    fn completes_companion_modes_without_adding_float_types() {
        let words = prepare_float_controls_with_properties(
            &shader(ExecutionMode::DenormPreserve, 64),
            "strict",
            true,
            &properties(),
            false,
        )
        .unwrap();
        let module = crate::spirv::load_words(words).unwrap();
        assert!(module.execution_modes.iter().any(|mode| mode.operands[1..]
            == [
                Operand::ExecutionMode(ExecutionMode::DenormPreserve),
                Operand::LiteralBit32(16)
            ]));
        assert!(!module
            .types_global_values
            .iter()
            .any(|inst| inst.class.opcode == Op::TypeFloat
                && inst.operands[0] == Operand::LiteralBit32(16)));
    }
}

pub(super) struct Kernel {
    engine: Arc<Engine>,
    desc: Arc<KernelDesc>,
    compute_pipeline: Arc<ComputePipeline>,
}

impl DeviceEngineKernel for Kernel {
    type Engine = Engine;
    type DeviceBuffer = DeviceBuffer;
    fn cached(
        engine: Arc<Self::Engine>,
        key: KernelKey,
        desc_fn: impl FnOnce() -> Result<Arc<KernelDesc>>,
    ) -> Result<Arc<Self>> {
        let KernelInner {
            desc,
            compute_pipeline,
        } = engine
            .kernels
            .entry(key)
            .or_try_insert_with(|| KernelInner::new(&engine, desc_fn()?))?
            .clone();
        Ok(Arc::new(Kernel {
            engine,
            desc,
            compute_pipeline,
        }))
    }
    fn engine(&self) -> &Arc<Self::Engine> {
        &self.engine
    }
    unsafe fn dispatch(
        &self,
        groups: u32,
        buffers: &[Arc<Self::DeviceBuffer>],
        push_consts: Vec<u8>,
        debug_printf_panic: Option<Arc<AtomicBool>>,
    ) -> Result<()> {
        let engine = &self.engine;
        if let Some(epoch) = buffers.iter().map(|x| x.epoch.load(Ordering::SeqCst)).max() {
            engine.wait_pending(epoch)?;
        }
        unsafe {
            engine.compute(
                &self.desc,
                &self.compute_pipeline,
                groups,
                buffers,
                &push_consts,
                debug_printf_panic,
            )
        }
    }
    fn desc(&self) -> &Arc<KernelDesc> {
        &self.desc
    }
}
