use super::{
    error::{DeviceIndexOutOfRange, DeviceUnavailable, OutOfDeviceMemory},
    DeviceEngine, DeviceEngineBuffer, DeviceEngineKernel, DeviceId, DeviceInfo, DeviceLost,
    DeviceOptions, Features, KernelDesc, KernelKey,
};

use crate::buffer::{MapError, MemKind, MemoryProperties};
use crate::stream::StreamingOptions;
mod streaming;
use anyhow::{bail, Error, Result};
use ash::vk::Handle;
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use dashmap::DashMap;
use parking_lot::Mutex;
use std::{
    mem::MaybeUninit,
    ops::Range,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
use streaming::{Command, Runtime, Wake};
pub(crate) use streaming::{CompletionPoint, RawStream};
use vulkano::{
    buffer::BufferMemory,
    memory::{MappedMemoryRange, MemoryPropertyFlags},
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
    sync::{semaphore::Semaphore, AccessFlags, DependencyInfo, MemoryBarrier, PipelineStages},
    VulkanObject,
};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct MoltenVkLoader;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
// MoltenVK is linked for the process lifetime and provides the Vulkan C ABI.
unsafe impl vulkano::library::Loader for MoltenVkLoader {
    unsafe fn get_instance_proc_addr(
        &self,
        instance: ash::vk::Instance,
        name: *const std::ffi::c_char,
    ) -> ash::vk::PFN_vkVoidFunction {
        unsafe { moltenvk_static::vkGetInstanceProcAddr(instance.as_raw() as *mut _, name) }
    }
}

pub struct Engine {
    info: Arc<DeviceInfo>,
    host_buffer_sender: Sender<Subbuffer<[u8]>>,
    host_buffer_receiver: Receiver<Subbuffer<[u8]>>,
    kernels: DashMap<KernelKey, KernelInner>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    queue: Arc<Queue>,
    queues: Vec<Arc<Queue>>,
    compute_queues: Vec<usize>,
    transfer_queues: Vec<usize>,
    families: Vec<u32>,
    runtime: Mutex<Runtime>,
    default_stream: Mutex<RawStream>,
    pub(super) options: StreamingOptions,
    wake: Arc<Wake>,
    failed: AtomicBool,
    _instance: Arc<Instance>,
}

impl Engine {
    fn host_buffer(&self) -> Result<HostBuffer<'_>> {
        loop {
            if self.failed.load(Ordering::Acquire) {
                return Err(DeviceLost(self.id()).into());
            }
            match self
                .host_buffer_receiver
                .recv_timeout(Duration::from_millis(10))
            {
                Ok(buffer) => {
                    return Ok(HostBuffer {
                        buffer: Some(buffer),
                        engine: self,
                    })
                }
                Err(RecvTimeoutError::Timeout) => (),
                Err(RecvTimeoutError::Disconnected) => return Err(DeviceLost(self.id()).into()),
            }
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.wait_all();
        for queue in &self.queues {
            let _ = queue.with(|mut queue| queue.wait_idle());
        }
        self.wake.notify();
    }
}

impl DeviceEngine for Engine {
    type DeviceBuffer = DeviceBuffer;
    type Kernel = Kernel;
    fn new(options: DeviceOptions) -> anyhow::Result<std::sync::Arc<Self>> {
        let DeviceOptions {
            index,
            optimal_features,
            streaming: options,
        } = options;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let library = VulkanLibrary::with_loader(MoltenVkLoader);
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let library = VulkanLibrary::new();
        let library = library.map_err(|e| Error::new(DeviceUnavailable).context(e))?;
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
        anyhow::ensure!(
            device_features.timeline_semaphore,
            "The device does not support timeline semaphores."
        );
        let properties = physical_device.queue_family_properties();
        let transfer_family = properties
            .iter()
            .position(|family| {
                family.queue_flags.contains(QueueFlags::TRANSFER)
                    && !family
                        .queue_flags
                        .intersects(QueueFlags::COMPUTE | QueueFlags::GRAPHICS)
            })
            .map(|family| family as u32)
            .unwrap_or(compute_family);
        let mut families = vec![compute_family];
        if transfer_family != compute_family {
            families.push(transfer_family);
        }
        let queue_create_infos = families
            .iter()
            .map(|&family| QueueCreateInfo {
                queue_family_index: family,
                queues: vec![1.0; properties[family as usize].queue_count.min(2) as usize],
                ..Default::default()
            })
            .collect();
        let (device, queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: device_features,
                queue_create_infos,
                ..Default::default()
            },
        )?;
        let queues: Vec<_> = queues.collect();
        let compute_queues: Vec<_> = queues
            .iter()
            .enumerate()
            .filter_map(|(i, queue)| (queue.queue_family_index() == compute_family).then_some(i))
            .collect();
        let transfer_queues: Vec<_> = queues
            .iter()
            .enumerate()
            .filter_map(|(i, queue)| (queue.queue_family_index() == transfer_family).then_some(i))
            .collect();
        let queue = queues[compute_queues[0]].clone();
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
                sharing: if families.len() > 1 {
                    vulkano::sync::Sharing::Concurrent(families.iter().copied().collect())
                } else {
                    vulkano::sync::Sharing::Exclusive
                },
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
            host_buffer_sender.send(inner).unwrap();
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
        let runtime = Mutex::new(Runtime::new(&queues, options)?);
        let default_stream = Mutex::new(RawStream::new(
            compute_queues[0],
            false,
            &device,
            options.dependency_capacity,
        )?);
        let engine = Arc::new(Self {
            info,
            host_buffer_sender,
            host_buffer_receiver,
            kernels,
            memory_allocator,
            queue,
            queues,
            compute_queues,
            transfer_queues,
            families,
            runtime,
            default_stream,
            options,
            wake: Arc::new(Wake::default()),
            failed: AtomicBool::new(false),
            _instance: instance,
        });
        engine.start();
        Ok(engine)
    }
    fn id(&self) -> DeviceId {
        let index = self.info.index;
        let handle = self.queue.device().handle().as_raw().try_into().unwrap();
        DeviceId { index, handle }
    }
    fn info(&self) -> &Arc<DeviceInfo> {
        &self.info
    }
    fn wait(&self) -> Result<()> {
        self.wait_all()
    }
}

fn new_semaphore(device: &Arc<Device>) -> Result<Semaphore> {
    use vulkano::sync::semaphore::{SemaphoreCreateInfo, SemaphoreType};
    Ok(Semaphore::new(
        device.clone(),
        SemaphoreCreateInfo {
            semaphore_type: SemaphoreType::Timeline,
            ..Default::default()
        },
    )?)
}

unsafe fn poll_semaphore(device: &Device, semaphore: &Semaphore, value: u64) -> ash::vk::Result {
    // Read the counter directly; zero-timeout waits register callbacks in MoltenVK.
    let mut completed = 0;
    let result = unsafe {
        (device.fns().v1_2.get_semaphore_counter_value)(
            device.handle(),
            semaphore.handle(),
            &mut completed,
        )
    };
    if result == ash::vk::Result::SUCCESS && completed < value {
        ash::vk::Result::TIMEOUT
    } else {
        result
    }
}

struct HostBuffer<'a> {
    buffer: Option<Subbuffer<[u8]>>,
    engine: &'a Engine,
}

impl Drop for HostBuffer<'_> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let _ = self.engine.host_buffer_sender.send(buffer);
        }
    }
}

struct Frame {
    queue: Arc<Queue>,
    command_allocator: Arc<StandardCommandBufferAllocator>,
    command_buffer_builder: Option<UnsafeCommandBufferBuilder>,
    descriptor_allocator: Arc<StandardDescriptorSetAllocator>,
    descriptor_sets: Vec<RawDescriptorSet>,
    buffers: Vec<Subbuffer<[u8]>>,
    command_buffer: Option<UnsafeCommandBuffer>,
    pipelines: Vec<Arc<ComputePipeline>>,
}

// SAFETY: Each frame owns a private command allocator, used only by this frame.
// Frame access (including recording, ending and allocation) is exclusive: either
// behind its mutex or owned by the worker. No other frame shares its command pool.
unsafe impl Send for Frame {}

impl Frame {
    fn new(queue: Arc<Queue>) -> Self {
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
        let buffers = Vec::with_capacity(8);
        Self {
            queue,
            command_allocator,
            command_buffer_builder,
            descriptor_allocator,
            descriptor_sets: Vec::with_capacity(1),
            buffers,
            command_buffer: None,
            pipelines: Vec::with_capacity(1),
        }
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
    unsafe fn record(&mut self, commands: impl IntoIterator<Item = Command>) -> Result<()> {
        unsafe {
            self.begin()?;
        }
        for command in commands {
            match command {
                Command::Copy(src, dst) => {
                    unsafe {
                        self.command_buffer_builder
                            .as_mut()
                            .unwrap()
                            .copy_buffer(&CopyBufferInfo::buffers(src.clone(), dst.clone()))?;
                    }
                    self.buffers.extend([src, dst]);
                }
                Command::Dispatch {
                    pipeline,
                    groups,
                    buffers,
                    push_consts,
                    ..
                } => {
                    unsafe {
                        self.compute(&pipeline, groups, &buffers, &push_consts)?;
                    }
                    self.pipelines.push(pipeline);
                }
                Command::Marker => (),
            }
            unsafe {
                self.memory_barrier()?;
            }
        }
        self.command_buffer = Some(unsafe { self.command_buffer_builder.take().unwrap().end()? });
        Ok(())
    }
    unsafe fn compute(
        &mut self,
        pipeline: &Arc<ComputePipeline>,
        groups: u32,
        buffers: &[Subbuffer<[u8]>],
        push_consts: &[u8],
    ) -> Result<()> {
        let builder = self.command_buffer_builder.as_mut().unwrap();
        unsafe {
            builder.bind_pipeline_compute(pipeline)?;
        }
        let pipeline_layout = pipeline.layout();
        if !buffers.is_empty() {
            let descriptor_set_layout = pipeline_layout.set_layouts().first().unwrap();
            let writes: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(binding, x)| WriteDescriptorSet::buffer(binding as u32, x.clone()))
                .collect();
            unsafe {
                let descriptor_set = RawDescriptorSet::new(
                    self.descriptor_allocator.clone(),
                    descriptor_set_layout,
                    0,
                )?;
                descriptor_set.update(&writes, &[])?;
                builder.bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipeline_layout,
                    0,
                    &[&descriptor_set],
                    &[],
                )?;
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
            builder.dispatch([groups, 1, 1])?;
        }
        self.buffers.extend_from_slice(buffers);
        Ok(())
    }
    unsafe fn memory_barrier(&mut self) -> Result<()> {
        // Make writes visible to later copies, kernels, and host reads.
        let stages = PipelineStages::ALL_COMMANDS;
        unsafe {
            self.command_buffer_builder
                .as_mut()
                .unwrap()
                .pipeline_barrier(&DependencyInfo {
                    memory_barriers: [MemoryBarrier {
                        src_stages: stages,
                        src_access: AccessFlags::MEMORY_WRITE,
                        dst_stages: stages | PipelineStages::HOST,
                        dst_access: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                        ..Default::default()
                    }]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                })?;
        }
        Ok(())
    }
    fn finish(&mut self) {
        self.command_buffer.take();
        self.command_buffer_builder.take();
        self.descriptor_sets.clear();
        self.buffers.clear();
        self.pipelines.clear();
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

#[derive(Default)]
struct CpuAccess {
    readers: usize,
    writer: bool,
    failed: bool,
    gpu_writer: Option<CompletionPoint>,
    gpu_readers: Vec<CompletionPoint>,
}

fn memory_filter(kind: MemKind) -> MemoryTypeFilter {
    use MemoryPropertyFlags as F;
    match kind {
        MemKind::Device => MemoryTypeFilter::PREFER_DEVICE,
        MemKind::Shared => MemoryTypeFilter {
            required_flags: F::HOST_VISIBLE,
            preferred_flags: F::DEVICE_LOCAL | F::HOST_CACHED,
            ..MemoryTypeFilter::empty()
        },
        MemKind::Upload => MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        MemKind::Readback => MemoryTypeFilter {
            required_flags: F::HOST_VISIBLE,
            preferred_flags: F::HOST_CACHED,
            not_preferred_flags: F::DEVICE_LOCAL,
        },
    }
}

pub(crate) struct DeviceMapping {
    buffer: DeviceBuffer,
    write: bool,
    ready: bool,
}

impl DeviceMapping {
    pub(crate) fn ptr(&self) -> *mut u8 {
        match &self.buffer.inner {
            Some(inner) => unsafe {
                inner
                    .mapped_slice()
                    .unwrap()
                    .as_ptr()
                    .cast::<u8>()
                    .add(self.buffer.offset)
            },
            None => std::ptr::NonNull::<u64>::dangling().as_ptr().cast(),
        }
    }
}

impl Drop for DeviceMapping {
    fn drop(&mut self) {
        let mut access = self.buffer.access.lock();
        if self.write {
            if self.ready && self.buffer.cache_op(true).is_err() {
                access.failed = true;
            }
            access.writer = false;
        } else {
            access.readers -= 1;
        }
    }
}

impl DeviceBuffer {
    fn cache_op(&self, flush: bool) -> Result<()> {
        let Some(inner) = &self.inner else {
            return Ok(());
        };
        let BufferMemory::Normal(memory) = inner.buffer().memory() else {
            unreachable!()
        };
        let Some(atom) = memory.atom_size() else {
            return Ok(());
        };
        let end = aligned_ceil(
            (inner.offset() + inner.size()) as usize,
            atom.as_devicesize() as usize,
        ) as u64;
        let range = MappedMemoryRange {
            offset: 0,
            size: end.min(memory.size()),
            ..Default::default()
        };
        // SAFETY: The whole allocation is reserved and its GPU accesses have completed.
        unsafe {
            if flush {
                memory.flush_range(range)?;
            } else {
                memory.invalidate_range(range)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
pub(super) struct DeviceBuffer {
    inner: Option<Subbuffer<[u8]>>,
    engine: Arc<Engine>,
    offset: usize,
    len: usize,
    access: Arc<Mutex<CpuAccess>>,
    kind: MemKind,
}

impl DeviceBuffer {
    const MAX_LEN: usize = i32::MAX as usize;
    const MAX_SIZE: usize = aligned_ceil(Self::MAX_LEN, Self::ALIGN);
    const ALIGN: usize = 256;
    const HOST_BUFFER_SIZE: usize = 32_000_000;
    fn slice_inner(&self) -> Option<Subbuffer<[u8]>> {
        if self.len == 0 {
            return None;
        }
        self.inner.as_ref().map(|buffer| {
            buffer
                .clone()
                .slice(self.offset as u64..(self.offset + self.len) as u64)
        })
    }
    fn host_visible(&self) -> bool {
        if let Some(inner) = self.inner.as_ref() {
            inner.mapped_slice().is_ok()
        } else {
            self.kind != MemKind::Device
        }
    }
}

impl DeviceEngineBuffer for DeviceBuffer {
    type Engine = Engine;
    fn engine(&self) -> &Arc<Self::Engine> {
        &self.engine
    }
    fn kind(&self) -> MemKind {
        self.kind
    }
    fn host_visible(&self) -> bool {
        self.host_visible()
    }
    fn memory_properties(&self) -> Option<MemoryProperties> {
        let inner = self.inner.as_ref()?;
        let BufferMemory::Normal(memory) = inner.buffer().memory() else {
            return None;
        };
        let index = memory.device_memory().memory_type_index() as usize;
        let flags = self
            .engine
            .queue
            .device()
            .physical_device()
            .memory_properties()
            .memory_types[index]
            .property_flags;
        Some(MemoryProperties {
            host_visible: flags.contains(MemoryPropertyFlags::HOST_VISIBLE),
            device_local: flags.contains(MemoryPropertyFlags::DEVICE_LOCAL),
            host_cached: flags.contains(MemoryPropertyFlags::HOST_CACHED),
            host_coherent: flags.contains(MemoryPropertyFlags::HOST_COHERENT),
        })
    }
    fn map(&self, write: bool, wait: bool) -> Result<DeviceMapping> {
        if !self.host_visible() {
            return Err(MapError::Unsupported.into());
        }
        let mut access = if wait {
            self.access.lock()
        } else {
            self.access.try_lock().ok_or(MapError::Busy)?
        };
        if access.failed {
            bail!("A mapped-memory cache operation failed.");
        }
        if access.writer || (write && access.readers != 0) {
            return Err(MapError::Conflict.into());
        }
        let points: Vec<_> = access
            .gpu_writer
            .iter()
            .chain(access.gpu_readers.iter())
            .cloned()
            .collect();
        if !wait {
            for point in &points {
                if !point.result()? {
                    return Err(MapError::Busy.into());
                }
            }
        }
        if write {
            access.writer = true;
        } else {
            access.readers += 1;
        }
        drop(access);
        let mut mapping = DeviceMapping {
            buffer: self.clone(),
            write,
            ready: false,
        };
        for point in points {
            self.engine.wait_point(&point)?;
        }
        let mut access = self.access.lock();
        if let Err(error) = self.cache_op(false) {
            access.failed = true;
            return Err(error);
        }
        mapping.ready = true;
        Ok(mapping)
    }
    fn copy(&self, dst: &Self) -> Result<()> {
        if self.len == 0 || (Arc::ptr_eq(&self.access, &dst.access) && self.offset == dst.offset) {
            return Ok(());
        }
        anyhow::ensure!(
            !Arc::ptr_eq(&self.access, &dst.access)
                || self.offset + self.len <= dst.offset
                || dst.offset + dst.len <= self.offset,
            "Copy ranges overlap."
        );
        self.engine.blocking_copy(
            self.slice_inner().unwrap(),
            dst.slice_inner().unwrap(),
            &[(self, false), (dst, true)],
        )
    }
    unsafe fn uninit(engine: Arc<Engine>, len: usize, kind: MemKind) -> Result<Self> {
        use vulkano::{memory::allocator::MemoryAllocatorError, VulkanError};
        let inner = if len > 0 {
            let len = aligned_ceil(len, Self::ALIGN);
            let usage =
                BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC;
            let buffer_info = BufferCreateInfo {
                usage,
                sharing: if engine.families.len() > 1 {
                    vulkano::sync::Sharing::Concurrent(engine.families.iter().copied().collect())
                } else {
                    vulkano::sync::Sharing::Exclusive
                },
                size: len.try_into().unwrap(),
                ..Default::default()
            };
            let allocation_info = AllocationCreateInfo {
                memory_type_filter: memory_filter(kind),
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
        let access = CpuAccess {
            gpu_readers: Vec::with_capacity(engine.options.dependency_capacity),
            ..Default::default()
        };
        Ok(Self {
            inner,
            engine,
            offset: 0,
            len,
            access: Arc::new(Mutex::new(access)),
            kind,
        })
    }
    fn upload(&self, data: &[u8]) -> Result<()> {
        debug_assert_eq!(data.len(), self.len);
        if self.len == 0 {
            return Ok(());
        }
        if self.host_visible() {
            let mapping = self.map(true, true)?;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), mapping.ptr(), data.len());
            }
            return Ok(());
        }
        let buffer = self.slice_inner().unwrap();
        let staging = self.engine.host_buffer()?;
        for (index, chunk) in data.chunks(Self::HOST_BUFFER_SIZE).enumerate() {
            let offset = (index * Self::HOST_BUFFER_SIZE) as u64;
            let host = staging
                .buffer
                .as_ref()
                .unwrap()
                .clone()
                .slice(..chunk.len() as u64);
            host.write()?.copy_from_slice(chunk);
            self.engine.blocking_copy(
                host,
                buffer.clone().slice(offset..offset + chunk.len() as u64),
                &[(self, true)],
            )?;
        }
        Ok(())
    }
    fn download(&self, data: &mut [MaybeUninit<u8>]) -> Result<()> {
        debug_assert_eq!(data.len(), self.len);
        if self.len == 0 {
            return Ok(());
        }
        if self.host_visible() {
            let mapping = self.map(false, true)?;
            unsafe {
                std::ptr::copy_nonoverlapping(mapping.ptr(), data.as_mut_ptr().cast(), data.len());
            }
            return Ok(());
        }
        let buffer = self.slice_inner().unwrap();
        let staging = self.engine.host_buffer()?;
        for (index, chunk) in data.chunks_mut(Self::HOST_BUFFER_SIZE).enumerate() {
            let offset = (index * Self::HOST_BUFFER_SIZE) as u64;
            let host = staging
                .buffer
                .as_ref()
                .unwrap()
                .clone()
                .slice(..chunk.len() as u64);
            self.engine.blocking_copy(
                buffer.clone().slice(offset..offset + chunk.len() as u64),
                host.clone(),
                &[(self, false)],
            )?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host.read()?.as_ptr(),
                    chunk.as_mut_ptr().cast(),
                    chunk.len(),
                );
            }
        }
        Ok(())
    }
    fn transfer(&self, dst: &Self) -> Result<()> {
        debug_assert_eq!(dst.len, self.len);
        if self.len == 0 {
            return Ok(());
        }
        if self.host_visible() {
            let source = self.map(false, true)?;
            return dst.upload(unsafe { std::slice::from_raw_parts(source.ptr(), self.len) });
        }
        if dst.host_visible() {
            let target = dst.map(true, true)?;
            return self.download(unsafe {
                std::slice::from_raw_parts_mut(target.ptr().cast(), self.len)
            });
        }
        // Release each device's staging buffer before acquiring the other's.
        let mut staging = vec![MaybeUninit::uninit(); self.len.min(Self::HOST_BUFFER_SIZE)];
        for offset in (0..self.len).step_by(Self::HOST_BUFFER_SIZE) {
            let len = (self.len - offset).min(staging.len());
            let source = Self {
                offset: self.offset + offset,
                len,
                ..self.clone()
            };
            let target = Self {
                offset: dst.offset + offset,
                len,
                ..dst.clone()
            };
            source.download(&mut staging[..len])?;
            target.upload(unsafe { std::slice::from_raw_parts(staging.as_ptr().cast(), len) })?;
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
        stream: &mut RawStream,
        groups: u32,
        buffers: &[Arc<Self::DeviceBuffer>],
        push_consts: Vec<u8>,
    ) -> Result<CompletionPoint> {
        let uses: Vec<_> = buffers
            .iter()
            .zip(self.desc.slice_descs)
            .map(|(buffer, desc)| (buffer.as_ref(), desc.mutable))
            .collect();
        let command = Command::Dispatch {
            desc: self.desc.clone(),
            pipeline: self.compute_pipeline.clone(),
            groups,
            buffers: buffers
                .iter()
                .map(|buffer| buffer.inner.as_ref().unwrap().clone())
                .collect(),
            push_consts,
        };
        self.engine.enqueue(stream, command, &uses)
    }
    fn desc(&self) -> &Arc<KernelDesc> {
        &self.desc
    }
}
