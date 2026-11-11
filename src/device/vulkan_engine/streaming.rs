use super::*;
use crate::stream::{Progress, StreamingOptions};
use parking_lot::Condvar;
use std::{collections::VecDeque, task::Waker};

pub(crate) type CompletionPoint = Arc<Completion>;

pub(crate) struct Completion {
    semaphore: Arc<Semaphore>,
    value: u64,
    submitted: AtomicBool,
    result: Mutex<Option<std::result::Result<(), Arc<str>>>>,
    waiters: Mutex<Vec<(u64, Waker)>>,
    next_waiter: AtomicU64,
}

impl Completion {
    fn new(semaphore: Arc<Semaphore>, value: u64, waiter_capacity: usize) -> Self {
        Self {
            semaphore,
            value,
            submitted: AtomicBool::new(false),
            result: Mutex::new(None),
            waiters: Mutex::new(Vec::with_capacity(waiter_capacity)),
            next_waiter: AtomicU64::new(0),
        }
    }
    pub(crate) fn result(&self) -> Result<bool> {
        match &*self.result.lock() {
            Some(Ok(())) => Ok(true),
            Some(Err(error)) => bail!("{error}"),
            None => Ok(false),
        }
    }
    fn gpu_complete(&self) -> Result<bool> {
        if !self.submitted.load(Ordering::Acquire) {
            return self.result();
        }
        match unsafe { poll_semaphore(self.semaphore.device(), &self.semaphore, self.value) } {
            ash::vk::Result::SUCCESS => Ok(true),
            ash::vk::Result::TIMEOUT => Ok(false),
            error => Err(error.into()),
        }
    }
    fn finish(&self, result: std::result::Result<(), Arc<str>>, wake: &mut Vec<Waker>) {
        *self.result.lock() = Some(result);
        wake.extend(self.waiters.lock().drain(..).map(|(_, waker)| waker));
    }
    pub(crate) fn register(&self, id: &mut Option<u64>, waker: &Waker) {
        let id = *id.get_or_insert_with(|| self.next_waiter.fetch_add(1, Ordering::Relaxed));
        let mut waiters = self.waiters.lock();
        if let Some((_, old)) = waiters.iter_mut().find(|(key, _)| *key == id) {
            old.clone_from(waker);
        } else {
            waiters.push((id, waker.clone()));
        }
    }
    pub(crate) fn unregister(&self, id: u64) {
        self.waiters.lock().retain(|(key, _)| *key != id);
    }
}

pub(crate) struct RawStream {
    semaphore: Arc<Semaphore>,
    queue: usize,
    transfer_only: bool,
    next: u64,
    tail: Option<CompletionPoint>,
    waits: Vec<CompletionPoint>,
}

impl RawStream {
    pub(super) fn new(
        queue: usize,
        transfer_only: bool,
        device: &Arc<Device>,
        capacity: usize,
    ) -> Result<Self> {
        Ok(Self {
            semaphore: Arc::new(new_semaphore(device)?),
            queue,
            transfer_only,
            next: 0,
            tail: None,
            waits: Vec::with_capacity(capacity),
        })
    }
    pub(crate) fn wait_event(&mut self, point: &CompletionPoint) {
        push_dependency(&mut self.waits, point);
    }
    pub(crate) fn reserve_dependencies(&mut self, additional: usize) {
        self.waits.reserve(additional);
    }
}

fn push_dependency(dependencies: &mut Vec<CompletionPoint>, point: &CompletionPoint) {
    if let Some(old) = dependencies
        .iter_mut()
        .find(|old| Arc::ptr_eq(&old.semaphore, &point.semaphore))
    {
        if old.value < point.value {
            *old = point.clone();
        }
    } else {
        dependencies.push(point.clone());
    }
}

pub(super) enum Command {
    Copy(Subbuffer<[u8]>, Subbuffer<[u8]>),
    Dispatch {
        desc: Arc<KernelDesc>,
        pipeline: Arc<ComputePipeline>,
        groups: u32,
        buffers: Vec<Subbuffer<[u8]>>,
        push_consts: Vec<u8>,
    },
    Marker,
}

struct Job {
    command: Command,
    point: CompletionPoint,
    dependencies: Vec<CompletionPoint>,
    queue: usize,
}

struct Submitted {
    frame: Frame,
    points: Vec<CompletionPoint>,
    dependencies: Vec<CompletionPoint>,
    queue: usize,
    panicked: Arc<AtomicBool>,
    messenger: Option<DebugUtilsMessenger>,
}

pub(super) struct Runtime {
    pending: VecDeque<Job>,
    submitted: Vec<Submitted>,
    free: Vec<Vec<Frame>>,
    failure: Option<Arc<str>>,
    failed_frame: Option<Frame>,
    last_debug: Option<CompletionPoint>,
    next_compute: usize,
    next_transfer: usize,
}

impl Runtime {
    pub(super) fn new(queues: &[Arc<Queue>], options: StreamingOptions) -> Result<Self> {
        anyhow::ensure!(
            options.batch_size > 0,
            "Streaming batch size must be nonzero."
        );
        let mut free: Vec<Vec<Frame>> = (0..queues.len())
            .map(|queue| {
                Vec::with_capacity(
                    options.capacity / queues.len()
                        + usize::from(queue < options.capacity % queues.len()),
                )
            })
            .collect();
        for index in 0..options.capacity {
            let queue = index % queues.len();
            free[queue].push(Frame::new(queues[queue].clone()));
        }
        Ok(Self {
            pending: VecDeque::with_capacity(options.capacity),
            submitted: Vec::with_capacity(options.capacity),
            free,
            failure: None,
            failed_frame: None,
            last_debug: None,
            next_compute: 0,
            next_transfer: 0,
        })
    }
}

#[derive(Default)]
pub(super) struct Wake {
    notified: Mutex<bool>,
    changed: Condvar,
}

impl Wake {
    pub(super) fn notify(&self) {
        *self.notified.lock() = true;
        self.changed.notify_all();
    }
    fn wait(&self, active: bool) {
        let mut notified = self.notified.lock();
        if !*notified {
            if active {
                self.changed
                    .wait_for(&mut notified, Duration::from_micros(100));
            } else {
                self.changed.wait(&mut notified);
            }
        }
        *notified = false;
    }
}

impl Engine {
    pub(super) fn start(self: &Arc<Self>) {
        let engine = Arc::downgrade(self);
        let wake = self.wake.clone();
        std::thread::spawn(move || loop {
            let Some(engine) = engine.upgrade() else {
                break;
            };
            let _ = engine.progress(engine.options.progress == Progress::Background);
            let active = {
                let state = engine.runtime.lock();
                state.failure.is_none()
                    && (!state.pending.is_empty()
                        || (engine.options.progress == Progress::Background
                            && !state.submitted.is_empty()))
            };
            drop(engine);
            wake.wait(active);
        });
    }

    pub(crate) fn stream(&self, transfer_only: bool, capacity: usize) -> Result<RawStream> {
        let mut state = self.runtime.lock();
        let queues = if transfer_only {
            &self.transfer_queues
        } else {
            &self.compute_queues
        };
        let next = if transfer_only {
            &mut state.next_transfer
        } else {
            &mut state.next_compute
        };
        let queue = queues[*next % queues.len()];
        *next = next.wrapping_add(1);
        RawStream::new(queue, transfer_only, self.queue.device(), capacity)
    }

    pub(crate) fn record_event(&self, stream: &mut RawStream) -> Result<CompletionPoint> {
        self.enqueue(stream, Command::Marker, &[])
    }

    pub(super) fn enqueue(
        &self,
        stream: &mut RawStream,
        command: Command,
        uses: &[(&DeviceBuffer, bool)],
    ) -> Result<CompletionPoint> {
        let result = self.accept(&mut self.runtime.lock(), stream, command, uses);
        self.wake.notify();
        result
    }

    fn accept(
        &self,
        state: &mut Runtime,
        stream: &mut RawStream,
        command: Command,
        uses: &[(&DeviceBuffer, bool)],
    ) -> Result<CompletionPoint> {
        anyhow::ensure!(
            !stream.transfer_only || !matches!(command, Command::Dispatch { .. }),
            "A transfer stream cannot dispatch kernels."
        );
        if let Some(error) = &state.failure {
            bail!("{error}");
        }
        state.pending.try_reserve(1)?;
        let mut uses = uses.to_vec();
        uses.sort_unstable_by_key(|(buffer, _)| Arc::as_ptr(&buffer.access));
        let mut merged: Vec<(&DeviceBuffer, bool)> = Vec::with_capacity(uses.len());
        for (buffer, write) in uses {
            anyhow::ensure!(
                std::ptr::eq(buffer.engine.as_ref(), self),
                "The buffer belongs to a different device."
            );
            if let Some((_, previous_write)) = merged
                .last_mut()
                .filter(|(last, _)| Arc::ptr_eq(&last.access, &buffer.access))
            {
                *previous_write |= write;
            } else {
                merged.push((buffer, write));
            }
        }
        let mut access: Vec<_> = merged
            .iter()
            .map(|(buffer, write)| (buffer.access.lock(), *write))
            .collect();
        let mut dependencies =
            Vec::with_capacity(self.options.dependency_capacity.max(stream.waits.len()));
        if let Some(tail) = &stream.tail {
            push_dependency(&mut dependencies, tail);
        }
        for point in &stream.waits {
            push_dependency(&mut dependencies, point);
        }
        for (access, write) in &access {
            if access.failed {
                bail!("A mapped-memory cache operation failed.");
            }
            if access.writer || access.readers != 0 {
                return Err(MapError::Conflict.into());
            }
            if let Some(point) = &access.gpu_writer {
                push_dependency(&mut dependencies, point);
            }
            if *write {
                for point in &access.gpu_readers {
                    push_dependency(&mut dependencies, point);
                }
            }
        }
        let debug = self.info.debug_printf && matches!(command, Command::Dispatch { .. });
        if debug {
            if let Some(point) = &state.last_debug {
                push_dependency(&mut dependencies, point);
            }
        }
        // Keep completed writers for memory visibility across queues.
        for dependency in &dependencies {
            dependency.result()?;
        }
        let value = stream
            .next
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("The stream timeline is exhausted."))?;
        let point = Arc::new(Completion::new(
            stream.semaphore.clone(),
            value,
            self.options.waiter_capacity,
        ));
        for (access, write) in &mut access {
            access
                .gpu_readers
                .retain(|point| !matches!(*point.result.lock(), Some(Ok(()))));
            if *write {
                access.gpu_writer = Some(point.clone());
                access.gpu_readers.clear();
            } else {
                push_dependency(&mut access.gpu_readers, &point);
            }
        }
        if debug {
            state.last_debug = Some(point.clone());
        }
        state.pending.push_back(Job {
            command,
            point: point.clone(),
            dependencies,
            queue: stream.queue,
        });
        stream.next = value;
        stream.tail = Some(point.clone());
        stream.waits.clear();
        drop(access);
        Ok(point)
    }

    fn progress(&self, retire: bool) -> Result<()> {
        let mut wake = Vec::new();
        let mut state = self.runtime.lock();
        if let Some(error) = &state.failure {
            bail!("{error}");
        }
        let result = self.progress_locked(&mut state, retire, &mut wake);
        if let Err(error) = &result {
            let error: Arc<str> = format!("{error:#}").into();
            state.failure = Some(error.clone());
            self.failed.store(true, Ordering::Release);
            for job in state.pending.drain(..) {
                job.point.finish(Err(error.clone()), &mut wake);
            }
            for submitted in &state.submitted {
                for point in &submitted.points {
                    point.finish(Err(error.clone()), &mut wake);
                }
            }
        }
        drop(state);
        for waker in wake {
            waker.wake();
        }
        result
    }

    fn progress_locked(
        &self,
        state: &mut Runtime,
        retire: bool,
        wake: &mut Vec<Waker>,
    ) -> Result<()> {
        if let Some(error) = &state.failure {
            bail!("{error}");
        }
        if retire {
            let mut index = 0;
            while index < state.submitted.len() {
                let submitted = &state.submitted[index];
                if !submitted.points.last().unwrap().gpu_complete()?
                    || submitted
                        .dependencies
                        .iter()
                        .any(|point| point.result.lock().is_none())
                {
                    index += 1;
                    continue;
                }
                let mut submitted = state.submitted.remove(index);
                drop(submitted.messenger.take());
                let result = if submitted.panicked.load(Ordering::Acquire) {
                    Err(Arc::<str>::from("The GPU kernel panicked."))
                } else {
                    submitted
                        .dependencies
                        .iter()
                        .find_map(|point| match &*point.result.lock() {
                            Some(Err(error)) => Some(Err(error.clone())),
                            _ => None,
                        })
                        .unwrap_or(Ok(()))
                };
                for point in &submitted.points {
                    point.finish(result.clone(), wake);
                }
                submitted.frame.finish();
                state.free[submitted.queue].push(submitted.frame);
            }
        }
        loop {
            // Submit producers first. Prefer work whose GPU dependencies have finished.
            let ready = state.pending.iter().position(|job| {
                let debug =
                    self.info.debug_printf && matches!(job.command, Command::Dispatch { .. });
                job.dependencies.iter().all(|point| {
                    if debug {
                        point.result.lock().is_some()
                    } else {
                        point.result.lock().is_some() || point.gpu_complete().unwrap_or(false)
                    }
                })
            });
            let ready = ready.or_else(|| {
                state.pending.iter().position(|job| {
                    let dispatch = matches!(job.command, Command::Dispatch { .. });
                    job.dependencies.iter().all(|point| {
                        point.result.lock().is_some()
                            || (!self.info.debug_printf || !dispatch)
                                && point.submitted.load(Ordering::Acquire)
                                && (dispatch || Arc::ptr_eq(&point.semaphore, &job.point.semaphore))
                    })
                })
            });
            let Some(index) = ready else { break };
            let job = state.pending.remove(index).unwrap();
            if let Some(error) =
                job.dependencies
                    .iter()
                    .find_map(|point| match &*point.result.lock() {
                        Some(Err(error)) => Some(error.clone()),
                        _ => None,
                    })
            {
                job.point.finish(Err(error), wake);
                continue;
            }
            let capacity = (state.pending.len() + 1).min(self.options.batch_size);
            let mut points = Vec::with_capacity(capacity);
            points.push(job.point.clone());
            let mut commands = Vec::with_capacity(capacity);
            let debug = self.info.debug_printf && matches!(job.command, Command::Dispatch { .. });
            // Fold adjacent work on one timeline only when it adds no external waits.
            if !debug {
                while points.len() < self.options.batch_size {
                    let Some(next) = state
                        .pending
                        .iter()
                        .position(|next| Arc::ptr_eq(&next.point.semaphore, &job.point.semaphore))
                    else {
                        break;
                    };
                    let candidate = &state.pending[next];
                    if (self.info.debug_printf
                        && matches!(candidate.command, Command::Dispatch { .. }))
                        || !candidate.dependencies.iter().all(|dependency| {
                            (Arc::ptr_eq(&dependency.semaphore, &job.point.semaphore)
                                && dependency.value <= points.last().unwrap().value)
                                || job.dependencies.iter().any(|prior| {
                                    Arc::ptr_eq(&prior.semaphore, &dependency.semaphore)
                                        && prior.value >= dependency.value
                                })
                        })
                    {
                        break;
                    }
                    let next = state.pending.remove(next).unwrap();
                    commands.push(next.command);
                    points.push(next.point);
                }
            }
            let result: Result<Submitted> = (|| {
                let mut frame = match state.free[job.queue].pop() {
                    Some(frame) => frame,
                    None => Frame::new(self.queues[job.queue].clone()),
                };
                let panicked = Arc::new(AtomicBool::new(false));
                let messenger = if self.info.debug_printf {
                    if let Command::Dispatch { desc, .. } = &job.command {
                        let name = desc.name.clone();
                        let panicked = panicked.clone();
                        Some(unsafe {
                            DebugUtilsMessenger::new(
                                self.queue.device().instance().clone(),
                                DebugUtilsMessengerCreateInfo {
                                    message_severity: DebugUtilsMessageSeverity::INFO,
                                    message_type: DebugUtilsMessageType::VALIDATION,
                                    ..DebugUtilsMessengerCreateInfo::user_callback(
                                        DebugUtilsMessengerCallback::new(move |_, _, msg| {
                                            if msg
                                                .message_id_name
                                                .is_some_and(|name| name.contains("DEBUG-PRINTF"))
                                            {
                                                eprintln!("[{name}] {}", msg.message);
                                                if msg.message.contains("[Rust panicked at ") {
                                                    panicked.store(true, Ordering::Release);
                                                }
                                            }
                                        }),
                                    )
                                },
                            )?
                        })
                    } else {
                        None
                    }
                } else {
                    None
                };
                unsafe {
                    frame.record(std::iter::once(job.command).chain(commands))?;
                }
                let result = unsafe {
                    self.queues[job.queue].with(|mut guard| {
                        queue_submit(
                            &self.queues[job.queue],
                            &mut guard,
                            frame.command_buffer.as_ref().unwrap(),
                            &job.point.semaphore,
                            points.last().unwrap().value,
                            &job.dependencies,
                        )
                    })
                };
                if let Err(error) = result {
                    // Retain resources until queue teardown if submission fails.
                    state.failed_frame = Some(frame);
                    return Err(error.into());
                }
                for point in &points {
                    point.submitted.store(true, Ordering::Release);
                }
                Ok(Submitted {
                    frame,
                    points: std::mem::take(&mut points),
                    dependencies: job.dependencies,
                    queue: job.queue,
                    panicked,
                    messenger,
                })
            })();
            match result {
                Ok(submitted) => state.submitted.push(submitted),
                Err(error) => {
                    let message: Arc<str> = format!("{error:#}").into();
                    for point in points {
                        point.finish(Err(message.clone()), wake);
                    }
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn poll(&self) -> Result<()> {
        self.progress(true)
    }

    pub(crate) fn wait_point(&self, point: &CompletionPoint) -> Result<()> {
        loop {
            if point.result()? {
                return Ok(());
            }
            let progress = self.poll();
            if point.result()? {
                return Ok(());
            }
            progress?;
            self.wake.wait(true);
        }
    }

    pub(super) fn wait_all(&self) -> Result<()> {
        let points: Vec<_> = {
            let state = self.runtime.lock();
            state
                .pending
                .iter()
                .map(|job| job.point.clone())
                .chain(
                    state
                        .submitted
                        .iter()
                        .flat_map(|job| job.points.iter().cloned()),
                )
                .collect()
        };
        for point in points {
            self.wait_point(&point)?;
        }
        self.poll()
    }

    pub(super) fn blocking_copy(
        &self,
        src: Subbuffer<[u8]>,
        dst: Subbuffer<[u8]>,
        uses: &[(&DeviceBuffer, bool)],
    ) -> Result<()> {
        let point = self.enqueue(
            &mut self.default_stream.lock(),
            Command::Copy(src, dst),
            uses,
        )?;
        self.wait_point(&point)
    }

    pub(crate) fn copy_on(
        &self,
        stream: &mut RawStream,
        src: &DeviceBuffer,
        dst: &DeviceBuffer,
    ) -> Result<CompletionPoint> {
        anyhow::ensure!(src.len == dst.len, "Copy lengths differ.");
        if Arc::ptr_eq(&src.access, &dst.access) && src.len != 0 {
            anyhow::ensure!(
                src.offset + src.len <= dst.offset || dst.offset + dst.len <= src.offset,
                "Copy ranges overlap."
            );
        }
        let command = match (src.slice_inner(), dst.slice_inner()) {
            (Some(src), Some(dst)) => Command::Copy(src, dst),
            _ => Command::Marker,
        };
        self.enqueue(stream, command, &[(src, false), (dst, true)])
    }
}

unsafe fn queue_submit(
    queue: &Queue,
    _guard: &mut QueueGuard,
    command_buffer: &UnsafeCommandBuffer,
    semaphore: &Semaphore,
    value: u64,
    dependencies: &[CompletionPoint],
) -> Result<(), ash::vk::Result> {
    let commands = [command_buffer.handle()];
    let signals = [semaphore.handle()];
    let values = [value];
    let waits: Vec<_> = dependencies
        .iter()
        .map(|point| point.semaphore.handle())
        .collect();
    let wait_values: Vec<_> = dependencies.iter().map(|point| point.value).collect();
    let stages = vec![ash::vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
    let mut timeline = ash::vk::TimelineSemaphoreSubmitInfo::default()
        .signal_semaphore_values(&values)
        .wait_semaphore_values(&wait_values);
    let submit = ash::vk::SubmitInfo::default()
        .command_buffers(&commands)
        .signal_semaphores(&signals)
        .wait_semaphores(&waits)
        .wait_dst_stage_mask(&stages)
        .push_next(&mut timeline);
    unsafe {
        (queue.device().fns().v1_0.queue_submit)(queue.handle(), 1, &submit, ash::vk::Fence::null())
            .result()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{device::RawDevice, stream::Event};
    use std::{
        future::Future,
        pin::pin,
        task::{Context, Poll, Wake as TaskWake},
    };

    fn engine() -> Result<Arc<Engine>> {
        Engine::new(DeviceOptions {
            index: 0,
            optimal_features: Features::all(),
            streaming: StreamingOptions {
                progress: Progress::Manual,
                ..Default::default()
            },
        })
    }

    #[test]
    fn batches_preserve_events_and_split_at_external_waits() -> Result<()> {
        let _device_lock = crate::device::TEST_DEVICE_LOCK.lock();
        let engine = engine()?;
        let mut first = engine.stream(false, 1)?;
        let mut second = engine.stream(false, 1)?;
        // Force one physical queue to check producer-before-consumer submission.
        second.queue = first.queue;
        let mut state = engine.runtime.lock();
        let a = engine.accept(&mut state, &mut first, Command::Marker, &[])?;
        let b = engine.accept(&mut state, &mut first, Command::Marker, &[])?;
        engine.progress_locked(&mut state, false, &mut Vec::new())?;
        assert_eq!(state.submitted.len(), 1);
        assert_eq!(state.submitted[0].points.len(), 2);
        second.wait_event(&b);
        let c = engine.accept(&mut state, &mut second, Command::Marker, &[])?;
        first.wait_event(&c);
        let d = engine.accept(&mut state, &mut first, Command::Marker, &[])?;
        drop(state);
        engine.wait_point(&d)?;
        assert!(a.result()? && b.result()? && c.result()?);
        Ok(())
    }

    #[test]
    fn recording_failure_resolves_accepted_events() -> Result<()> {
        let _device_lock = crate::device::TEST_DEVICE_LOCK.lock();
        let engine = engine()?;
        let buffer = unsafe { DeviceBuffer::uninit(engine.clone(), 4, MemKind::Shared)? };
        let mut stream = engine.stream(false, 1)?;
        let mut state = engine.runtime.lock();
        // Bypass public overlap checks to inject a recording error before submission.
        let raw = buffer.slice_inner().unwrap();
        let failed = engine.accept(
            &mut state,
            &mut stream,
            Command::Copy(raw.clone(), raw),
            &[],
        )?;
        let dependent = engine.accept(&mut state, &mut stream, Command::Marker, &[])?;
        drop(state);
        assert!(engine.wait_point(&failed).is_err());
        assert!(engine.wait_point(&dependent).is_err());
        assert!(engine.record_event(&mut stream).is_err());
        Ok(())
    }

    #[test]
    fn aliased_copy_merges_access_and_rejects_overlap() -> Result<()> {
        let _device_lock = crate::device::TEST_DEVICE_LOCK.lock();
        let engine = engine()?;
        let buffer = Arc::new(unsafe { DeviceBuffer::uninit(engine.clone(), 8, MemKind::Shared)? });
        buffer.upload(&[1, 2, 3, 4, 0, 0, 0, 0])?;
        let src = buffer.slice(0..4).unwrap();
        let dst = buffer.slice(4..8).unwrap();
        let mut stream = engine.stream(true, 1)?;
        engine.wait_point(&engine.copy_on(&mut stream, &src, &dst)?)?;
        let mapping = buffer.map(false, true)?;
        assert_eq!(
            unsafe { std::slice::from_raw_parts(mapping.ptr(), 8) },
            &[1, 2, 3, 4, 1, 2, 3, 4]
        );
        drop(mapping);
        assert!(engine
            .copy_on(&mut stream, &src, &buffer.slice(2..6).unwrap())
            .is_err());
        assert!(buffer.map(true, false).is_ok());
        Ok(())
    }

    #[derive(Default)]
    struct CountWake(AtomicU64);
    impl TaskWake for CountWake {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn async_waiters_replace_wakers_and_unregister_on_drop() -> Result<()> {
        let _device_lock = crate::device::TEST_DEVICE_LOCK.lock();
        let engine = engine()?;
        let stream = engine.stream(false, 1)?;
        let point = Arc::new(Completion::new(stream.semaphore, 1, 2));
        let event = Event {
            device: RawDevice { engine },
            point: point.clone(),
        };
        let old = Arc::new(CountWake::default());
        let current = Arc::new(CountWake::default());
        let old_waker = Waker::from(old.clone());
        let waker = Waker::from(current.clone());
        let mut first = pin!(event.wait_async());
        assert!(first
            .as_mut()
            .poll(&mut Context::from_waker(&old_waker))
            .is_pending());
        assert!(first
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending());
        let mut second = Box::pin(event.wait_async());
        assert!(second
            .as_mut()
            .poll(&mut Context::from_waker(&old_waker))
            .is_pending());
        drop(second);
        let mut wake = Vec::new();
        point.finish(Err("injected completion failure".into()), &mut wake);
        for waker in wake {
            waker.wake();
        }
        assert_eq!(old.0.load(Ordering::Relaxed), 0);
        assert_eq!(current.0.load(Ordering::Relaxed), 1);
        assert!(matches!(
            first.as_mut().poll(&mut Context::from_waker(&waker)),
            Poll::Ready(Err(_))
        ));
        assert!(point.waiters.lock().is_empty());
        Ok(())
    }
}
