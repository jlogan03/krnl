use super::*;
use crate::stream::{Event, Progress, StreamingOptions};
use std::{
    future::Future,
    pin::pin,
    sync::{mpsc, Arc},
    task::{Context, Poll, Wake, Waker},
    time::Duration,
};

#[test]
fn streaming_pipeline_reuses_slots() -> Result<()> {
    let _device_lock = crate::device::TEST_DEVICE_LOCK.lock();
    const SLOTS: usize = 3; // Upload, compute, and download can occupy separate slots.
    const CHUNKS: usize = 7; // Reuse every slot and finish with a partial ring.
    const LEN: usize = 257; // Exercise a partial workgroup.
    let device = Device::builder().build()?;
    let mut uploads = device.transfer_stream()?;
    let mut compute = device.stream()?;
    let mut downloads = device.transfer_stream()?;
    let kernel = kernels::cast_u32_u32::builder()?.build(device.clone())?;
    let mut slots = (0..SLOTS)
        .map(|_| {
            Ok((
                Buffer::<u32>::zeros(Mem::Upload(&device), LEN)?,
                Buffer::<u32>::zeros(Mem::Device(&device), LEN)?,
                Buffer::<u32>::zeros(Mem::Device(&device), LEN)?,
                Buffer::<u32>::zeros(Mem::Readback(&device), LEN)?,
                None::<Event>,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    for step in 0..CHUNKS + SLOTS {
        let (upload, input, output, readback, done) = &mut slots[step % SLOTS];
        if let Some(done) = done.take() {
            done.wait()?;
            let values = readback.map_read()?;
            for (index, value) in values.iter().enumerate() {
                assert_eq!(*value, ((step - SLOTS) * LEN + index) as u32);
            }
        }
        if step >= CHUNKS {
            continue;
        }
        for (index, value) in upload.map_write()?.iter_mut().enumerate() {
            *value = (step * LEN + index) as u32;
        }
        let uploaded = uploads.copy(upload.as_slice(), input.as_slice_mut())?;
        compute.wait_event(&uploaded)?;
        let computed = kernel.dispatch(&mut compute, input.as_slice(), output.as_slice_mut())?;
        downloads.wait_event(&computed)?;
        *done = Some(downloads.copy(output.as_slice(), readback.as_slice_mut())?);
    }
    Ok(())
}

#[test]
fn streaming_access_order_and_mapping() -> Result<()> {
    let _device_lock = crate::device::TEST_DEVICE_LOCK.lock();
    let device = Device::builder()
        .streaming(StreamingOptions {
            progress: Progress::Manual,
            ..Default::default()
        })
        .build()?;
    let mut first = device.stream()?;
    let mut second = device.stream()?;
    let mut source = Buffer::from_elem(Mem::Shared(&device), 1, 11u32)?;
    let mut before = Buffer::<u32>::zeros(Mem::Readback(&device), 1)?;
    let mut after = Buffer::<u32>::zeros(Mem::Readback(&device), 1)?;
    let fill = kernels::fill_u32::builder()?.build(device.clone())?;

    first.copy(source.as_slice(), before.as_slice_mut())?;
    assert_eq!(
        source
            .try_map_write()
            .err()
            .unwrap()
            .downcast_ref::<MapError>(),
        Some(&MapError::Busy)
    );
    // The write must follow the earlier read, even though it uses another stream.
    fill.dispatch(&mut second, 29, source.as_slice_mut())?;
    first
        .copy(source.as_slice(), after.as_slice_mut())?
        .wait()?;
    assert_eq!(&*before.map_read()?, &[11]);
    assert_eq!(&*after.map_read()?, &[29]);

    let guard = source.map_read()?;
    let error = first
        .copy(source.as_slice(), after.as_slice_mut())
        .err()
        .unwrap();
    assert_eq!(error.downcast_ref::<MapError>(), Some(&MapError::Conflict));
    drop(guard);
    first
        .copy(source.as_slice(), after.as_slice_mut())?
        .wait()?;
    Ok(())
}

#[test]
fn streaming_kernels_share_readers_and_join_events() -> Result<()> {
    let _device_lock = crate::device::TEST_DEVICE_LOCK.lock();
    let device = Device::builder().build()?;
    let input = Buffer::allocate_with(Mem::Shared(&device), 17, |i| i as u32)?;
    let mut a = Buffer::<u32>::zeros(Mem::Readback(&device), input.len())?;
    let mut b = Buffer::<u32>::zeros(Mem::Readback(&device), input.len())?;
    let mut first = device.stream()?;
    let mut second = device.stream()?;
    let mut join = device.transfer_stream_with_capacity(2)?;
    let kernel = kernels::cast_u32_u32::builder()?.build(device.clone())?;
    let (a_done, b_done) = std::thread::scope(|scope| -> Result<_> {
        let a = scope.spawn(|| kernel.dispatch(&mut first, input.as_slice(), a.as_slice_mut()));
        let b = scope.spawn(|| kernel.dispatch(&mut second, input.as_slice(), b.as_slice_mut()));
        Ok((a.join().unwrap()?, b.join().unwrap()?))
    })?;
    join.wait_event(&a_done)?;
    join.wait_event(&b_done)?;
    let done = join.record_event()?;
    drop((input, first, second, join, a_done, b_done));
    done.wait()?;
    assert_eq!(&*a.map_read()?, &*b.map_read()?);
    assert_eq!(a.map_read()?[16], 16);
    Ok(())
}

#[test]
fn streaming_storage_grows_beyond_initial_capacity() -> Result<()> {
    let _device_lock = crate::device::TEST_DEVICE_LOCK.lock();
    // Exceed the former 1,024-command limit.
    const COPIES: usize = 1025;
    // Counting both 256 KiB buffers per copy exceeded the former 512 MiB limit.
    const LEN: usize = 256 * 1024 / std::mem::size_of::<u32>();
    let device = Device::builder()
        .streaming(StreamingOptions {
            capacity: 1,
            // Keep all commands unretired until the final wait.
            progress: Progress::Manual,
            ..Default::default()
        })
        .build()?;
    let source = Buffer::from_elem(Mem::Shared(&device), LEN, 7u32)?;
    let mut output = Buffer::<u32>::zeros(Mem::Readback(&device), LEN)?;
    let mut stream = device.stream()?;
    for _ in 0..COPIES {
        stream.copy(source.as_slice(), output.as_slice_mut())?;
    }
    stream.record_event()?.wait()?;
    assert!(output.map_read()?.iter().all(|&value| value == 7));
    Ok(())
}

#[test]
fn streaming_rejects_wrong_device_host_and_transfer_dispatch() -> Result<()> {
    let _device_lock = crate::device::TEST_DEVICE_LOCK.lock();
    let device = Device::builder().build()?;
    let other = Device::builder().build()?;
    let mut stream = device.transfer_stream()?;
    let source = Buffer::from_elem(Mem::Shared(&other), 1, 1u32)?;
    let mut output = Buffer::<u32>::zeros(Mem::Readback(&device), 1)?;
    assert!(stream
        .copy(source.as_slice(), output.as_slice_mut())
        .is_err());
    assert!(stream
        .copy(Slice::from_host_slice(&[1u32]), output.as_slice_mut())
        .is_err());
    let kernel = kernels::fill_u32::builder()?.build(device.clone())?;
    assert!(kernel
        .dispatch(&mut stream, 2, output.as_slice_mut())
        .is_err());
    assert!(output.try_map_write().is_ok());
    Ok(())
}

struct ChannelWake(mpsc::Sender<()>);
impl Wake for ChannelWake {
    fn wake(self: Arc<Self>) {
        let _ = self.0.send(());
    }
}

fn await_event(event: Event) -> Result<()> {
    let (send, recv) = mpsc::channel();
    let waker = Waker::from(Arc::new(ChannelWake(send)));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(event.wait_async());
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(result) => return result,
            Poll::Pending => {
                recv.recv_timeout(Duration::from_secs(10))
                    .expect("The completion future did not wake.");
            }
        }
    }
}

#[test]
fn streaming_async_waiters_and_dropped_events() -> Result<()> {
    let _device_lock = crate::device::TEST_DEVICE_LOCK.lock();
    let device = Device::builder().build()?;
    let mut stream = device.stream()?;
    let source = Buffer::from_elem(Mem::Upload(&device), 1024, 13u32)?;
    let mut output = Buffer::<u32>::zeros(Mem::Readback(&device), source.len())?;
    let done = stream.copy(source.as_slice(), output.as_slice_mut())?;
    let first = done.clone();
    let second = done.clone();
    let a = std::thread::spawn(move || await_event(first));
    let b = std::thread::spawn(move || await_event(second));
    drop(done.wait_async());
    drop((source, stream, done));
    a.join().unwrap()?;
    b.join().unwrap()?;
    assert!(output.map_read()?.iter().all(|value| *value == 13));
    Ok(())
}

#[test]
fn streaming_copy_views_and_empty_slices() -> Result<()> {
    let _device_lock = crate::device::TEST_DEVICE_LOCK.lock();
    let device = Device::builder().build()?;
    let source = Buffer::allocate_with(Mem::Upload(&device), 7, |i| i as u8)?;
    let mut output = Buffer::<u8>::zeros(Mem::Readback(&device), 5)?;
    let mut stream = device.transfer_stream()?;
    // Byte offsets and lengths need not be multiples of four.
    stream
        .copy(source.slice(1..4).unwrap(), output.slice_mut(1..4).unwrap())?
        .wait()?;
    stream
        .copy(source.slice(..0).unwrap(), output.slice_mut(..0).unwrap())?
        .wait()?;
    assert_eq!(&*output.map_read()?, &[0, 1, 2, 3, 0]);
    Ok(())
}
