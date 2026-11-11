//! Ordered GPU work and completion events.
//!
//! Streams retain submitted resources. Dropping an event does not cancel work.
//! Different streams can overlap when their buffer accesses and queues permit it.

#[cfg(not(feature = "device"))]
use crate::device::error::DeviceUnavailable;
use crate::{
    buffer::{Slice, SliceMut},
    device::Device,
    scalar::Scalar,
};
use anyhow::Result;
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

/// How the device retires completed submissions and wakes tasks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Progress {
    /// Use a device worker to retire work and wake waiting tasks.
    #[default]
    Background,
    /// Call [`Device::poll`] to retire work and wake waiting tasks.
    ///
    /// Submission still progresses in the background. Blocking waits drive polling.
    Manual,
}

/// Initial storage, batch size, and progress mode for a device's streaming runtime.
///
/// Storage grows beyond the initial capacities as needed.
/// Vulkan and argument preparation can still allocate during submission.
#[derive(Clone, Copy, Debug)]
pub struct StreamingOptions {
    /// Initial number of submission records and reusable command-storage slots.
    pub capacity: usize,
    /// Initial dependency capacity for streams, commands, and allocation readers.
    pub dependency_capacity: usize,
    /// Initial number of async waiters per event.
    pub waiter_capacity: usize,
    /// Maximum commands combined in one Vulkan submission. Must be nonzero.
    pub batch_size: usize,
    /// Completion polling mode.
    pub progress: Progress,
}

impl Default for StreamingOptions {
    fn default() -> Self {
        Self {
            capacity: 8,
            dependency_capacity: 4,
            waiter_capacity: 1,
            batch_size: 32,
            progress: Progress::Background,
        }
    }
}

/// An ordered sequence of GPU commands on one device.
///
/// General streams accept kernels and copies. Transfer streams accept copies.
/// CPU mappings must be released before enqueueing access to their allocations.
pub struct Stream {
    #[cfg(feature = "device")]
    pub(crate) device: crate::device::RawDevice,
    #[cfg(feature = "device")]
    pub(crate) inner: crate::device::RawStream,
}

impl Stream {
    /// Enqueues a copy between equal-length slices on this stream's device.
    ///
    /// Host allocations and overlapping copy ranges are rejected.
    pub fn copy<T: Scalar>(&mut self, src: Slice<'_, T>, dst: SliceMut<'_, T>) -> Result<Event> {
        #[cfg(feature = "device")]
        {
            anyhow::ensure!(src.len() == dst.len(), "Copy lengths differ.");
            let src = src.as_scalar_slice();
            let dst: crate::buffer::ScalarSliceMut = dst.into();
            let src = src
                .device_buffer()
                .ok_or_else(|| anyhow::anyhow!("Use a GPU allocation for a streaming copy."))?;
            let dst = dst
                .device_buffer_mut()
                .ok_or_else(|| anyhow::anyhow!("Use a GPU allocation for a streaming copy."))?;
            self.device
                .copy_on(&mut self.inner, src, dst)
                .map(|point| Event {
                    device: self.device.clone(),
                    point,
                })
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = (src, dst);
            Err(DeviceUnavailable.into())
        }
    }

    /// Adds an event dependency to subsequent commands without waiting on the CPU.
    pub fn wait_event(&mut self, event: &Event) -> Result<()> {
        #[cfg(feature = "device")]
        {
            anyhow::ensure!(
                self.device == event.device,
                "The event belongs to a different device."
            );
            self.inner.wait_event(&event.point);
            Ok(())
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = event;
            Err(DeviceUnavailable.into())
        }
    }

    /// Marks completion of preceding commands and event dependencies.
    pub fn record_event(&mut self) -> Result<Event> {
        #[cfg(feature = "device")]
        {
            let point = self.device.record_event(&mut self.inner)?;
            Ok(Event {
                device: self.device.clone(),
                point,
            })
        }
        #[cfg(not(feature = "device"))]
        {
            Err(DeviceUnavailable.into())
        }
    }

    /// Reserves space for additional explicit event dependencies.
    pub fn reserve_dependencies(&mut self, additional: usize) {
        #[cfg(feature = "device")]
        self.inner.reserve_dependencies(additional);
        #[cfg(not(feature = "device"))]
        let _ = additional;
    }

    /// Returns this stream's device.
    pub fn device(&self) -> Device {
        #[cfg(feature = "device")]
        {
            self.device.clone().into()
        }
        #[cfg(not(feature = "device"))]
        {
            Device::host()
        }
    }
}

/// An immutable completion point. Clones refer to the same work.
#[derive(Clone)]
pub struct Event {
    #[cfg(feature = "device")]
    pub(crate) device: crate::device::RawDevice,
    #[cfg(feature = "device")]
    pub(crate) point: crate::device::CompletionPoint,
}

impl Event {
    /// Checks completion and drives retirement without waiting for GPU work.
    pub fn is_complete(&self) -> Result<bool> {
        #[cfg(feature = "device")]
        {
            if self.point.result()? {
                return Ok(true);
            }
            let progress = self.device.poll();
            match self.point.result()? {
                true => Ok(true),
                false => progress.map(|()| false),
            }
        }
        #[cfg(not(feature = "device"))]
        {
            Err(DeviceUnavailable.into())
        }
    }

    /// Waits for completion and reports execution errors.
    pub fn wait(&self) -> Result<()> {
        #[cfg(feature = "device")]
        {
            self.device.wait_point(&self.point)
        }
        #[cfg(not(feature = "device"))]
        {
            Err(DeviceUnavailable.into())
        }
    }

    /// Waits without blocking the executor thread.
    ///
    /// With manual progress, another task or thread must call [`Device::poll`].
    pub fn wait_async(&self) -> EventWait {
        EventWait {
            event: self.clone(),
            #[cfg(feature = "device")]
            waiter: None,
        }
    }
}

/// A completion future. Dropping it removes its wake registration.
pub struct EventWait {
    event: Event,
    #[cfg(feature = "device")]
    waiter: Option<u64>,
}

impl Future for EventWait {
    type Output = Result<()>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        #[cfg(feature = "device")]
        {
            let this = &mut *self;
            this.event.point.register(&mut this.waiter, cx.waker());
            let result = match this.event.is_complete() {
                Ok(false) => return Poll::Pending,
                Ok(true) => Ok(()),
                Err(error) => Err(error),
            };
            if let Some(waiter) = this.waiter.take() {
                this.event.point.unregister(waiter);
            }
            Poll::Ready(result)
        }
        #[cfg(not(feature = "device"))]
        {
            let this = &mut *self;
            let _ = cx;
            Poll::Ready(this.event.wait())
        }
    }
}

impl Drop for EventWait {
    fn drop(&mut self) {
        #[cfg(feature = "device")]
        if let Some(waiter) = self.waiter {
            self.event.point.unregister(waiter);
        }
    }
}
