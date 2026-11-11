use super::{RawSlice, RawSliceInner};
use crate::{device::Device, scalar::Scalar};
use anyhow::{bail, Result};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

/// Selects memory for a new allocation.
///
/// Mapped placements require CPU access. Locality and caching are preferences.
/// GPU variants reject [`Device::host()`]. Allocations retain their device handle.
#[derive(Clone, Copy, Debug)]
pub enum Mem<'a> {
    /// Ordinary CPU memory.
    Host,
    /// GPU memory, preferably device-local. CPU mapping is optional.
    Device(&'a Device),
    /// CPU-mappable GPU memory, preferably device-local and cached.
    Shared(&'a Device),
    /// CPU-mappable GPU memory suited to sequential CPU writes.
    Upload(&'a Device),
    /// CPU-mappable GPU memory suited to CPU reads.
    Readback(&'a Device),
}

#[cfg(feature = "device")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MemKind {
    Device,
    Shared,
    Upload,
    Readback,
}

impl Mem<'_> {
    pub(crate) fn device(self) -> Result<Device> {
        match self {
            Self::Host => Ok(Device::host()),
            Self::Device(d) | Self::Shared(d) | Self::Upload(d) | Self::Readback(d) => {
                if d.is_host() {
                    bail!("Use Mem::Host for host allocations.");
                }
                Ok(d.clone())
            }
        }
    }

    #[cfg(feature = "device")]
    pub(crate) fn kind(self) -> MemKind {
        match self {
            Self::Host | Self::Device(_) => MemKind::Device,
            Self::Shared(_) => MemKind::Shared,
            Self::Upload(_) => MemKind::Upload,
            Self::Readback(_) => MemKind::Readback,
        }
    }
}

/// Properties of the Vulkan memory type selected for an allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryProperties {
    /// The CPU can map this memory.
    pub host_visible: bool,
    /// The memory is local to the GPU.
    pub device_local: bool,
    /// CPU reads use the host cache.
    pub host_cached: bool,
    /// CPU access does not require explicit cache flushes or invalidations.
    pub host_coherent: bool,
}

/// A CPU mapping cannot proceed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MapError {
    /// The allocation is not CPU-mappable.
    #[error("The allocation does not support CPU mapping.")]
    Unsupported,
    /// The allocation is not ready for CPU access.
    #[error("The allocation is not ready for CPU access.")]
    Busy,
    /// A live CPU guard prevents this access.
    #[error("The allocation has a conflicting CPU mapping.")]
    Conflict,
}

pub(super) struct Mapping {
    ptr: *mut u8,
    len: usize,
    #[cfg(feature = "device")]
    _device: Option<crate::device::DeviceMapping>,
}

impl Mapping {
    #[cfg_attr(not(feature = "device"), allow(unused_variables))]
    pub(super) fn new(raw: &RawSlice, write: bool, wait: bool) -> Result<Self> {
        match &raw.inner {
            RawSliceInner::Host(host) => Ok(Self {
                ptr: host.ptr,
                len: host.len,
                #[cfg(feature = "device")]
                _device: None,
            }),
            #[cfg(feature = "device")]
            RawSliceInner::Device(buffer) => {
                let mapping = buffer.map(write, wait)?;
                Ok(Self {
                    ptr: mapping.ptr(),
                    len: buffer.len(),
                    _device: Some(mapping),
                })
            }
        }
    }

    pub(super) fn ptr(&self) -> *mut u8 {
        self.ptr
    }
}

/// Keeps an allocation available for CPU reads until dropped.
///
/// GPU submissions that use the allocation return [`MapError::Conflict`].
pub struct ReadGuard<'a, T: Scalar> {
    pub(super) mapping: Mapping,
    pub(super) _borrow: PhantomData<&'a [T]>,
}

impl<T: Scalar> Deref for ReadGuard<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: The buffer is initialized, aligned, and reserved for CPU access.
        unsafe {
            std::slice::from_raw_parts(self.mapping.ptr.cast(), self.mapping.len / size_of::<T>())
        }
    }
}

/// Keeps an allocation available for exclusive CPU access until dropped.
///
/// Dropping the guard flushes CPU writes when the memory requires it.
/// A flush failure prevents further mapping and GPU use of the allocation.
pub struct WriteGuard<'a, T: Scalar> {
    pub(super) mapping: Mapping,
    pub(super) _borrow: PhantomData<&'a mut [T]>,
}

impl<T: Scalar> Deref for WriteGuard<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: The buffer is initialized, aligned, and reserved for CPU access.
        unsafe {
            std::slice::from_raw_parts(self.mapping.ptr.cast(), self.mapping.len / size_of::<T>())
        }
    }
}

impl<T: Scalar> DerefMut for WriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: The guard holds exclusive access to the initialized buffer.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.mapping.ptr.cast(),
                self.mapping.len / size_of::<T>(),
            )
        }
    }
}
