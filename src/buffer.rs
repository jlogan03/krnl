/*!

Buffers store data, and can be arguments to [kernels](crate::kernel). Buffers on the [host](crate::device::Device::host)
store data inline, essentially as a [`Vec`] or [slice](https://doc.rust-lang.org/std/primitive.slice.html).

[`BufferBase`](crate::buffer::BufferBase) is a generic buffer. [`ScalarBufferBase`](crate::buffer::ScalarBufferBase) is a dynamically typed buffer.

# Memory placement

Choose [`Mem`](crate::buffer::Mem) when an operation allocates storage:

| Placement | Access | Preference |
| --- | --- | --- |
| `Host` | CPU | Ordinary host storage |
| `Device(&device)` | GPU | Device-local memory |
| `Shared(&device)` | CPU and GPU | Device-local, CPU-cached memory |
| `Upload(&device)` | CPU and GPU | Sequential CPU writes |
| `Readback(&device)` | CPU and GPU | CPU-cached memory |

GPU placements support kernels and copies. Locality and caching are preferences.
Several placements can select the same memory type. Use
[`memory_properties()`](crate::buffer::BufferBase::memory_properties) to inspect the allocation.

[`allocate_with()`](crate::buffer::BufferBase::allocate_with) writes each element directly into
CPU-accessible storage. It does not allocate a temporary vector or first fill
the storage with zeros. It accepts every placement except `Device`.

```no_run
# use krnl::{anyhow::Result, buffer::{Buffer, Mem}, device::Device};
# fn main() -> Result<()> {
let device = Device::builder().build()?;
let input = Buffer::allocate_with(Mem::Upload(&device), 1024, |i| i as u32)?;
let mut work = input.to_owned(Mem::Device(&device))?;
work.fill(1)?;
let result = work.to_owned(Mem::Readback(&device))?;
let read = result.map_read()?;
assert_eq!(read[0], 1);
# Ok(())
# }
```

Copies complete before returning. Mapping waits for pending GPU access.
`try_map_read` and `try_map_write` return [`MapError::Busy`](crate::buffer::MapError::Busy) instead of waiting.
Mapping never changes placement or copies data. `Device` storage supports mapping
only when its selected memory type permits it.

Guards reserve the whole allocation, including its slices and bitcasts.
A write guard excludes other CPU access. Any CPU guard prevents GPU submission
for that allocation. Drop the guard before dispatching a kernel or GPU copy.
Guards handle cache invalidation and flushing when the memory requires them.

`Shared` describes memory access. [`ArcBuffer`](crate::buffer::ArcBuffer) provides shared ownership.
Direct host-slice access remains limited to `Host` storage. Serialization maps
the buffer for reading. Copy unmappable storage to `Host` before serialization.

# Example
```
use krnl::{anyhow::Result, device::Device, buffer::{Buffer, Mem, Slice, SliceMut}};

fn saxpy(x: Slice<f32>, alpha: f32, mut y: SliceMut<f32>) -> Result<()> {
    if let Some((x, y)) = x.as_host_slice().zip(y.as_host_slice_mut()) {
        for (x, y) in x.iter().copied().zip(y) {
            *y += alpha * x;
        }
        return Ok(());
    }
    todo!()
}

fn main() -> Result<()> {
    let x = vec![1f32];
    let alpha = 2f32;
    let y = vec![0f32];
    # if false {
    let device = Device::builder().build().ok().unwrap_or(Device::host());
    # }
    # let device = Device::host();
    let mem = if device.is_host() { Mem::Host } else { Mem::Device(&device) };
    let x = Buffer::from_host_vec(x).into_owned(mem)?;
    let mut y = Buffer::from_host_vec(y).into_owned(mem)?;
    saxpy(x.as_slice(), alpha, y.as_slice_mut())?;
    let y = y.into_host_vec()?;
    println!("{y:?}");
    Ok(())
}
```
*/

#[cfg(doc)]
use crate::device::error::DeviceLost;
#[cfg(all(not(feature = "device"), doc))]
use crate::device::Features;
use crate::{
    device::{Device, DeviceInner},
    scalar::{Scalar, ScalarElem, ScalarType},
};
#[cfg(feature = "device")]
use crate::{
    device::{DeviceBuffer, Features},
    macros::module,
};
use anyhow::{bail, Result};
use bytemuck::PodCastError;
use dry::{macro_for, macro_wrap};
use half::{bf16, f16};
use paste::paste;
#[cfg(feature = "serde")]
use serde::{de::Deserializer, ser::Serializer, Deserialize, Serialize};
use std::{
    fmt::{self, Debug},
    marker::PhantomData,
    mem::{forget, size_of},
    ops::{Bound, RangeBounds},
    sync::Arc,
};

/// Errors.
pub mod error {
    #[cfg(feature = "device")]
    use crate::device::DeviceId;
    use std::fmt::{self, Debug, Display};

    /// No more memory on the device.
    #[derive(Clone, Copy, Debug, thiserror::Error)]
    pub struct OutOfDeviceMemory(
        #[cfg(feature = "device")]
        #[allow(unused)]
        pub(crate) DeviceId,
    );

    impl Display for OutOfDeviceMemory {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            Debug::fmt(self, f)
        }
    }

    /// Device buffers are limited to [`i32::MAX`] (2147483647) bytes .
    #[derive(Clone, Copy, Debug, thiserror::Error)]
    pub struct DeviceBufferTooLarge {
        #[cfg(feature = "device")]
        #[allow(unused)]
        pub(crate) bytes: usize,
    }

    impl Display for DeviceBufferTooLarge {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            Debug::fmt(self, f)
        }
    }
}
#[cfg(doc)]
use error::{DeviceBufferTooLarge, OutOfDeviceMemory};

mod memory;
use memory::Mapping;
#[cfg(feature = "device")]
pub(crate) use memory::MemKind;
pub use memory::{MapError, Mem, MemoryProperties, ReadGuard, WriteGuard};

#[derive(Copy, Clone)]
struct RawHostSlice {
    ptr: *mut u8,
    len: usize,
}

impl RawHostSlice {
    #[cfg(feature = "device")]
    unsafe fn as_bytes<'a>(&self) -> &'a [u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    #[cfg(feature = "device")]
    unsafe fn as_uninit_bytes<'a>(&mut self) -> &'a mut [std::mem::MaybeUninit<u8>] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast(), self.len) }
    }
}

#[derive(Clone)]
struct RawSlice {
    inner: RawSliceInner,
}

impl RawSlice {
    fn matches(&self, mem: Mem<'_>) -> Result<bool> {
        let device = mem.device()?;
        Ok(match &self.inner {
            RawSliceInner::Host(_) => device.is_host(),
            #[cfg(feature = "device")]
            RawSliceInner::Device(buffer) => device == self.device() && buffer.kind() == mem.kind(),
        })
    }
    fn memory_properties(&self) -> Option<MemoryProperties> {
        match &self.inner {
            RawSliceInner::Host(_) => None,
            #[cfg(feature = "device")]
            RawSliceInner::Device(buffer) => buffer.memory_properties(),
        }
    }
    fn is_host_visible(&self) -> bool {
        match &self.inner {
            RawSliceInner::Host(_) => true,
            #[cfg(feature = "device")]
            RawSliceInner::Device(buffer) => buffer.host_visible(),
        }
    }
    fn device(&self) -> Device {
        match &self.inner {
            RawSliceInner::Host(_) => Device::host(),
            #[cfg(feature = "device")]
            RawSliceInner::Device(buffer) => buffer.device().into(),
        }
    }
    fn len(&self) -> usize {
        match &self.inner {
            RawSliceInner::Host(raw) => raw.len,
            #[cfg(feature = "device")]
            RawSliceInner::Device(buffer) => buffer.len(),
        }
    }
    fn bitcast(self, scalar_type: ScalarType) -> Result<Self, PodCastError> {
        let (index, len) = match &self.inner {
            RawSliceInner::Host(raw) => (raw.ptr as usize, raw.len),
            #[cfg(feature = "device")]
            RawSliceInner::Device(buffer) => (buffer.offset(), buffer.len()),
        };
        let width = scalar_type.size();
        if index % width != 0 {
            Err(PodCastError::TargetAlignmentGreaterAndInputNotAligned)
        } else if len % width != 0 {
            Err(PodCastError::OutputSliceWouldHaveSlop)
        } else {
            Ok(self)
        }
    }
    fn slice(mut self, range: impl RangeBounds<usize>, scalar_type: ScalarType) -> Option<Self> {
        let width = scalar_type.size();
        let start = match range.start_bound() {
            Bound::Included(x) => x.checked_mul(width)?,
            Bound::Excluded(x) => x.checked_mul(width)?.checked_add(width)?,
            Bound::Unbounded => 0,
        };
        if start > self.len() {
            return None;
        }
        let end = match range.end_bound() {
            Bound::Included(x) => x.checked_mul(width)?.checked_add(width)?,
            Bound::Excluded(x) => x.checked_mul(width)?,
            Bound::Unbounded => self.len(),
        };
        if end > self.len() {
            return None;
        }
        match &mut self.inner {
            RawSliceInner::Host(raw) => {
                if start > end {
                    return None;
                }
                raw.ptr = unsafe { raw.ptr.add(start) };
                raw.len = end - start;
            }
            #[cfg(feature = "device")]
            RawSliceInner::Device(buffer) => {
                *buffer = buffer.slice(start..end)?;
            }
        }
        Some(self)
    }
}

#[derive(Clone)]
enum RawSliceInner {
    Host(RawHostSlice),
    #[cfg(feature = "device")]
    Device(DeviceBuffer),
}

#[derive(derive_more::Deref, derive_more::DerefMut)]
struct RawBuffer {
    #[deref]
    #[deref_mut]
    slice: RawSlice,
    cap: usize,
    width: usize,
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        #[cfg_attr(not(feature = "device"), allow(irrefutable_let_patterns))]
        if let RawSliceInner::Host(slice) = self.inner {
            let ptr = slice.ptr;
            let len = slice.len / self.width;
            let cap = self.cap / self.width;
            unsafe {
                match self.width {
                    1 => {
                        Vec::<std::mem::MaybeUninit<u8>>::from_raw_parts(ptr as _, len, cap);
                    }
                    2 => {
                        Vec::<std::mem::MaybeUninit<u16>>::from_raw_parts(ptr as _, len, cap);
                    }
                    4 => {
                        Vec::<std::mem::MaybeUninit<u32>>::from_raw_parts(ptr as _, len, cap);
                    }
                    8 => {
                        Vec::<std::mem::MaybeUninit<u64>>::from_raw_parts(ptr as _, len, cap);
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}

mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

macro_for!($S in [ScalarBufferRepr, ScalarArcBufferRepr] {
    impl Sealed for $S {}
    unsafe impl Send for $S {}
    unsafe impl Sync for $S {}
});

macro_for!($S in [ScalarSliceRepr,ScalarSliceMutRepr, ScalarCowBufferRepr] {
    impl Sealed for $S<'_> {}
    unsafe impl Send for $S<'_> {}
    unsafe impl Sync for $S<'_> {}
});

/** Marker trait for scalar buffers.

This trait can not be implemented outside the crate. */
pub trait ScalarData: Sealed {
    #[doc(hidden)]
    fn as_scalar_slice(&self) -> ScalarSliceRepr<'_>;
    #[doc(hidden)]
    fn get_scalar_slice_mut(&mut self) -> Option<ScalarSliceMutRepr<'_>> {
        None
    }
    #[doc(hidden)]
    fn device(&self) -> Device {
        self.as_scalar_slice().raw.device()
    }
    #[doc(hidden)]
    fn scalar_type(&self) -> ScalarType {
        self.as_scalar_slice().scalar_type
    }
    #[doc(hidden)]
    fn len(&self) -> usize {
        let slice = self.as_scalar_slice();
        slice.raw.len() / slice.scalar_type().size()
    }
    #[doc(hidden)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[doc(hidden)]
    fn try_into_scalar_buffer(self) -> Result<ScalarBufferRepr, Self>
    where
        Self: Sized,
    {
        Err(self)
    }
    #[doc(hidden)]
    fn try_into_scalar_arc_buffer(self) -> Result<ScalarArcBufferRepr, Self>
    where
        Self: Sized,
    {
        match self.try_into_scalar_buffer() {
            Ok(buffer) => Ok(ScalarArcBufferRepr::from_scalar_buffer(buffer)),
            Err(this) => Err(this),
        }
    }
    #[doc(hidden)]
    fn to_scalar_arc_buffer(&self, mem: Mem<'_>) -> Result<ScalarArcBufferRepr> {
        Ok(ScalarArcBufferRepr::from_scalar_buffer(
            self.as_scalar_slice().to_scalar_buffer(mem)?,
        ))
    }
}

/// Marker trait for mutable scalar buffers.
pub trait ScalarDataMut: ScalarData {
    #[doc(hidden)]
    fn as_scalar_slice_mut(&mut self) -> ScalarSliceMutRepr<'_>;
}

/// Marker trait for owned scalar buffers.
pub trait ScalarDataOwned: ScalarData {
    #[doc(hidden)]
    fn from_scalar_buffer(buffer: ScalarBufferRepr) -> Self
    where
        Self: Sized;
    #[doc(hidden)]
    fn make_scalar_slice_mut(&mut self, mem: Mem<'_>) -> Result<ScalarSliceMutRepr<'_>>;
}

/// [`ScalarBuffer`] representation.
pub struct ScalarBufferRepr {
    raw: RawBuffer,
    scalar_type: ScalarType,
}

impl<T: Scalar> From<BufferRepr<T>> for ScalarBufferRepr {
    fn from(buffer: BufferRepr<T>) -> Self {
        Self {
            raw: buffer.raw,
            scalar_type: T::SCALAR_TYPE,
        }
    }
}

impl ScalarData for ScalarBufferRepr {
    fn as_scalar_slice(&self) -> ScalarSliceRepr<'_> {
        ScalarSliceRepr {
            raw: self.raw.slice.clone(),
            scalar_type: self.scalar_type,
            _m: PhantomData,
        }
    }
    fn get_scalar_slice_mut(&mut self) -> Option<ScalarSliceMutRepr<'_>> {
        Some(self.as_scalar_slice_mut())
    }
    fn try_into_scalar_buffer(self) -> Result<Self, Self>
    where
        Self: Sized,
    {
        Ok(self)
    }
}

impl ScalarDataMut for ScalarBufferRepr {
    fn as_scalar_slice_mut(&mut self) -> ScalarSliceMutRepr<'_> {
        ScalarSliceMutRepr {
            raw: self.raw.slice.clone(),
            scalar_type: self.scalar_type,
            _m: PhantomData,
        }
    }
}

impl ScalarDataOwned for ScalarBufferRepr {
    fn from_scalar_buffer(buffer: Self) -> Self
    where
        Self: Sized,
    {
        buffer
    }
    fn make_scalar_slice_mut(&mut self, _mem: Mem<'_>) -> Result<ScalarSliceMutRepr<'_>> {
        Ok(self.as_scalar_slice_mut())
    }
}

/// [`ScalarSlice`] representation.
#[derive(Clone)]
pub struct ScalarSliceRepr<'a> {
    raw: RawSlice,
    scalar_type: ScalarType,
    _m: PhantomData<&'a ()>,
}

impl<'a> ScalarSliceRepr<'a> {
    fn to_scalar_buffer(&self, mem: Mem<'_>) -> Result<ScalarBufferRepr> {
        Ok(ScalarSlice { data: self.clone() }.to_owned(mem)?.data)
    }
    fn bitcast(
        self,
        scalar_type: ScalarType,
    ) -> Result<ScalarSliceRepr<'a>, bytemuck::PodCastError> {
        let raw = self.raw.bitcast(scalar_type)?;
        Ok(ScalarSliceRepr {
            raw,
            scalar_type,
            _m: Default::default(),
        })
    }
    fn slice(self, range: impl RangeBounds<usize>) -> Option<Self> {
        Some(Self {
            raw: self.raw.slice(range, self.scalar_type)?,
            ..self
        })
    }
}

impl<'a, T: Scalar> From<SliceRepr<'a, T>> for ScalarSliceRepr<'a> {
    fn from(slice: SliceRepr<'a, T>) -> Self {
        Self {
            raw: slice.raw,
            scalar_type: T::SCALAR_TYPE,
            _m: PhantomData,
        }
    }
}

impl ScalarData for ScalarSliceRepr<'_> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr<'_> {
        ScalarSliceRepr {
            raw: self.raw.clone(),
            scalar_type: self.scalar_type,
            _m: PhantomData,
        }
    }
}

#[cfg(feature = "serde")]
impl Serialize for ScalarSliceRepr<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::{Error, SerializeSeq, SerializeTupleStruct};

        let scalar_type = self.scalar_type;
        let mapping = Mapping::new(&self.raw, false, true).map_err(S::Error::custom)?;
        // SAFETY: The mapping retains read access to initialized bytes.
        let data = BufferData {
            bytes: unsafe { std::slice::from_raw_parts(mapping.ptr(), self.raw.len()) },
        };
        let mut tuple_struct = serializer.serialize_tuple_struct("Buffer", 3)?;
        tuple_struct.serialize_field(&scalar_type)?;
        tuple_struct.serialize_field(&self.len())?;
        tuple_struct.serialize_field(&data)?;

        struct BufferData<'a> {
            bytes: &'a [u8],
        }

        impl Serialize for BufferData<'_> {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                let len = self.bytes.len();
                let items = len / 8 + (len % 8 != 0) as usize;
                let mut seq = serializer.serialize_seq(Some(items))?;
                for chunk in self.bytes.chunks(8) {
                    let mut item = 0u64;
                    bytemuck::bytes_of_mut(&mut item)[..chunk.len()].copy_from_slice(chunk);
                    seq.serialize_element(&item.to_be())?;
                }
                seq.end()
            }
        }

        tuple_struct.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for ScalarBufferRepr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{DeserializeSeed, Error, SeqAccess, Visitor};

        struct BufferVisitor;

        impl<'de> Visitor<'de> for BufferVisitor {
            type Value = ScalarBufferRepr;
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(formatter, "struct Buffer")
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let scalar_type = if let Some(scalar_type) = seq.next_element::<ScalarType>()? {
                    scalar_type
                } else {
                    return Err(A::Error::custom("expected ScalarType"));
                };
                let len = if let Some(len) = seq.next_element::<usize>()? {
                    len
                } else {
                    return Err(A::Error::custom("expected usize"));
                };
                let visitor = BufferDataVisitor { scalar_type, len };
                if let Some(buffer) = seq.next_element_seed(visitor)? {
                    Ok(buffer)
                } else {
                    Err(A::Error::custom("expected sequence"))
                }
            }
        }

        struct BufferDataVisitor {
            scalar_type: ScalarType,
            len: usize,
        }

        impl BufferDataVisitor {
            fn parse<'de, T: Scalar, A>(self, mut seq: A) -> Result<ScalarBufferRepr, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let Self { scalar_type, len } = self;
                let mut data = Vec::<T>::with_capacity(len);
                while data.len() < len {
                    if let Some(item) = seq.next_element::<u64>()? {
                        let item = u64::from_be(item);
                        data.extend(bytemuck::cast_slice(&[item]));
                    } else {
                        break;
                    }
                }
                if data.len() != len {
                    return Err(A::Error::invalid_length(
                        data.len(),
                        &len.to_string().as_str(),
                    ));
                }
                assert_eq!(size_of::<T>(), scalar_type.size());
                let raw = BufferRepr::from_vec(data).raw;
                Ok(ScalarBufferRepr { raw, scalar_type })
            }
        }

        impl<'de> Visitor<'de> for BufferDataVisitor {
            type Value = ScalarBufferRepr;
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                write!(formatter, "buffer data")
            }
            fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                match self.scalar_type.size() {
                    4 => self.parse::<u32, _>(seq),
                    _ => todo!(),
                }
            }
        }

        impl<'de> DeserializeSeed<'de> for BufferDataVisitor {
            type Value = ScalarBufferRepr;
            fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: Deserializer<'de>,
            {
                deserializer.deserialize_seq(self)
            }
        }

        deserializer.deserialize_tuple_struct("Buffer", 3, BufferVisitor)
    }
}

/// [`ScalarBuffer`] representation.
#[derive(Clone)]
pub struct ScalarSliceMutRepr<'a> {
    raw: RawSlice,
    scalar_type: ScalarType,
    _m: PhantomData<&'a ()>,
}

impl<'a> ScalarSliceMutRepr<'a> {
    fn bitcast_mut(
        self,
        scalar_type: ScalarType,
    ) -> Result<ScalarSliceMutRepr<'a>, bytemuck::PodCastError> {
        let raw = self.raw.bitcast(scalar_type)?;
        Ok(ScalarSliceMutRepr {
            raw,
            scalar_type,
            _m: Default::default(),
        })
    }
    fn slice(self, range: impl RangeBounds<usize>) -> Option<Self> {
        Some(Self {
            raw: self.raw.slice(range, self.scalar_type)?,
            ..self
        })
    }
    fn copy_from_scalar_slice(&mut self, src: &ScalarSliceRepr<'_>) -> Result<()> {
        if self.scalar_type() != src.scalar_type() {
            bail!(
                "Can not copy slice from {:?} to {:?}",
                src.scalar_type(),
                self.scalar_type()
            );
        }
        macro_wrap!(paste! {
            match self.scalar_type() {
                macro_for!($T in  [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                    ScalarType::[<$T:upper>] => {
                        SliceMutRepr::<$T>::try_from(self.as_scalar_slice_mut())
                            .ok()
                            .unwrap()
                            .copy_from_slice(&src.clone().try_into().ok().unwrap())
                    }
                })
                _ => unreachable!(),
            }
        })
    }
}

impl<'a, T: Scalar> From<SliceMutRepr<'a, T>> for ScalarSliceMutRepr<'a> {
    fn from(slice: SliceMutRepr<'a, T>) -> Self {
        Self {
            raw: slice.raw,
            scalar_type: T::SCALAR_TYPE,
            _m: PhantomData,
        }
    }
}

impl ScalarData for ScalarSliceMutRepr<'_> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr<'_> {
        ScalarSliceRepr {
            raw: self.raw.clone(),
            scalar_type: self.scalar_type,
            _m: PhantomData,
        }
    }
}

impl ScalarDataMut for ScalarSliceMutRepr<'_> {
    fn as_scalar_slice_mut(&mut self) -> ScalarSliceMutRepr<'_> {
        ScalarSliceMutRepr {
            raw: self.raw.clone(),
            scalar_type: self.scalar_type,
            _m: PhantomData,
        }
    }
}

/// [`ScalarArcBuffer`] representation.
#[derive(Clone)]
pub struct ScalarArcBufferRepr {
    raw: Arc<RawBuffer>,
    scalar_type: ScalarType,
}

impl From<ScalarBufferRepr> for ScalarArcBufferRepr {
    fn from(buffer: ScalarBufferRepr) -> Self {
        Self::from_scalar_buffer(buffer)
    }
}

impl<T: Scalar> From<ArcBufferRepr<T>> for ScalarArcBufferRepr {
    fn from(buffer: ArcBufferRepr<T>) -> Self {
        Self {
            raw: buffer.raw,
            scalar_type: T::SCALAR_TYPE,
        }
    }
}

impl ScalarData for ScalarArcBufferRepr {
    fn as_scalar_slice(&self) -> ScalarSliceRepr<'_> {
        ScalarSliceRepr {
            raw: self.raw.slice.clone(),
            scalar_type: self.scalar_type,
            _m: PhantomData,
        }
    }
    fn get_scalar_slice_mut(&mut self) -> Option<ScalarSliceMutRepr<'_>> {
        let raw = Arc::get_mut(&mut self.raw)?;
        Some(ScalarSliceMutRepr {
            raw: raw.slice.clone(),
            scalar_type: self.scalar_type,
            _m: PhantomData,
        })
    }
    fn try_into_scalar_buffer(self) -> Result<ScalarBufferRepr, Self> {
        match Arc::try_unwrap(self.raw) {
            Ok(raw) => Ok(ScalarBufferRepr {
                raw,
                scalar_type: self.scalar_type,
            }),
            Err(raw) => Err(Self {
                raw,
                scalar_type: self.scalar_type,
            }),
        }
    }
    fn try_into_scalar_arc_buffer(self) -> Result<Self, Self> {
        Ok(self)
    }
    fn to_scalar_arc_buffer(&self, _mem: Mem<'_>) -> Result<Self> {
        Ok(self.clone())
    }
}

impl ScalarDataOwned for ScalarArcBufferRepr {
    fn from_scalar_buffer(buffer: ScalarBufferRepr) -> Self {
        Self {
            #[allow(clippy::arc_with_non_send_sync)]
            raw: Arc::new(buffer.raw),
            scalar_type: buffer.scalar_type,
        }
    }
    fn make_scalar_slice_mut(&mut self, mem: Mem<'_>) -> Result<ScalarSliceMutRepr<'_>> {
        if let Some(raw) = Arc::get_mut(&mut self.raw) {
            return Ok(ScalarSliceMutRepr {
                raw: raw.slice.clone(),
                scalar_type: self.scalar_type,
                _m: PhantomData,
            });
        }
        #[allow(clippy::arc_with_non_send_sync)]
        {
            self.raw = Arc::new(self.as_scalar_slice().to_scalar_buffer(mem)?.raw);
        }
        Ok(ScalarSliceMutRepr {
            raw: self.raw.slice.clone(),
            scalar_type: self.scalar_type,
            _m: PhantomData,
        })
    }
}

/// [`ScalarCowBuffer`] representation.
pub enum ScalarCowBufferRepr<'a> {
    #[doc(hidden)]
    Borrowed(ScalarSliceRepr<'a>),
    #[doc(hidden)]
    Owned(ScalarBufferRepr),
}

impl<'a> From<ScalarSliceRepr<'a>> for ScalarCowBufferRepr<'a> {
    fn from(slice: ScalarSliceRepr<'a>) -> Self {
        Self::Borrowed(slice)
    }
}

impl<'a> From<ScalarBufferRepr> for ScalarCowBufferRepr<'a> {
    fn from(buffer: ScalarBufferRepr) -> Self {
        Self::Owned(buffer)
    }
}

impl<'a, T: Scalar> From<CowBufferRepr<'a, T>> for ScalarCowBufferRepr<'a> {
    fn from(buffer: CowBufferRepr<'a, T>) -> Self {
        match buffer {
            CowBufferRepr::Borrowed(slice) => Self::Borrowed(slice.into()),
            CowBufferRepr::Owned(buffer) => Self::Owned(buffer.into()),
        }
    }
}

impl<'a> ScalarData for ScalarCowBufferRepr<'a> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr<'_> {
        match self {
            Self::Borrowed(slice) => slice.clone(),
            Self::Owned(buffer) => buffer.as_scalar_slice(),
        }
    }
    fn try_into_scalar_buffer(self) -> Result<ScalarBufferRepr, Self>
    where
        Self: Sized,
    {
        match self {
            Self::Borrowed(_) => Err(self),
            Self::Owned(buffer) => Ok(buffer),
        }
    }
    fn get_scalar_slice_mut(&mut self) -> Option<ScalarSliceMutRepr<'_>> {
        match self {
            Self::Borrowed(_) => None,
            Self::Owned(buffer) => buffer.get_scalar_slice_mut(),
        }
    }
}

impl<'a> ScalarDataOwned for ScalarCowBufferRepr<'a> {
    fn from_scalar_buffer(buffer: ScalarBufferRepr) -> Self
    where
        Self: Sized,
    {
        Self::Owned(buffer)
    }
    fn make_scalar_slice_mut(&mut self, mem: Mem<'_>) -> Result<ScalarSliceMutRepr<'_>> {
        match self {
            Self::Borrowed(slice) => {
                *self = Self::Owned(slice.to_scalar_buffer(mem)?);
                Ok(self.get_scalar_slice_mut().unwrap())
            }
            Self::Owned(buffer) => buffer.make_scalar_slice_mut(mem),
        }
    }
}

/** Dynamically typed buffer.

Use [`From`] to convert from [`BufferBase`].
*/
#[derive(Clone)]
pub struct ScalarBufferBase<S: ScalarData> {
    data: S,
}

/** ScalarBuffer

See [`ScalarBufferBase`]. */
pub type ScalarBuffer = ScalarBufferBase<ScalarBufferRepr>;
/** ScalarSlice

See [`ScalarBufferBase`]. */
pub type ScalarSlice<'a> = ScalarBufferBase<ScalarSliceRepr<'a>>;
/** ScalarSliceMut

See [`ScalarBufferBase`]. */
pub type ScalarSliceMut<'a> = ScalarBufferBase<ScalarSliceMutRepr<'a>>;
/** ScalarArcBuffer

Like [`Arc`], a clonable shared scalar buffer.

See [`ScalarBufferBase`]. */
pub type ScalarArcBuffer = ScalarBufferBase<ScalarArcBufferRepr>;
/** CowBuffer

Like [`Cow`](::std::borrow::Cow), can be created from a [`ScalarSlice`] or [`ScalarBuffer`].

See [`BufferBase`]. */
pub type ScalarCowBuffer<'a> = ScalarBufferBase<ScalarCowBufferRepr<'a>>;

impl<S: ScalarDataOwned> ScalarBufferBase<S> {
    /// Allocate a scalar buffer.
    ///
    /// # Safety
    /// The buffer will not be initialized.
    ///
    /// # Errors
    /// - [`DeviceLost`]
    /// - [`DeviceBufferTooLarge`]
    /// - [`OutOfDeviceMemory`]
    ///
    /// See [`ScalarBufferBase::zeros()`] for a safe alternative.
    pub unsafe fn uninit(mem: Mem<'_>, len: usize, scalar_type: ScalarType) -> Result<Self> {
        macro_wrap!(paste! {
            match scalar_type {
                macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                    ScalarType::[<$T:upper>] => Ok(unsafe { Buffer::<$T>::uninit(mem, len)? }.into()),
                })
                _ => unreachable!(),
            }
        })
    }
    /** Create a scalar buffer filled with `elem'.

    # Errors
    - [`DeviceLost`]
    - [`DeviceBufferTooLarge`]
    - [`OutOfDeviceMemory`]
    - Could not dispatch the kernel.
        - This may require [`Features`] for the type.

    See [.fill()](ScalarBufferBase::fill).
     */
    pub fn from_elem(mem: Mem<'_>, len: usize, elem: ScalarElem) -> Result<Self> {
        macro_wrap!(paste! {
            match elem {
                macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                  ScalarElem::[<$T:upper>](elem) => Ok(Buffer::from_elem(mem, len, elem)?.into()),
                })
                _ => unreachable!(),
            }
        })
    }
    /** Create a scalar buffer filled with 0's.

    See [`from_elem`](ScalarBufferBase::from_elem). */
    pub fn zeros(mem: Mem<'_>, len: usize, scalar_type: ScalarType) -> Result<Self> {
        Self::from_elem(mem, len, ScalarElem::zero(scalar_type))
    }
    /** Create a scalar buffer filled with 1's.

    See [`from_elem`](ScalarBufferBase::from_elem). */
    pub fn ones(mem: Mem<'_>, len: usize, scalar_type: ScalarType) -> Result<Self> {
        Self::from_elem(mem, len, ScalarElem::one(scalar_type))
    }
    /// Create a scalar buffer from a [`ScalarBuffer`].
    pub fn from_scalar_buffer(buffer: ScalarBuffer) -> Self {
        let data = S::from_scalar_buffer(buffer.data);
        Self { data }
    }
}

impl<S: ScalarData> ScalarBufferBase<S> {
    /// The device.
    pub fn device(&self) -> Device {
        self.data.device()
    }
    /// Returns the selected Vulkan memory properties, if an allocation exists.
    pub fn memory_properties(&self) -> Option<MemoryProperties> {
        self.data.as_scalar_slice().raw.memory_properties()
    }
    /// Reports whether this storage supports CPU mapping.
    pub fn is_host_visible(&self) -> bool {
        self.data.as_scalar_slice().raw.is_host_visible()
    }
    /// The scalar_type.
    pub fn scalar_type(&self) -> ScalarType {
        self.data.scalar_type()
    }
    /// The length.
    pub fn len(&self) -> usize {
        self.data.len()
    }
    /// Is the buffer empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Borrow as a scalar slice.
    pub fn as_scalar_slice(&self) -> ScalarSlice<'_> {
        let data = self.data.as_scalar_slice();
        ScalarSlice { data }
    }
    /// Borrow as a mutable scalar slice.
    pub fn as_scalar_slice_mut(&mut self) -> ScalarSliceMut<'_>
    where
        S: ScalarDataMut,
    {
        let data = self.data.as_scalar_slice_mut();
        ScalarSliceMut { data }
    }
    /// Borrows as a mutable scalar slice if possible.
    pub fn get_scalar_slice_mut(&mut self) -> Option<ScalarSliceMut<'_>> {
        Some(ScalarSliceMut {
            data: self.data.get_scalar_slice_mut()?,
        })
    }
    /// Borrows as a mutable scalar slice. Uses `mem` if a copy is needed.
    pub fn make_scalar_slice_mut(&mut self, mem: Mem<'_>) -> Result<ScalarSliceMut<'_>>
    where
        S: ScalarDataOwned,
    {
        Ok(ScalarSliceMut {
            data: self.data.make_scalar_slice_mut(mem)?,
        })
    }
    /// Copies into the requested memory.
    pub fn to_owned(&self, mem: Mem<'_>) -> Result<ScalarBuffer> {
        let slice = self.as_scalar_slice();
        macro_wrap!(paste! { match slice.scalar_type() {
            macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                ScalarType::[<$T:upper>] => Ok(Slice::<$T>::try_from(slice).ok().unwrap().to_owned(mem)?.into()),
            })
            _ => unreachable!(),
        }})
    }
    /// Reuses owned storage if it matches `mem`. Copies otherwise.
    pub fn into_owned(self, mem: Mem<'_>) -> Result<ScalarBuffer> {
        if !self.data.as_scalar_slice().raw.matches(mem)? {
            return self.to_owned(mem);
        }
        match self.data.try_into_scalar_buffer() {
            Ok(data) => Ok(ScalarBuffer { data }),
            Err(data) => Self { data }.to_owned(mem),
        }
    }
    /// Reuses storage when possible and returns shared ownership.
    pub fn into_shared(self, mem: Mem<'_>) -> Result<ScalarArcBuffer> {
        if !self.data.as_scalar_slice().raw.matches(mem)? {
            return self.to_owned(mem).map(Into::into);
        }
        let data = match self.data.try_into_scalar_arc_buffer() {
            Ok(data) => data,
            Err(data) => data.to_scalar_arc_buffer(mem)?,
        };
        Ok(ScalarArcBuffer { data })
    }
    /// Clones shared storage if it matches `mem`. Copies otherwise.
    pub fn to_shared(&self, mem: Mem<'_>) -> Result<ScalarArcBuffer> {
        if !self.data.as_scalar_slice().raw.matches(mem)? {
            return self.to_owned(mem).map(Into::into);
        }
        Ok(ScalarArcBuffer {
            data: self.data.to_scalar_arc_buffer(mem)?,
        })
    }
    /** Fills the buffer with `elem`.

    See [`BufferBase::fill`]. */
    pub fn fill(&mut self, elem: ScalarElem) -> Result<()>
    where
        S: ScalarDataMut,
    {
        let slice = self.as_scalar_slice_mut();
        macro_wrap!(paste! { match elem {
            macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                ScalarElem::[<$T:upper>](elem) => {
                    SliceMut::<$T>::try_from(slice).ok().unwrap().fill(elem)?;
                }
            })
            _ => unreachable!(),
        }});
        Ok(())
    }
    /** Casts to `scalar_type`.

    See [`BufferBase::cast`]. */
    pub fn cast(&self, mem: Mem<'_>, scalar_type: ScalarType) -> Result<ScalarBuffer> {
        macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            if let Ok(x) = Slice::<$X>::try_from(self.as_scalar_slice()) {
                macro_wrap!(paste! {
                    match scalar_type {
                        macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                            ScalarType::[<$Y:upper>] => {
                                return x.cast::<$Y>(mem).map(Into::into);
                            }
                        })
                        _ => (),
                    }
                });
            }
        });
        unreachable!()
    }
    /** Casts into `scalar_type`.

    See [`.cast()`](ScalarBufferBase::cast). */
    pub fn cast_into(self, mem: Mem<'_>, scalar_type: ScalarType) -> Result<ScalarBuffer> {
        if self.scalar_type() == scalar_type {
            self.into_owned(mem)
        } else {
            self.cast(mem, scalar_type)
        }
    }
    /** Casts into `scalar_type` as a scalar arc buffer.

    See [`.cast()`](ScalarBufferBase::cast). */
    pub fn cast_shared(&self, mem: Mem<'_>, scalar_type: ScalarType) -> Result<ScalarArcBuffer> {
        if self.scalar_type() == scalar_type {
            self.to_shared(mem)
        } else {
            self.cast(mem, scalar_type).map(Into::into)
        }
    }
    /** Reinterpret as a slice with `scalar_type`.

    See [`bytemuck::cast_slice`]. */
    pub fn bitcast(&self, scalar_type: ScalarType) -> Result<ScalarSlice<'_>, PodCastError> {
        let data = self.data.as_scalar_slice().bitcast(scalar_type)?;
        Ok(ScalarSlice { data })
    }
    /** Reinterpret as a mutable slice with `scalar_type`.

    See [`bytemuck::cast_slice_mut`]. */
    pub fn bitcast_mut(
        &mut self,
        scalar_type: ScalarType,
    ) -> Result<ScalarSliceMut<'_>, PodCastError>
    where
        S: DataMut,
    {
        let data = self.data.as_scalar_slice_mut().bitcast_mut(scalar_type)?;
        Ok(ScalarSliceMut { data })
    }
    /** Copies from src.

    See [`BufferBase::copy_from_slice`]. */
    pub fn copy_from_scalar_slice(&mut self, src: &ScalarSlice) -> Result<()>
    where
        S: ScalarDataMut,
    {
        self.data
            .as_scalar_slice_mut()
            .copy_from_scalar_slice(&src.data)
    }
    /** A subslice with `range`.

    See [`BufferBase::slice`]. */
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Option<ScalarSlice<'_>> {
        let data = self.data.as_scalar_slice().slice(range)?;
        Some(ScalarSlice { data })
    }
    /** A mutable subslice with `range`.

    See [`BufferBase::slice_mut`]. */
    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> Option<ScalarSliceMut<'_>>
    where
        S: ScalarDataMut,
    {
        let data = self.data.as_scalar_slice_mut().slice(range)?;
        Some(ScalarSliceMut { data })
    }
}

#[cfg(feature = "device")]
impl ScalarSlice<'_> {
    pub(crate) fn device_buffer(&self) -> Option<&DeviceBuffer> {
        if let RawSliceInner::Device(buffer) = &self.data.raw.inner {
            Some(buffer)
        } else {
            None
        }
    }
}

#[cfg(feature = "device")]
impl ScalarSliceMut<'_> {
    pub(crate) fn device_buffer_mut(&self) -> Option<&DeviceBuffer> {
        if let RawSliceInner::Device(buffer) = &self.data.raw.inner {
            Some(buffer)
        } else {
            None
        }
    }
}

impl<T: Scalar, S: ScalarDataOwned> From<Buffer<T>> for ScalarBufferBase<S> {
    fn from(buffer: Buffer<T>) -> Self {
        let data = S::from_scalar_buffer(ScalarBufferRepr::from(buffer.data));
        Self { data }
    }
}

impl<'a, T: Scalar> From<Slice<'a, T>> for ScalarSlice<'a> {
    fn from(slice: Slice<'a, T>) -> Self {
        let data = ScalarSliceRepr {
            raw: slice.data.raw,
            scalar_type: T::SCALAR_TYPE,
            _m: PhantomData,
        };
        Self { data }
    }
}

impl<'a, T: Scalar> From<SliceMut<'a, T>> for ScalarSliceMut<'a> {
    fn from(slice: SliceMut<'a, T>) -> Self {
        let data = ScalarSliceMutRepr {
            raw: slice.data.raw,
            scalar_type: T::SCALAR_TYPE,
            _m: PhantomData,
        };
        Self { data }
    }
}

impl From<ScalarBuffer> for ScalarArcBuffer {
    fn from(buffer: ScalarBuffer) -> Self {
        Self {
            data: buffer.data.into(),
        }
    }
}

impl<T: Scalar> From<ArcBuffer<T>> for ScalarArcBuffer {
    fn from(buffer: ArcBuffer<T>) -> Self {
        Self {
            data: buffer.data.into(),
        }
    }
}

impl From<ScalarBuffer> for ScalarCowBuffer<'_> {
    fn from(buffer: ScalarBuffer) -> Self {
        Self {
            data: buffer.data.into(),
        }
    }
}

impl<'a> From<ScalarSlice<'a>> for ScalarCowBuffer<'a> {
    fn from(slice: ScalarSlice<'a>) -> Self {
        Self {
            data: slice.data.into(),
        }
    }
}

impl<'a, T: Scalar> From<CowBuffer<'a, T>> for ScalarCowBuffer<'a> {
    fn from(buffer: CowBuffer<'a, T>) -> Self {
        Self {
            data: buffer.data.into(),
        }
    }
}

impl<S: ScalarData> Debug for ScalarBufferBase<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ScalarBufferBase")
            .field("device", &self.device())
            .field("scalar_type", &self.scalar_type())
            .field("len", &self.len())
            .finish()
    }
}

#[cfg(feature = "serde")]
impl<S1: ScalarData> Serialize for ScalarBufferBase<S1> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.data.as_scalar_slice().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, S: ScalarDataOwned> Deserialize<'de> for ScalarBufferBase<S> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self {
            data: S::from_scalar_buffer(ScalarBufferRepr::deserialize(deserializer)?),
        })
    }
}

macro_for!($S in [BufferRepr, ArcBufferRepr] {
    impl<T: Scalar> Sealed for $S<T> {}
    unsafe impl<T: Scalar> Send for $S<T> {}
    unsafe impl<T: Scalar> Sync for $S<T> {}
});

macro_for!($S in [SliceRepr, SliceMutRepr, CowBufferRepr] {
    impl<T: Scalar> Sealed for $S<'_, T> {}
    unsafe impl<T: Scalar> Send for $S<'_, T> {}
    unsafe impl<T: Scalar> Sync for $S<'_, T> {}
});

/// Marker trait for buffers.
pub trait Data: ScalarData {
    /// The type of the buffer.
    type Elem: Scalar;
    #[doc(hidden)]
    fn as_slice(&self) -> SliceRepr<'_, Self::Elem>;
    #[doc(hidden)]
    fn get_slice_mut(&mut self) -> Option<SliceMutRepr<'_, Self::Elem>> {
        None
    }
    #[doc(hidden)]
    fn as_host_slice(&self) -> Option<&[Self::Elem]> {
        self.as_slice().into_host_slice()
    }
    #[doc(hidden)]
    fn try_into_buffer(self) -> Result<BufferRepr<Self::Elem>, Self>
    where
        Self: Sized,
    {
        Err(self)
    }
    #[doc(hidden)]
    fn try_into_arc_buffer(self) -> Result<ArcBufferRepr<Self::Elem>, Self>
    where
        Self: Sized,
    {
        match self.try_into_buffer() {
            Ok(buffer) => Ok(ArcBufferRepr::from_buffer(buffer)),
            Err(this) => Err(this),
        }
    }
    #[doc(hidden)]
    fn to_arc_buffer(&self, mem: Mem<'_>) -> Result<ArcBufferRepr<Self::Elem>>
    where
        Self: Sized,
    {
        Ok(ArcBufferRepr::from_buffer(self.as_slice().to_buffer(mem)?))
    }
}

/// Marker trait for mutable buffers.
pub trait DataMut: Data + ScalarDataMut {
    #[doc(hidden)]
    fn as_slice_mut(&mut self) -> SliceMutRepr<'_, Self::Elem>;
    #[doc(hidden)]
    fn as_host_slice_mut(&mut self) -> Option<&mut [Self::Elem]> {
        self.as_slice_mut().into_host_slice_mut()
    }
}

/// Marker trait for owned buffers.
pub trait DataOwned: Data {
    #[doc(hidden)]
    fn from_buffer(buffer: BufferRepr<Self::Elem>) -> Self;
    #[doc(hidden)]
    fn make_slice_mut(&mut self, mem: Mem<'_>) -> Result<SliceMutRepr<'_, Self::Elem>>;
}

/// [`Buffer`] representation.
pub struct BufferRepr<T> {
    raw: RawBuffer,
    _m: PhantomData<T>,
}

impl<T: Scalar> BufferRepr<T> {
    fn from_vec(mut vec: Vec<T>) -> Self {
        let width = std::mem::size_of::<T>();
        let ptr = vec.as_mut_ptr().cast();
        let len = vec.len() * width;
        let cap = vec.capacity() * width;
        forget(vec);
        let raw = RawBuffer {
            slice: RawSlice {
                inner: RawSliceInner::Host(RawHostSlice { ptr, len }),
            },
            cap,
            width,
        };
        Self {
            raw,
            _m: PhantomData,
        }
    }
    unsafe fn uninit(mem: Mem<'_>, len: usize) -> Result<Self> {
        let device = mem.device()?;
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| anyhow::anyhow!("Buffer size overflow."))?;
        match device.inner() {
            DeviceInner::Host => {
                let mut vec = Vec::<std::mem::MaybeUninit<T>>::new();
                vec.try_reserve_exact(len)?;
                let raw = RawBuffer {
                    slice: RawSlice {
                        inner: RawSliceInner::Host(RawHostSlice {
                            ptr: vec.as_mut_ptr().cast(),
                            len: bytes,
                        }),
                    },
                    cap: vec.capacity() * size_of::<T>(),
                    width: size_of::<T>(),
                };
                forget(vec);
                Ok(Self {
                    raw,
                    _m: PhantomData,
                })
            }
            #[cfg(feature = "device")]
            DeviceInner::Device(device) => {
                let width = size_of::<T>();
                let cap = bytes;
                let device_buffer =
                    unsafe { DeviceBuffer::uninit(device.clone(), cap, mem.kind())? };
                let raw = RawBuffer {
                    slice: RawSlice {
                        inner: RawSliceInner::Device(device_buffer),
                    },
                    cap,
                    width,
                };
                Ok(Self {
                    raw,
                    _m: PhantomData,
                })
            }
        }
    }
    fn into_vec(self) -> Result<Vec<T>> {
        match self.raw.inner {
            RawSliceInner::Host(raw) => {
                let width = size_of::<T>();
                if width == self.raw.width {
                    let vec = unsafe {
                        Vec::from_raw_parts(raw.ptr as _, raw.len / width, self.raw.cap / width)
                    };
                    std::mem::forget(self.raw);
                    Ok(vec)
                } else {
                    Ok(
                        unsafe { std::slice::from_raw_parts(raw.ptr as _, raw.len / width) }
                            .to_vec(),
                    )
                }
            }
            #[cfg(feature = "device")]
            RawSliceInner::Device(_) => self.as_slice().to_vec(),
        }
    }
}

impl<T: Scalar> ScalarData for BufferRepr<T> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr<'_> {
        ScalarSliceRepr {
            raw: self.raw.clone(),
            scalar_type: T::SCALAR_TYPE,
            _m: PhantomData,
        }
    }
}

impl<T: Scalar> ScalarDataMut for BufferRepr<T> {
    fn as_scalar_slice_mut(&mut self) -> ScalarSliceMutRepr<'_> {
        ScalarSliceMutRepr {
            raw: self.raw.clone(),
            scalar_type: T::SCALAR_TYPE,
            _m: PhantomData,
        }
    }
}

impl<T: Scalar> Data for BufferRepr<T> {
    type Elem = T;
    fn as_slice(&self) -> SliceRepr<'_, Self::Elem> {
        SliceRepr {
            raw: self.raw.slice.clone(),
            _m: PhantomData,
        }
    }
    fn get_slice_mut(&mut self) -> Option<SliceMutRepr<'_, T>> {
        Some(self.as_slice_mut())
    }
    fn try_into_buffer(self) -> Result<Self, Self>
    where
        Self: Sized,
    {
        Ok(self)
    }
}

impl<T: Scalar> DataMut for BufferRepr<T> {
    fn as_slice_mut(&mut self) -> SliceMutRepr<'_, Self::Elem> {
        SliceMutRepr {
            raw: self.raw.slice.clone(),
            _m: PhantomData,
        }
    }
}

impl<T: Scalar> DataOwned for BufferRepr<T> {
    fn from_buffer(buffer: Self) -> Self {
        buffer
    }
    fn make_slice_mut(&mut self, _mem: Mem<'_>) -> Result<SliceMutRepr<'_, T>> {
        Ok(self.as_slice_mut())
    }
}

impl<T: Scalar> TryFrom<ScalarBufferRepr> for BufferRepr<T> {
    type Error = ScalarBufferRepr;
    fn try_from(buffer: ScalarBufferRepr) -> Result<Self, Self::Error> {
        if buffer.scalar_type() == T::SCALAR_TYPE {
            Ok(Self {
                raw: buffer.raw,
                _m: Default::default(),
            })
        } else {
            Err(buffer)
        }
    }
}

/// [`Slice`] representation.
#[derive(Clone)]
pub struct SliceRepr<'a, T> {
    raw: RawSlice,
    _m: PhantomData<&'a T>,
}

impl<'a, T: Scalar> SliceRepr<'a, T> {
    fn from_host_slice(host_slice: &'a [T]) -> Self {
        let ptr = host_slice.as_ptr() as *mut u8;
        let len = std::mem::size_of_val(host_slice);
        let raw = RawSlice {
            inner: RawSliceInner::Host(RawHostSlice { ptr, len }),
        };
        Self {
            raw,
            _m: PhantomData,
        }
    }
    fn into_host_slice(self) -> Option<&'a [T]> {
        match &self.raw.inner {
            RawSliceInner::Host(raw) => {
                let slice =
                    unsafe { std::slice::from_raw_parts(raw.ptr as _, raw.len / size_of::<T>()) };
                Some(slice)
            }
            #[cfg(feature = "device")]
            _ => None,
        }
    }
    fn to_vec(&self) -> Result<Vec<T>> {
        if let Some(slice) = self.as_host_slice() {
            Ok(slice.to_vec())
        } else {
            self.copy_to(Mem::Host)?.into_vec()
        }
    }
    fn to_buffer(&self, mem: Mem<'_>) -> Result<BufferRepr<T>> {
        self.copy_to(mem)
    }
    fn copy_to(&self, mem: Mem<'_>) -> Result<BufferRepr<T>> {
        let mut output = unsafe { BufferRepr::uninit(mem, self.len())? };
        output.as_slice_mut().copy_from_slice(self)?;
        output.device().wait()?;
        Ok(output)
    }
    fn bitcast<Y: Scalar>(self) -> Result<SliceRepr<'a, Y>, bytemuck::PodCastError> {
        let raw = self.raw.bitcast(Y::SCALAR_TYPE)?;
        Ok(SliceRepr {
            raw,
            _m: PhantomData,
        })
    }
    fn slice(self, range: impl RangeBounds<usize>) -> Option<Self> {
        Some(Self {
            raw: self.raw.slice(range, T::SCALAR_TYPE)?,
            ..self
        })
    }
}

impl<'a, T: Scalar> ScalarData for SliceRepr<'a, T> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr<'_> {
        ScalarSliceRepr {
            raw: self.raw.clone(),
            scalar_type: T::SCALAR_TYPE,
            _m: PhantomData,
        }
    }
}

impl<T: Scalar> Data for SliceRepr<'_, T> {
    type Elem = T;
    fn as_slice(&self) -> SliceRepr<'_, T> {
        self.clone()
    }
}

impl<'a, T: Scalar> TryFrom<ScalarSliceRepr<'a>> for SliceRepr<'a, T> {
    type Error = ScalarSliceRepr<'a>;
    fn try_from(slice: ScalarSliceRepr<'a>) -> Result<Self, Self::Error> {
        if slice.scalar_type() == T::SCALAR_TYPE {
            Ok(Self {
                raw: slice.raw,
                _m: Default::default(),
            })
        } else {
            Err(slice)
        }
    }
}

/// [`SliceMut`] representation.
pub struct SliceMutRepr<'a, T> {
    raw: RawSlice,
    _m: PhantomData<&'a T>,
}

impl<'a, T: Scalar> SliceMutRepr<'a, T> {
    fn from_host_slice_mut(host_slice: &'a mut [T]) -> Self {
        let ptr = host_slice.as_mut_ptr().cast();
        let len = std::mem::size_of_val(host_slice);
        let raw = RawSlice {
            inner: RawSliceInner::Host(RawHostSlice { ptr, len }),
        };
        Self {
            raw,
            _m: PhantomData,
        }
    }
    fn into_host_slice_mut(self) -> Option<&'a mut [T]> {
        match &self.raw.inner {
            RawSliceInner::Host(raw) => {
                let slice = unsafe {
                    std::slice::from_raw_parts_mut(raw.ptr as _, raw.len / size_of::<T>())
                };
                Some(slice)
            }
            #[cfg(feature = "device")]
            _ => None,
        }
    }
    fn copy_from_slice(&mut self, src: &SliceRepr<'_, T>) -> Result<()> {
        if self.len() != src.len() {
            bail!(
                "source slice length ({}) does not match destination slice length ({})",
                src.len(),
                self.len()
            );
        }
        match (&mut self.raw.inner, &src.raw.inner) {
            (RawSliceInner::Host(dst), RawSliceInner::Host(src)) => {
                unsafe {
                    std::ptr::copy(src.ptr, dst.ptr, src.len);
                }
                Ok(())
            }
            #[cfg(feature = "device")]
            (RawSliceInner::Host(dst), RawSliceInner::Device(src)) => {
                src.download(unsafe { dst.as_uninit_bytes() })
            }
            #[cfg(feature = "device")]
            (RawSliceInner::Device(dst), RawSliceInner::Host(src)) => {
                dst.upload(unsafe { src.as_bytes() })
            }
            #[cfg(feature = "device")]
            (RawSliceInner::Device(dst), RawSliceInner::Device(src_buffer)) => {
                if dst.device() != src_buffer.device() {
                    return src_buffer.transfer(dst);
                }
                src_buffer.copy(dst)
            }
        }
    }
    fn bitcast<Y: Scalar>(self) -> Result<SliceMutRepr<'a, Y>, bytemuck::PodCastError> {
        let raw = self.raw.bitcast(Y::SCALAR_TYPE)?;
        Ok(SliceMutRepr {
            raw,
            _m: PhantomData,
        })
    }
    fn slice(self, range: impl RangeBounds<usize>) -> Option<Self> {
        Some(Self {
            raw: self.raw.slice(range, T::SCALAR_TYPE)?,
            ..self
        })
    }
}

impl<T: Scalar> ScalarData for SliceMutRepr<'_, T> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr<'_> {
        ScalarSliceRepr {
            raw: self.raw.clone(),
            scalar_type: T::SCALAR_TYPE,
            _m: PhantomData,
        }
    }
}

impl<T: Scalar> ScalarDataMut for SliceMutRepr<'_, T> {
    fn as_scalar_slice_mut(&mut self) -> ScalarSliceMutRepr<'_> {
        ScalarSliceMutRepr {
            raw: self.raw.clone(),
            scalar_type: T::SCALAR_TYPE,
            _m: PhantomData,
        }
    }
}

impl<T: Scalar> Data for SliceMutRepr<'_, T> {
    type Elem = T;
    fn as_slice(&self) -> SliceRepr<'_, T> {
        SliceRepr {
            raw: self.raw.clone(),
            _m: Default::default(),
        }
    }
}

impl<T: Scalar> DataMut for SliceMutRepr<'_, T> {
    fn as_slice_mut(&mut self) -> SliceMutRepr<'_, T> {
        SliceMutRepr {
            raw: self.raw.clone(),
            _m: Default::default(),
        }
    }
}

impl<'a, T: Scalar> TryFrom<ScalarSliceMutRepr<'a>> for SliceMutRepr<'a, T> {
    type Error = ScalarSliceMutRepr<'a>;
    fn try_from(slice: ScalarSliceMutRepr<'a>) -> Result<Self, Self::Error> {
        if slice.scalar_type() == T::SCALAR_TYPE {
            Ok(Self {
                raw: slice.raw,
                _m: Default::default(),
            })
        } else {
            Err(slice)
        }
    }
}

/// [`ArcBuffer`] representation.
#[derive(Clone)]
pub struct ArcBufferRepr<T> {
    raw: Arc<RawBuffer>,
    _m: PhantomData<T>,
}

impl<T: Scalar> From<BufferRepr<T>> for ArcBufferRepr<T> {
    fn from(buffer: BufferRepr<T>) -> Self {
        Self {
            #[allow(clippy::arc_with_non_send_sync)]
            raw: Arc::new(buffer.raw),
            _m: PhantomData,
        }
    }
}

impl<T: Scalar> TryFrom<ScalarArcBufferRepr> for ArcBufferRepr<T> {
    type Error = ScalarArcBufferRepr;
    fn try_from(buffer: ScalarArcBufferRepr) -> Result<Self, Self::Error> {
        if T::SCALAR_TYPE == buffer.scalar_type {
            Ok(Self {
                raw: buffer.raw,
                _m: PhantomData,
            })
        } else {
            Err(buffer)
        }
    }
}

impl<T: Scalar> ScalarData for ArcBufferRepr<T> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr<'_> {
        self.as_slice().into()
    }
    fn get_scalar_slice_mut(&mut self) -> Option<ScalarSliceMutRepr<'_>> {
        self.get_slice_mut().map(Into::into)
    }
    fn try_into_scalar_buffer(self) -> Result<ScalarBufferRepr, Self> {
        self.try_into_buffer().map(Into::into)
    }
    fn try_into_scalar_arc_buffer(self) -> Result<ScalarArcBufferRepr, Self> {
        self.try_into_arc_buffer().map(Into::into)
    }
    fn to_scalar_arc_buffer(&self, _mem: Mem<'_>) -> Result<ScalarArcBufferRepr> {
        Ok(self.clone().into())
    }
}

impl<T: Scalar> Data for ArcBufferRepr<T> {
    type Elem = T;
    fn as_slice(&self) -> SliceRepr<'_, T> {
        SliceRepr {
            raw: self.raw.slice.clone(),
            _m: PhantomData,
        }
    }
    fn get_slice_mut(&mut self) -> Option<SliceMutRepr<'_, T>> {
        let raw = Arc::get_mut(&mut self.raw)?;
        Some(SliceMutRepr {
            raw: raw.slice.clone(),
            _m: PhantomData,
        })
    }
    fn try_into_buffer(self) -> Result<BufferRepr<T>, Self> {
        match Arc::try_unwrap(self.raw) {
            Ok(raw) => Ok(BufferRepr {
                raw,
                _m: PhantomData,
            }),
            Err(raw) => Err(Self {
                raw,
                _m: PhantomData,
            }),
        }
    }
    fn try_into_arc_buffer(self) -> Result<Self, Self>
    where
        Self: Sized,
    {
        Ok(self)
    }
    fn to_arc_buffer(&self, _mem: Mem<'_>) -> Result<Self> {
        Ok(self.clone())
    }
}

impl<T: Scalar> DataOwned for ArcBufferRepr<T> {
    fn from_buffer(buffer: BufferRepr<T>) -> Self {
        Self {
            #[allow(clippy::arc_with_non_send_sync)]
            raw: Arc::new(buffer.raw),
            _m: PhantomData,
        }
    }
    fn make_slice_mut(&mut self, mem: Mem<'_>) -> Result<SliceMutRepr<'_, T>> {
        if let Some(raw) = Arc::get_mut(&mut self.raw) {
            return Ok(SliceMutRepr {
                raw: raw.slice.clone(),
                _m: PhantomData,
            });
        }
        #[allow(clippy::arc_with_non_send_sync)]
        {
            self.raw = Arc::new(self.as_slice().to_buffer(mem)?.raw);
        }
        Ok(SliceMutRepr {
            raw: self.raw.slice.clone(),
            _m: PhantomData,
        })
    }
}

/// [`CowBuffer`] representation.
pub enum CowBufferRepr<'a, T> {
    #[doc(hidden)]
    Borrowed(SliceRepr<'a, T>),
    #[doc(hidden)]
    Owned(BufferRepr<T>),
}

impl<'a, T> From<SliceRepr<'a, T>> for CowBufferRepr<'a, T> {
    fn from(slice: SliceRepr<'a, T>) -> Self {
        Self::Borrowed(slice)
    }
}

impl<'a, T> From<BufferRepr<T>> for CowBufferRepr<'a, T> {
    fn from(buffer: BufferRepr<T>) -> Self {
        Self::Owned(buffer)
    }
}

impl<'a, T: Scalar> TryFrom<ScalarCowBufferRepr<'a>> for CowBufferRepr<'a, T> {
    type Error = ScalarCowBufferRepr<'a>;
    fn try_from(buffer: ScalarCowBufferRepr<'a>) -> Result<Self, Self::Error> {
        match buffer {
            ScalarCowBufferRepr::Borrowed(slice) => SliceRepr::try_from(slice)
                .map(Into::into)
                .map_err(Into::into),
            ScalarCowBufferRepr::Owned(buffer) => BufferRepr::try_from(buffer)
                .map(Into::into)
                .map_err(Into::into),
        }
    }
}

impl<'a, T: Scalar> ScalarData for CowBufferRepr<'a, T> {
    fn as_scalar_slice(&self) -> ScalarSliceRepr<'_> {
        self.as_slice().into()
    }
    fn get_scalar_slice_mut(&mut self) -> Option<ScalarSliceMutRepr<'_>> {
        self.get_slice_mut().map(Into::into)
    }
    fn try_into_scalar_buffer(self) -> Result<ScalarBufferRepr, Self> {
        self.try_into_buffer().map(Into::into)
    }
}

impl<'a, T: Scalar> Data for CowBufferRepr<'a, T> {
    type Elem = T;
    fn as_slice(&self) -> SliceRepr<'_, T> {
        match self {
            Self::Borrowed(slice) => slice.clone(),
            Self::Owned(buffer) => buffer.as_slice(),
        }
    }
    fn get_slice_mut(&mut self) -> Option<SliceMutRepr<'_, T>> {
        match self {
            Self::Borrowed(_) => None,
            Self::Owned(buffer) => buffer.get_slice_mut(),
        }
    }
    fn try_into_buffer(self) -> Result<BufferRepr<T>, Self> {
        match self {
            Self::Borrowed(_) => Err(self),
            Self::Owned(buffer) => Ok(buffer),
        }
    }
}

impl<'a, T: Scalar> DataOwned for CowBufferRepr<'a, T> {
    fn from_buffer(buffer: BufferRepr<T>) -> Self {
        Self::Owned(buffer)
    }
    fn make_slice_mut(&mut self, mem: Mem<'_>) -> Result<SliceMutRepr<'_, T>> {
        match self {
            Self::Borrowed(slice) => {
                *self = Self::Owned(slice.to_buffer(mem)?);
                Ok(self.get_slice_mut().unwrap())
            }
            Self::Owned(buffer) => buffer.make_slice_mut(mem),
        }
    }
}

/** A buffer.

Use [`TryFrom`] to convert from [`ScalarBufferBase`].
*/
#[derive(Clone)]
pub struct BufferBase<S: Data> {
    data: S,
}

/** Buffer

See [`BufferBase`]. */
pub type Buffer<T> = BufferBase<BufferRepr<T>>;
/** Slice

See [`BufferBase`]. */
pub type Slice<'a, T> = BufferBase<SliceRepr<'a, T>>;
/** SliceMut

See [`BufferBase`]. */
pub type SliceMut<'a, T> = BufferBase<SliceMutRepr<'a, T>>;
/** ArcBuffer

Like [`Arc`], a clonable shared buffer.

See [`BufferBase`]. */
pub type ArcBuffer<T> = BufferBase<ArcBufferRepr<T>>;
/** CowBuffer

Like [`Cow`](::std::borrow::Cow), can be created from a [`Slice`] or [`Buffer`].

See [`BufferBase`]. */
pub type CowBuffer<'a, T> = BufferBase<CowBufferRepr<'a, T>>;

impl<T: Scalar, S: DataOwned<Elem = T>> From<Vec<T>> for BufferBase<S> {
    fn from(vec: Vec<T>) -> Self {
        Self::from_host_vec(vec)
    }
}

impl<'a, T: Scalar> From<&'a [T]> for Slice<'a, T> {
    fn from(host_slice: &'a [T]) -> Self {
        Self::from_host_slice(host_slice)
    }
}

impl<'a, T: Scalar> From<&'a mut [T]> for SliceMut<'a, T> {
    fn from(host_slice: &'a mut [T]) -> Self {
        Self::from_host_slice_mut(host_slice)
    }
}

impl<T: Scalar> TryFrom<ScalarBuffer> for Buffer<T> {
    type Error = ScalarBuffer;
    fn try_from(buffer: ScalarBuffer) -> Result<Self, Self::Error> {
        match buffer.data.try_into() {
            Ok(data) => Ok(Self { data }),
            Err(data) => Err(ScalarBuffer { data }),
        }
    }
}

impl<'a, T: Scalar> TryFrom<ScalarSlice<'a>> for Slice<'a, T> {
    type Error = ScalarSlice<'a>;
    fn try_from(slice: ScalarSlice<'a>) -> Result<Self, Self::Error> {
        match slice.data.try_into() {
            Ok(data) => Ok(Self { data }),
            Err(data) => Err(ScalarSlice { data }),
        }
    }
}

impl<'a, T: Scalar> TryFrom<ScalarSliceMut<'a>> for SliceMut<'a, T> {
    type Error = ScalarSliceMut<'a>;
    fn try_from(slice: ScalarSliceMut<'a>) -> Result<Self, Self::Error> {
        match slice.data.try_into() {
            Ok(data) => Ok(Self { data }),
            Err(data) => Err(ScalarSliceMut { data }),
        }
    }
}

impl<T: Scalar> TryFrom<ScalarArcBuffer> for ArcBuffer<T> {
    type Error = ScalarArcBuffer;
    fn try_from(buffer: ScalarArcBuffer) -> Result<Self, Self::Error> {
        match buffer.data.try_into() {
            Ok(data) => Ok(Self { data }),
            Err(data) => Err(ScalarArcBuffer { data }),
        }
    }
}

impl<T: Scalar> From<Buffer<T>> for ArcBuffer<T> {
    fn from(buffer: Buffer<T>) -> Self {
        Self {
            data: buffer.data.into(),
        }
    }
}

impl<'a, T: Scalar> From<Slice<'a, T>> for CowBuffer<'a, T> {
    fn from(slice: Slice<'a, T>) -> Self {
        Self {
            data: slice.data.into(),
        }
    }
}

impl<T: Scalar> From<Buffer<T>> for CowBuffer<'_, T> {
    fn from(buffer: Buffer<T>) -> Self {
        Self {
            data: buffer.data.into(),
        }
    }
}

impl<'a, T: Scalar> TryFrom<ScalarCowBuffer<'a>> for CowBuffer<'a, T> {
    type Error = ScalarCowBuffer<'a>;
    fn try_from(buffer: ScalarCowBuffer<'a>) -> Result<Self, Self::Error> {
        match buffer.data.try_into() {
            Ok(data) => Ok(Self { data }),
            Err(data) => Err(ScalarCowBuffer { data }),
        }
    }
}

impl<T: Scalar, S: DataOwned<Elem = T>> Default for BufferBase<S> {
    fn default() -> Self {
        Self::zeros(Mem::Host, 0).unwrap()
    }
}

impl<T: Scalar, S: DataOwned<Elem = T>> BufferBase<S> {
    /// Allocate a buffer.
    ///
    /// # Safety
    /// The buffer will not be initialized.
    ///
    /// # Errors
    /// - [`DeviceLost`]
    /// - [`DeviceBufferTooLarge`]
    /// - [`OutOfDeviceMemory`]
    ///
    /// See [`zeros()`](Buffer::zeros) for a safe alternative.
    pub unsafe fn uninit(mem: Mem<'_>, len: usize) -> Result<Self> {
        let data = S::from_buffer(unsafe { BufferRepr::uninit(mem, len)? });
        Ok(Self { data })
    }
    /** Create a buffer filled with `elem`

    # Errors
    - [`DeviceLost`]
    - [`DeviceBufferTooLarge`]
    - [`OutOfDeviceMemory`]

    See [`.fill()`](BufferBase::fill).
    */
    pub fn from_elem(mem: Mem<'_>, len: usize, elem: T) -> Result<Self> {
        if !matches!(mem, Mem::Device(_)) {
            return Self::allocate_with(mem, len, |_| elem);
        }
        let mut output = unsafe { Buffer::uninit(mem, len)? };
        output.fill(elem)?;
        output.device().wait()?;
        Ok(Self {
            data: S::from_buffer(output.data),
        })
    }
    /** Create a buffer filled with 0's.

    See [`from_elem`](BufferBase::from_elem). */
    pub fn zeros(mem: Mem<'_>, len: usize) -> Result<Self> {
        Self::from_elem(mem, len, T::zero())
    }
    /** Create a buffer filled with 1's.

    See [`from_elem`](BufferBase::from_elem). */
    pub fn ones(mem: Mem<'_>, len: usize) -> Result<Self> {
        Self::from_elem(mem, len, T::one())
    }
    /// Initializes CPU-accessible storage directly, in index order.
    pub fn allocate_with(
        mem: Mem<'_>,
        len: usize,
        mut init: impl FnMut(usize) -> T,
    ) -> Result<Self> {
        if matches!(mem, Mem::Device(_)) {
            bail!("Use a mapped placement for allocate_with.");
        }
        let output = unsafe { BufferRepr::<T>::uninit(mem, len)? };
        {
            let mapping = Mapping::new(&output.raw.slice, true, true)?;
            let ptr = mapping.ptr().cast::<T>();
            for i in 0..len {
                // SAFETY: Each slot lies in the allocation and is written before exposure.
                unsafe {
                    ptr.add(i).write(init(i));
                }
            }
        }
        Ok(Self {
            data: S::from_buffer(output),
        })
    }
    /// Adopts a host vector without copying.
    pub fn from_host_vec(vec: Vec<T>) -> Self {
        let data = S::from_buffer(BufferRepr::from_vec(vec));
        Self { data }
    }
    /// Create a buffer from a [`Buffer`].
    pub fn from_buffer(buffer: Buffer<T>) -> Self {
        let data = S::from_buffer(buffer.data);
        Self { data }
    }
}

impl<'a, T: Scalar> Slice<'a, T> {
    /// Create a slice from a `&[T]`.
    pub fn from_host_slice(host_slice: &'a [T]) -> Self {
        let data = SliceRepr::from_host_slice(host_slice);
        Self { data }
    }
}

impl<'a, T: Scalar> SliceMut<'a, T> {
    /// Create a mutable slice from a `&mut [T]`.
    pub fn from_host_slice_mut(host_slice: &'a mut [T]) -> Self {
        let data = SliceMutRepr::from_host_slice_mut(host_slice);
        Self { data }
    }
}

impl<T: Scalar, S: Data<Elem = T>> BufferBase<S> {
    /// The device.
    pub fn device(&self) -> Device {
        self.data.device()
    }
    /// The scalar type.
    pub fn scalar_type(&self) -> ScalarType {
        self.data.scalar_type()
    }
    /// The length.
    pub fn len(&self) -> usize {
        self.data.len()
    }
    /// Is the buffer empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Borrow as a slice.
    pub fn as_slice(&self) -> Slice<'_, T> {
        let data = self.data.as_slice();
        Slice { data }
    }
    /// Borrow as a mutable slice.
    pub fn as_slice_mut(&mut self) -> SliceMut<'_, T>
    where
        S: DataMut,
    {
        let data = self.data.as_slice_mut();
        SliceMut { data }
    }
    /// Borrows as a mutable slice if possible.
    pub fn get_slice_mut(&mut self) -> Option<SliceMut<'_, T>> {
        Some(SliceMut {
            data: self.data.get_slice_mut()?,
        })
    }
    /// Borrows as a mutable slice. Uses `mem` if a copy is needed.
    pub fn make_slice_mut(&mut self, mem: Mem<'_>) -> Result<SliceMut<'_, T>>
    where
        S: DataOwned,
    {
        Ok(SliceMut {
            data: self.data.make_slice_mut(mem)?,
        })
    }
    /** Borrow as a host slice.

    Returns [`None`] if not on the host. */
    pub fn as_host_slice(&self) -> Option<&[T]> {
        self.data.as_host_slice()
    }
    /** Borrow as a mutable host slice.

    Returns [`None`] if not on the host. */
    pub fn as_host_slice_mut(&mut self) -> Option<&mut [T]>
    where
        S: DataMut,
    {
        self.data.as_host_slice_mut()
    }
    /// Borrow as a scalar slice.
    pub fn as_scalar_slice(&self) -> ScalarSlice<'_> {
        let data = self.data.as_scalar_slice();
        ScalarSlice { data }
    }
    /// Borrow as a mutable scalar slice.
    pub fn as_scalar_slice_mut(&mut self) -> ScalarSliceMut<'_>
    where
        S: DataMut,
    {
        let data = self.data.as_scalar_slice_mut();
        ScalarSliceMut { data }
    }
    /// Copies into the requested memory.
    pub fn to_owned(&self, mem: Mem<'_>) -> Result<Buffer<T>> {
        Ok(Buffer {
            data: self.data.as_slice().copy_to(mem)?,
        })
    }
    /// Reuses owned storage if it matches `mem`. Copies otherwise.
    pub fn into_owned(self, mem: Mem<'_>) -> Result<Buffer<T>> {
        if !self.data.as_slice().raw.matches(mem)? {
            return self.to_owned(mem);
        }
        match self.data.try_into_buffer() {
            Ok(data) => Ok(Buffer { data }),
            Err(data) => Self { data }.to_owned(mem),
        }
    }
    /// Reuses storage when possible and returns shared ownership.
    pub fn into_shared(self, mem: Mem<'_>) -> Result<ArcBuffer<T>> {
        if !self.data.as_slice().raw.matches(mem)? {
            return self.to_owned(mem).map(Into::into);
        }
        let data = match self.data.try_into_arc_buffer() {
            Ok(data) => data,
            Err(data) => data.to_arc_buffer(mem)?,
        };
        Ok(ArcBuffer { data })
    }
    /// Clones shared storage if it matches `mem`. Copies otherwise.
    pub fn to_shared(&self, mem: Mem<'_>) -> Result<ArcBuffer<T>> {
        if !self.data.as_slice().raw.matches(mem)? {
            return self.to_owned(mem).map(Into::into);
        }
        Ok(ArcBuffer {
            data: self.data.to_arc_buffer(mem)?,
        })
    }
    /// Returns host storage, copying if the allocation cannot become a `Vec`.
    pub fn into_host_vec(self) -> Result<Vec<T>> {
        match self.data.try_into_buffer() {
            Ok(data) => data.into_vec(),
            Err(data) => data.as_slice().to_vec(),
        }
    }
    /// Copies into a host vector.
    pub fn to_host_vec(&self) -> Result<Vec<T>> {
        self.data.as_slice().to_vec()
    }

    /// Waits for GPU work and reserves CPU read access.
    pub fn map_read(&self) -> Result<ReadGuard<'_, T>> {
        self.map_read_impl(true)
    }
    /// Reserves CPU read access without waiting for GPU work.
    pub fn try_map_read(&self) -> Result<ReadGuard<'_, T>> {
        self.map_read_impl(false)
    }
    fn map_read_impl(&self, wait: bool) -> Result<ReadGuard<'_, T>> {
        Ok(ReadGuard {
            mapping: Mapping::new(&self.data.as_slice().raw, false, wait)?,
            _borrow: PhantomData,
        })
    }
    /// Waits for GPU work and reserves exclusive CPU access.
    pub fn map_write(&mut self) -> Result<WriteGuard<'_, T>>
    where
        S: DataMut,
    {
        self.map_write_impl(true)
    }
    /// Reserves exclusive CPU access without waiting for GPU work.
    pub fn try_map_write(&mut self) -> Result<WriteGuard<'_, T>>
    where
        S: DataMut,
    {
        self.map_write_impl(false)
    }
    fn map_write_impl(&mut self, wait: bool) -> Result<WriteGuard<'_, T>>
    where
        S: DataMut,
    {
        Ok(WriteGuard {
            mapping: Mapping::new(&self.data.as_slice_mut().raw, true, wait)?,
            _borrow: PhantomData,
        })
    }
    /// Returns the selected Vulkan memory properties, if an allocation exists.
    pub fn memory_properties(&self) -> Option<MemoryProperties> {
        self.data.as_slice().raw.memory_properties()
    }
    /// Whether this allocation supports CPU mapping.
    pub fn is_host_visible(&self) -> bool {
        self.data.as_slice().raw.is_host_visible()
    }
    /** Fills with `elem`.

    # Errors
    - [`DeviceLost`]
    - The kernel could not be dispatched.
        - This may require [`Features`] for the type. */
    pub fn fill(&mut self, elem: T) -> Result<()>
    where
        S: DataMut,
    {
        if self.is_empty() {
            return Ok(());
        }
        if let Some(y) = self.as_host_slice_mut() {
            for y in y.iter_mut() {
                *y = elem;
            }
            return Ok(());
        }
        #[cfg(not(feature = "device"))]
        {
            unreachable!()
        }
        #[cfg(feature = "device")]
        {
            fn copied_bytes<X: Scalar, Y: Scalar>(x: X) -> Y {
                assert!(size_of::<Y>() >= size_of::<X>());
                let mut y = Y::default();
                for (y, x) in bytemuck::bytes_of_mut(&mut y)
                    .iter_mut()
                    .zip(bytemuck::bytes_of(&x).iter().cycle())
                {
                    *y = *x;
                }
                y
            }
            let device = self.device();
            let features = device.info().unwrap().features();
            if features.contains(Features::INT64) {
                if let Ok(y) = self.bitcast_mut::<u64>() {
                    let x = copied_bytes(elem);
                    return kernels::fill_u64::builder()?.build(device)?.dispatch(x, y);
                }
            }
            if let Ok(y) = self.bitcast_mut::<u32>() {
                let x = copied_bytes(elem);
                return kernels::fill_u32::builder()?.build(device)?.dispatch(x, y);
            }
            if features.contains(
                Features::INT16
                    .union(Features::BUFFER16)
                    .union(Features::PUSH_CONSTANT16),
            ) {
                if let Ok(y) = self.bitcast_mut::<u16>() {
                    let x = copied_bytes(elem);
                    return kernels::fill_u16::builder()?.build(device)?.dispatch(x, y);
                }
            }
            if features.contains(
                Features::INT8
                    .union(Features::BUFFER8)
                    .union(Features::PUSH_CONSTANT8),
            ) {
                if let Ok(y) = self.bitcast_mut::<u8>() {
                    let x = copied_bytes(elem);
                    return kernels::fill_u8::builder()?.build(device)?.dispatch(x, y);
                }
            }
            if self.bitcast::<u16>().is_ok() {
                bail!("Device {device:?} does not support 16 bit operations!");
            } else {
                bail!("Device {device:?} does not support 8 bit operations!");
            }
        }
    }
    /** Casts to `Y`.

    The source and destination must use the same device.

    # Errors
    - [`DeviceLost`]
    - The kernel could not be dispatched.
        - This may require [`Features`] for the type. */
    pub fn cast<Y: Scalar>(&self, mem: Mem<'_>) -> Result<Buffer<Y>> {
        if self.device() != mem.device()? {
            bail!("Copy to the target device before casting.");
        }
        if let Some(x) = self.as_host_slice() {
            use half::slice::HalfFloatSliceExt;
            macro_for!($H in [f16, bf16] {
                if T::SCALAR_TYPE == <$H>::SCALAR_TYPE {
                    let x = bytemuck::cast_slice::<_, $H>(x);
                    let output = match Y::SCALAR_TYPE {
                        ScalarType::F32 => Some(ScalarBuffer::from(Buffer::from_host_vec(x.to_f32_vec()))),
                        ScalarType::F64 => Some(ScalarBuffer::from(Buffer::from_host_vec(x.to_f64_vec()))),
                        _ => None,
                    };
                    if let Some(output) = output {
                        return Ok(output.try_into().ok().unwrap());
                    }
                }
            });
            return Buffer::allocate_with(mem, x.len(), |i| x[i].cast());
        }
        let mut output = unsafe { Buffer::uninit(mem, self.len())? };
        self.as_slice().cast_impl(&mut output.as_slice_mut())?;
        output.device().wait()?;
        Ok(output)
    }
    /** Casts into `Y`.

    See [`.cast()`](BufferBase::cast). */
    pub fn cast_into<Y: Scalar>(self, mem: Mem<'_>) -> Result<Buffer<Y>> {
        if T::SCALAR_TYPE == Y::SCALAR_TYPE {
            let buffer = self.into_owned(mem)?;
            Ok(ScalarBuffer::from(buffer).try_into().ok().unwrap())
        } else {
            self.cast(mem)
        }
    }
    /** Casts to `Y` as an arc buffer.

    See [`.cast()`](BufferBase::cast). */
    pub fn cast_shared<Y: Scalar>(&self, mem: Mem<'_>) -> Result<ArcBuffer<Y>> {
        if T::SCALAR_TYPE == Y::SCALAR_TYPE {
            let buffer = self.to_shared(mem)?;
            Ok(ScalarArcBuffer::from(buffer).try_into().ok().unwrap())
        } else {
            self.cast(mem).map(Into::into)
        }
    }
    /** Reinterpret as a slice with type `Y`.

    See [`bytemuck::cast_slice`]. */
    pub fn bitcast<Y: Scalar>(&self) -> Result<Slice<'_, Y>, bytemuck::PodCastError> {
        let data = self.data.as_slice().bitcast()?;
        Ok(Slice { data })
    }
    /** Reinterpret as a mutable slice with type `Y`.

    See [`bytemuck::cast_slice_mut`]. */
    pub fn bitcast_mut<Y: Scalar>(&mut self) -> Result<SliceMut<'_, Y>, bytemuck::PodCastError>
    where
        S: DataMut,
    {
        let data = self.data.as_slice_mut().bitcast()?;
        Ok(SliceMut { data })
    }
    /** Copies from src.

    # Errors
    - `src` is not the same length.
    - [`DeviceLost`]
    - The kernel could not be dispatched.
    */
    pub fn copy_from_slice(&mut self, src: &Slice<'_, T>) -> Result<()>
    where
        S: DataMut,
    {
        self.data.as_slice_mut().copy_from_slice(&src.data)?;
        Ok(self.device().wait()?)
    }
    /** A subslice with `range`.

    Returns None if range is out of bounds.

    See [`<[_]>::get()`](https://doc.rust-lang.org/std/primitive.slice.html#method.get). */
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Option<Slice<'_, T>> {
        let data = self.data.as_slice().slice(range)?;
        Some(Slice { data })
    }
    /** A mutable subslice with `range`.

    Returns None if range is out of bounds.

    See [`<[_]>::get_mut()`](https://doc.rust-lang.org/std/primitive.slice.html#method.get_mut). */
    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> Option<SliceMut<'_, T>>
    where
        S: DataMut,
    {
        let data = self.data.as_slice_mut().slice(range)?;
        Some(SliceMut { data })
    }
}

impl<T: Scalar> Slice<'_, T> {
    fn cast_impl<Y: Scalar>(&self, output: &mut SliceMut<'_, Y>) -> Result<()> {
        debug_assert_eq!(self.len(), output.len());
        if output.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "device")]
        {
            device_scalar_buffer_cast_impl(self.as_scalar_slice(), output.as_scalar_slice_mut())
        }
        #[cfg(not(feature = "device"))]
        {
            unreachable!()
        }
    }
}

#[cfg(feature = "device")]
fn device_scalar_buffer_cast_impl(x: ScalarSlice<'_>, y: ScalarSliceMut<'_>) -> Result<()> {
    macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        let x = match Slice::<$X>::try_from(x) {
            Ok(x) => {
                macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
                    let y = match SliceMut::<$Y>::try_from(y) {
                        Ok(y) => {
                            let builder = paste! {
                                kernels::[<cast_ $X _ $Y>]::builder()?
                            };
                            return builder.build(y.device())?.dispatch(x, y);
                        }
                        Err(y) => y,
                    };
                });
                let _ = y;
                unreachable!()
            }
            Err(x) => x,
        };
    });
    let _ = x;
    unreachable!()
}

impl<S: Data> Debug for BufferBase<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("BufferBase")
            .field("device", &self.device())
            .field("scalar_type", &self.scalar_type())
            .field("len", &self.len())
            .finish()
    }
}

#[cfg(feature = "serde")]
impl<S1: Data> Serialize for BufferBase<S1> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.data.as_scalar_slice().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Scalar, S: DataOwned<Elem = T>> Deserialize<'de> for BufferBase<S> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;
        let scalar_buffer = ScalarBufferRepr::deserialize(deserializer)?;
        let buffer = BufferRepr::try_from(scalar_buffer).map_err(|e| {
            D::Error::custom(format!(
                "Expected {:?}, found {:?}!",
                T::SCALAR_TYPE,
                e.scalar_type()
            ))
        })?;
        Ok(Self {
            data: S::from_buffer(buffer),
        })
    }
}

#[cfg(feature = "device")]
#[module]
#[krnl(crate=crate)]
mod kernels {
    use dry::macro_for;
    #[cfg(not(target_arch = "spirv"))]
    use krnl_core;
    use krnl_core::macros::kernel;
    #[cfg(target_arch = "spirv")]
    use krnl_core::{
        half::{bf16, f16},
        scalar::Scalar,
    };
    use paste::paste;

    macro_for!($T in [u8, u16, u32, u64]  {
       paste! {
           #[kernel]
           pub fn [<fill_ $T>](x: $T, #[item] y: &mut $T) {
               *y = x;
           }
       }
    });

    macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            paste! {
                #[kernel]
                pub fn [<cast_ $X _ $Y>](#[item] x: $X, #[item] y: &mut $Y) {
                    *y = x.cast();
                }
            }
        });
    });
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::*;
    use serde_test::{assert_tokens, Token};

    #[test]
    fn buffer_serde_tokens() {
        let input = vec![1u32, 2, 3, 4];
        let items: Vec<u64> = input
            .chunks(2)
            .map(|chunk| {
                let mut item = 0u64;
                bytemuck::bytes_of_mut(&mut item).copy_from_slice(bytemuck::cast_slice(chunk));
                item.to_be()
            })
            .collect();
        let buffer = Buffer::from_host_vec(input.clone());
        let tokens = [
            Token::TupleStruct {
                name: "Buffer",
                len: 3,
            },
            Token::Str("U32"),
            Token::U64(4),
            Token::Seq { len: Some(2) },
            Token::U64(items[0]),
            Token::U64(items[1]),
            Token::SeqEnd,
            Token::TupleStructEnd,
        ];

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(transparent)]
        struct BufferWrap(Buffer<u32>);

        impl PartialEq for BufferWrap {
            fn eq(&self, other: &Self) -> bool {
                self.0.as_host_slice().unwrap() == other.0.as_host_slice().unwrap()
            }
        }

        impl Eq for BufferWrap {}

        assert_tokens(&BufferWrap(buffer), &tokens);
    }

    #[test]
    fn buffer_serde_json() {
        let x_vec = vec![1u32, 2, 3, 4];
        let string = serde_json::to_string(&Buffer::from_host_vec(x_vec.clone())).unwrap();
        let y_vec = serde_json::from_str::<Buffer<u32>>(&string)
            .unwrap()
            .into_host_vec()
            .unwrap();
        assert_eq!(x_vec, y_vec);
    }

    #[test]
    fn buffer_serde_bincode2() {
        let x_vec = vec![1u32, 2, 3, 4];
        let bytes = bincode2::serialize(&Buffer::from_host_vec(x_vec.clone())).unwrap();
        let y_vec = bincode2::deserialize::<Buffer<u32>>(&bytes)
            .unwrap()
            .into_host_vec()
            .unwrap();
        assert_eq!(x_vec, y_vec);
    }
}
