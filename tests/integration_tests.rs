use dry::macro_for;
use half::{bf16, f16};
use krnl::buffer::{Buffer, MapError, SliceMut};
#[cfg(not(target_family = "wasm"))]
use krnl::{buffer::ArcBuffer, device::Features, scalar::ScalarType};
use krnl::{
    buffer::{Mem, Slice},
    device::Device,
    scalar::Scalar,
};
#[cfg(not(target_family = "wasm"))]
use libtest_mimic::{Arguments, Trial};
use paste::paste;
#[cfg(not(target_family = "wasm"))]
use std::{mem::size_of, str::FromStr};
#[cfg(target_family = "wasm")]
use wasm_bindgen_test::wasm_bindgen_test as test;

#[cfg(all(target_family = "wasm", run_in_browser))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg(target_family = "wasm")]
fn main() {}

#[cfg(not(target_family = "wasm"))]
fn main() {
    let args = Arguments::from_args();
    let tests = if cfg!(feature = "device") && !cfg!(miri) {
        let devices: Vec<_> = [Device::builder().build().unwrap()]
            .into_iter()
            .chain((1..).map_while(|i| Device::builder().index(i).build().ok()))
            .collect();
        if devices.is_empty() {
            panic!("No device!");
        }
        let device_infos: Vec<_> = devices.iter().map(|x| x.info().unwrap()).collect();
        println!("devices: {device_infos:#?}");
        let krnl_device = std::env::var("KRNL_DEVICE");
        let device_index = if let Ok(krnl_device) = krnl_device.as_ref() {
            usize::from_str(krnl_device).unwrap()
        } else {
            0
        };
        let device_index2 = usize::from(device_index == 0);
        println!("KRNL_DEVICE = {krnl_device:?}");
        println!("testing device {device_index}");
        let device = devices.first().unwrap();
        let device2 = devices.get(1);
        if device2.is_some() {
            println!("using device {device_index2} for `buffer_device_to_device`");
        }
        tests(&Device::host(), None)
            .into_iter()
            .chain(tests(device, device2))
            .collect()
    } else {
        tests(&Device::host(), None).into_iter().collect()
    };
    libtest_mimic::run(&args, tests).exit()
}

#[cfg(not(target_family = "wasm"))]
fn device_test(device: &Device, name: &str, f: impl Fn(Device) + Send + Sync + 'static) -> Trial {
    let name = format!(
        "{name}_{}",
        if device.is_host() { "host" } else { "device" }
    );
    let device = device.clone();
    Trial::test(name, move || {
        f(device);
        Ok(())
    })
}

#[cfg(not(target_family = "wasm"))]
fn tests(device: &Device, device2: Option<&Device>) -> impl IntoIterator<Item = Trial> {
    buffer_tests(device, device2)
}

#[cfg(not(target_family = "wasm"))]
fn buffer_tests(device: &Device, device2: Option<&Device>) -> impl IntoIterator<Item = Trial> {
    let features = device
        .info()
        .map(|x| x.features())
        .unwrap_or(Features::empty());
    let mut tests = Vec::new();

    tests.push(device_test(device, "buffer_from_vec", buffer_from_vec));
    tests.push(device_test(device, "buffer_placement", buffer_placement));
    tests.push(device_test(
        device,
        "buffer_placement_errors",
        buffer_placement_errors,
    ));
    if device.is_device() {
        tests.push(device_test(
            device,
            "buffer_mapping_conflicts",
            buffer_mapping_conflicts,
        ));
        tests.push(device_test(
            device,
            "buffer_concurrent_downloads",
            buffer_concurrent_downloads,
        ));
    }

    if device.is_device() {
        #[cfg(feature = "device")]
        tests.push(Trial::test("device_buffer_too_large", {
            let device = device.clone();
            move || {
                device_buffer_too_large(device);
                Ok(())
            }
        }));
        tests.push(
            Trial::test("buffer_device_to_device", {
                let device = device.clone();
                let device2 = device2.cloned();
                move || {
                    buffer_transfer(device, device2.unwrap());
                    Ok(())
                }
            })
            .with_ignored_flag(device2.is_none()),
        );
    }

    macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        paste! {
            {
                let ignore = if device.is_host() {
                    false
                } else {
                    match size_of::<$T>() {
                        1 => !features.contains(Features::INT8 | Features::BUFFER8 | Features::PUSH_CONSTANT8),
                        2 => !features.contains(Features::INT16 | Features::BUFFER16 | Features::PUSH_CONSTANT16),
                        4 => false,
                        8 => !features.contains(Features::INT64),
                        _ => unreachable!(),
                    }
                };
                let trial = paste! {
                    device_test(device, stringify!([<buffer_fill_ $T>]), [<buffer_fill>]::<$T>)
                };
                tests.push(trial.with_ignored_flag(ignore));
            }
        }
    });

    fn buffer_cast_features(x: ScalarType, y: ScalarType) -> Features {
        fn features(ty: ScalarType) -> Features {
            use ScalarType::*;
            match ty {
                U8 | I8 => Features::INT8 | Features::BUFFER8,
                U16 | I16 => Features::INT16 | Features::BUFFER16,
                F16 | BF16 => Features::INT8 | Features::INT16 | Features::BUFFER16,
                U32 | I32 | F32 => Features::empty(),
                U64 | I64 => Features::INT64,
                F64 => Features::INT64 | Features::FLOAT64,
                _ => unreachable!(),
            }
        }
        features(x).union(features(y))
    }

    macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
            {
                let ignore = !device.is_host() && !features.contains(buffer_cast_features($X::SCALAR_TYPE, $Y::SCALAR_TYPE));
                paste! {
                    let trial = device_test(device, stringify!([<buffer_cast_ $X _ $Y>]), [<buffer_cast>]::<$X, $Y>);
                    tests.push(trial.with_ignored_flag(ignore));
                    let trial = device_test(device, stringify!([<buffer_bitcast_ $X _ $Y>]), [<buffer_bitcast>]::<$X, $Y>);
                    tests.push(trial.with_ignored_flag(ignore));
                }
            }
        });
    });

    tests
}

fn test_mem(device: &Device) -> Mem<'_> {
    if device.is_host() {
        Mem::Host
    } else {
        Mem::Device(device)
    }
}

fn buffer_test_lengths() -> impl ExactSizeIterator<Item = usize> {
    [0, 1, 3, 4, 16, 67, 157].into_iter()
}
fn buffer_transfer_test_lengths() -> impl ExactSizeIterator<Item = usize> {
    #[cfg(not(miri))]
    {
        [0, 1, 3, 4, 16, 345, 9_337_791].into_iter()
    }
    #[cfg(miri)]
    {
        [0, 1, 3, 4, 16, 345].into_iter()
    }
}

fn buffer_from_vec(device: Device) {
    let n = buffer_transfer_test_lengths().last().unwrap();
    let x = (10..20).cycle().take(n).collect::<Vec<_>>();
    for n in buffer_transfer_test_lengths() {
        let x = &x[..n];
        let y = Slice::from(x)
            .to_owned(test_mem(&device))
            .unwrap()
            .into_host_vec()
            .unwrap();
        assert_eq!(y.len(), n);
        if x != y.as_slice() {
            for (i, (x, y)) in x.iter().zip(y).enumerate() {
                assert_eq!(&y, x, "i: {i}, n: {n}");
            }
        }
    }
}

#[cfg(feature = "device")]
fn device_buffer_too_large(device: Device) {
    use krnl::buffer::error::DeviceBufferTooLarge;
    let error = unsafe {
        Buffer::<u32>::uninit(Mem::Device(&device), (i32::MAX / 4 + 1).try_into().unwrap())
    }
    .err()
    .unwrap();
    error.downcast_ref::<DeviceBufferTooLarge>().unwrap();
}

#[cfg(not(target_family = "wasm"))]
fn buffer_transfer(device: Device, device2: Device) {
    let n = buffer_transfer_test_lengths().last().unwrap();
    let x = (10..20).cycle().take(n).collect::<Vec<_>>();
    for n in buffer_transfer_test_lengths() {
        let x = &x[..n];
        let y = Slice::from(x)
            .to_owned(test_mem(&device))
            .unwrap()
            .to_owned(test_mem(&device2))
            .unwrap()
            .into_host_vec()
            .unwrap();
        if x != y.as_slice() {
            for (i, (x, y)) in x.iter().zip(y).enumerate() {
                assert_eq!(&y, x, "i: {i}, n: {n}");
            }
        }
    }
}

fn buffer_fill<T: Scalar>(device: Device) {
    let elem = T::one();
    let n = buffer_test_lengths().last().unwrap();
    let x = (10..20)
        .cycle()
        .map(|x| T::from_u32(x).unwrap())
        .take(n)
        .collect::<Vec<_>>();
    for n in buffer_test_lengths() {
        let x = &x[..n];
        let mut y = Slice::from(x).to_owned(test_mem(&device)).unwrap();
        y.fill(elem).unwrap();
        let y: Vec<T> = y.into_host_vec().unwrap();
        for (i, y) in y.into_iter().enumerate() {
            assert_eq!(y, elem, "i: {i}, n: {n}");
        }
    }
}

fn buffer_cast<X: Scalar, Y: Scalar>(device: Device) {
    let n = buffer_test_lengths().last().unwrap();
    let x = (10..20)
        .cycle()
        .map(|x| X::from_u32(x).unwrap())
        .take(n)
        .collect::<Vec<_>>();
    for n in buffer_test_lengths() {
        let x = &x[..n];
        let y = Slice::<X>::from(x)
            .into_owned(test_mem(&device))
            .unwrap()
            .cast_into::<Y>(test_mem(&device))
            .unwrap()
            .into_host_vec()
            .unwrap();
        for (i, (x, y)) in x.iter().zip(y.iter()).enumerate() {
            assert_eq!(*y, x.cast::<Y>(), "i: {i}, n: {n}");
        }
    }
}

fn buffer_bitcast<X: Scalar, Y: Scalar>(device: Device) {
    let x_host = vec![0u64; 16];
    let x_host: &[X] = &bytemuck::cast_slice(&x_host)[..16];
    let x = Slice::from(x_host).to_owned(test_mem(&device)).unwrap();
    for i in 0..=16 {
        for range in [i..16, 0..i] {
            let bytemuck_result =
                bytemuck::try_cast_slice::<X, Y>(&x_host[range.clone()]).map(|_| ());
            let result = x.slice(range).unwrap().bitcast::<Y>().map(|_| ());
            #[cfg(miri)]
            let _ = (bytemuck_result, result);
            #[cfg(not(miri))]
            assert_eq!(result, bytemuck_result);
        }
    }
}

fn mapped_placements(device: &Device) -> Vec<Mem<'_>> {
    if device.is_host() {
        vec![Mem::Host]
    } else {
        vec![
            Mem::Shared(device),
            Mem::Upload(device),
            Mem::Readback(device),
        ]
    }
}

fn buffer_placement(device: Device) {
    for mem in mapped_placements(&device) {
        // Empty, single-element, odd, and 256-byte allocation boundaries.
        for len in [0, 1, 3, 256 / std::mem::size_of::<u32>()] {
            let mut next = 0;
            let mut x = Buffer::allocate_with(mem, len, |i| {
                assert_eq!(i, next);
                next += 1;
                i as u32
            })
            .unwrap();
            assert_eq!(next, len);
            assert!(x.is_host_visible());
            if len != 0 && device.is_device() {
                assert!(x.memory_properties().unwrap().host_visible);
            }
            assert_eq!(
                &*x.map_read().unwrap(),
                &(0..len as u32).collect::<Vec<_>>()
            );
            x.try_map_write().unwrap().fill(1);
            assert_eq!(&*x.try_map_read().unwrap(), &vec![1; len]);
            if len > 1 {
                let mut tail = x.slice_mut(1..).unwrap();
                tail.map_write().unwrap().fill(2);
                let bytes = tail.bitcast::<u8>().unwrap();
                assert_eq!(
                    &*bytes.map_read().unwrap(),
                    bytemuck::cast_slice::<_, u8>(&vec![2u32; len - 1])
                );
            }
            let y = x.to_owned(test_mem(&device)).unwrap();
            if !y.is_host_visible() {
                assert_eq!(
                    y.try_map_read().err().unwrap().downcast_ref::<MapError>(),
                    Some(&MapError::Unsupported)
                );
            }
            let mut output = vec![0; len];
            SliceMut::from(output.as_mut_slice())
                .copy_from_slice(&y.as_slice())
                .unwrap();
            assert_eq!(output, x.to_host_vec().unwrap());
            #[cfg(feature = "serde")]
            assert_eq!(
                serde_json::to_string(&x).unwrap(),
                serde_json::to_string(&Buffer::from_host_vec(output.clone())).unwrap()
            );
            // Both casts dispatch kernels on a GPU. The intermediate stays on that GPU.
            let y = x.cast::<f32>(mem).unwrap().cast::<u32>(mem).unwrap();
            assert_eq!(&*y.map_read().unwrap(), output.as_slice());
            x.fill(3).unwrap();
            match x.try_map_read() {
                Ok(read) => assert_eq!(&*read, &vec![3; len]),
                Err(error) => assert_eq!(error.downcast_ref::<MapError>(), Some(&MapError::Busy)),
            }
            assert_eq!(&*x.map_read().unwrap(), &vec![3; len]);
        }
        // Copies accept byte offsets and lengths without integer shader features.
        let x = Buffer::allocate_with(mem, 4, |i| i as u8).unwrap();
        let y = x.slice(1..).unwrap().to_owned(mem).unwrap();
        assert_eq!(&*y.map_read().unwrap(), &[1, 2, 3]);
    }
}

#[cfg(not(target_family = "wasm"))]
fn buffer_placement_errors(device: Device) {
    let host = Device::host();
    for mem in [
        Mem::Device(&host),
        Mem::Shared(&host),
        Mem::Upload(&host),
        Mem::Readback(&host),
    ] {
        assert!(Buffer::<u32>::zeros(mem, 0).is_err());
    }
    assert!(Buffer::<u32>::allocate_with(Mem::Device(&device), 0, |_| unreachable!()).is_err());
    assert!(Buffer::<u64>::allocate_with(Mem::Host, usize::MAX, |_| unreachable!()).is_err());
    for mem in mapped_placements(&device) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Buffer::allocate_with(mem, 2, |i| {
                assert_eq!(i, 0, "initializer panic after one write");
                i as u32
            })
            .unwrap()
        }));
        assert!(result.is_err());
        assert_eq!(
            &*Buffer::<u32>::zeros(mem, 2).unwrap().map_read().unwrap(),
            &[0, 0]
        );
        let mut original: ArcBuffer<_> = Buffer::<u32>::ones(mem, 2).unwrap().into();
        let alias = original.clone();
        original
            .make_slice_mut(Mem::Host)
            .unwrap()
            .map_write()
            .unwrap()
            .fill(0);
        assert!(original.device().is_host());
        assert_eq!(&*alias.map_read().unwrap(), &[1, 1]);
    }
}

#[cfg(not(target_family = "wasm"))]
fn buffer_concurrent_downloads(device: Device) {
    use std::{
        sync::{mpsc, Arc, Barrier},
        time::Duration,
    };
    // Exceed the 32 MB staging buffer so each download needs two chunks.
    const LEN: usize = 32_000_000 / std::mem::size_of::<u32>() + 1;
    let input: ArcBuffer<u32> = Buffer::ones(Mem::Device(&device), LEN).unwrap().into();
    let start = Arc::new(Barrier::new(2));
    let (sender, done) = mpsc::channel();
    for _ in 0..2 {
        let (input, start, sender) = (input.clone(), start.clone(), sender.clone());
        std::thread::spawn(move || {
            start.wait();
            let output = input.to_host_vec().unwrap();
            assert_eq!(output.len(), LEN);
            assert!(output.iter().all(|&x| x == 1));
            sender.send(()).unwrap();
        });
    }
    for _ in 0..2 {
        done.recv_timeout(Duration::from_secs(30))
            .expect("concurrent downloads must complete");
    }
}

#[cfg(not(target_family = "wasm"))]
fn buffer_mapping_conflicts(device: Device) {
    let mut x = Buffer::<u32>::ones(Mem::Shared(&device), 2).unwrap();
    let shared: ArcBuffer<_> = x.to_owned(Mem::Shared(&device)).unwrap().into();
    let alias = shared.clone();
    let view = shared.slice(1..).unwrap();
    let bytes = view.bitcast::<u8>().unwrap();
    let read = bytes.map_read().unwrap();
    let other_read = alias.map_read().unwrap();
    for error in [
        alias.to_owned(Mem::Device(&device)).err().unwrap(),
        alias.cast::<f32>(Mem::Shared(&device)).err().unwrap(),
    ] {
        assert_eq!(error.downcast_ref::<MapError>(), Some(&MapError::Conflict));
    }
    drop((read, other_read));
    x.copy_from_slice(&alias.as_slice()).unwrap();
    assert_eq!(&*x.map_read().unwrap(), &[1, 1]);
    // A forgotten guard must keep its access reservation.
    std::mem::forget(x.map_write().unwrap());
    assert_eq!(
        x.map_read().err().unwrap().downcast_ref::<MapError>(),
        Some(&MapError::Conflict)
    );
    assert_eq!(
        x.fill(0).err().unwrap().downcast_ref::<MapError>(),
        Some(&MapError::Conflict)
    );
}

#[cfg(target_family = "wasm")]
#[test]
fn buffer_placement_host() {
    buffer_placement(Device::host());
}

#[test]
fn buffer_from_vec_host() {
    buffer_from_vec(Device::host());
}

#[cfg(target_family = "wasm")]
macro_for!($T in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
    paste! {
        #[test]
        fn [<buffer_fill_ $T _host>]() {
            buffer_fill::<$T>(Device::host());
        }
    }
});

macro_for!($X in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
    macro_for!($Y in [u8, i8, u16, i16, f16, bf16, u32, i32, f32, u64, i64, f64] {
        paste! {
            #[test]
            fn [<buffer_cast_ $X _ $Y _host>]() {
                buffer_cast::<$X, $Y>(Device::host());
            }
            #[test]
            fn [<buffer_bitcast_ $X _ $Y _host>]() {
                buffer_bitcast::<$X, $Y>(Device::host());
            }
        }
    });
});
