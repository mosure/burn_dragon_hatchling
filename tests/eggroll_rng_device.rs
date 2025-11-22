use std::panic;

use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use burn_wgpu::Wgpu;

use burn_dragon_hatchling::eggroll::{
    EggrollConfig, EggrollKey, EggrollNoiseTensor, EggrollNoiser, EggrollParamSpec, EsTreeKey,
};
use burn_dragon_hatchling::wgpu::init_runtime;

fn noise_vec<B: Backend>(device: &B::Device) -> Vec<f32> {
    let spec = EggrollParamSpec {
        id: burn::module::ParamId::from(1u64),
        path: "test".into(),
        shape: (3, 4),
        rank: 2,
        sigma_scale: 1.0,
        stack: None,
    };
    let noiser: EggrollNoiser<B> =
        EggrollNoiser::new(vec![spec.clone()], EggrollConfig::default(), device);
    let es_key = EsTreeKey::new(EggrollKey::from_seed(123)).with_step(7);
    let noise = match noiser.noise_block(&spec, &es_key, 5) {
        EggrollNoiseTensor::D2(noise) => noise,
        EggrollNoiseTensor::D3(_) => panic!("expected 2D noise block"),
    };
    noise
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap()
}

#[test]
fn rng_matches_between_cpu_and_wgpu() {
    let cpu_device = <NdArray<f32> as Backend>::Device::default();
    let gpu_device = <Wgpu<f32> as Backend>::Device::default();

    // Skip gracefully if wgpu init fails in the environment.
    if let Err(_) = panic::catch_unwind(|| init_runtime(&gpu_device)) {
        eprintln!("Skipping device-independence test: wgpu runtime unavailable");
        return;
    }

    let cpu = noise_vec::<NdArray<f32>>(&cpu_device);
    let gpu = match panic::catch_unwind(|| noise_vec::<Wgpu<f32>>(&gpu_device)) {
        Ok(v) => v,
        Err(_) => {
            eprintln!("Skipping device-independence test: wgpu backend unavailable");
            return;
        }
    };

    assert_eq!(cpu.len(), gpu.len());
    for (a, b) in cpu.iter().zip(gpu.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "rng mismatch between cpu and wgpu: {a} vs {b}"
        );
    }
}
