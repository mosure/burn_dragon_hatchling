use burn::module::ParamId;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EggrollKey {
    pub words: [u32; 2],
}

impl EggrollKey {
    pub fn from_seed(seed: u64) -> Self {
        Self {
            words: [
                (seed & 0xFFFF_FFFF) as u32,
                (seed >> 32) as u32 ^ 0x9E37_79B9,
            ],
        }
    }
}

pub fn fold_in(key: EggrollKey, data: u64) -> EggrollKey {
    let mixed = mix64(data);
    EggrollKey {
        words: [
            (mix64(key.words[0] as u64 ^ mixed) & 0xFFFF_FFFF) as u32,
            (mix64(key.words[1] as u64).wrapping_add(mixed) & 0xFFFF_FFFF) as u32,
        ],
    }
}

#[derive(Clone, Debug)]
pub struct EsTreeKey {
    pub base_key: EggrollKey,
    pub step: u64,
}

impl EsTreeKey {
    pub fn new(base_key: EggrollKey) -> Self {
        Self { base_key, step: 0 }
    }

    pub fn with_step(mut self, step: u64) -> Self {
        self.step = step;
        self
    }

    pub fn for_param(&self, param_id: ParamId) -> EggrollKey {
        let step_key = fold_in(self.base_key, self.step);
        fold_in(step_key, param_id.val())
    }

    pub fn for_param_thread(&self, param_id: ParamId, thread_id: u32) -> EggrollKey {
        let param_key = self.for_param(param_id);
        fold_in(param_key, thread_id as u64)
    }

    pub fn for_axis(&self, param_id: ParamId, thread_id: u32, axis_tag: u64) -> EggrollKey {
        let thread_key = self.for_param_thread(param_id, thread_id);
        fold_in(thread_key, axis_tag)
    }
}

/// Generate normally distributed f32 samples using a counter-based stream.
///
/// This intentionally mirrors the counter/key split used by JAX's PRNG design
/// (though the underlying mixing is splitmix64 here rather than Random123).
pub fn normal_f32<B: Backend>(
    key: EggrollKey,
    count: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    normal_f32_from_offset::<B>(key, count, 0, device)
}

pub fn normal_f32_from_offset<B: Backend>(
    key: EggrollKey,
    count: usize,
    counter_offset: u64,
    device: &B::Device,
) -> Tensor<B, 1> {
    let mut values = Vec::with_capacity(count);
    let mut counter = counter_offset;

    while values.len() < count {
        let u1 = unit_from_key(key, counter);
        counter = counter.wrapping_add(1);
        let u2 = unit_from_key(key, counter);
        counter = counter.wrapping_add(1);

        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = core::f64::consts::TAU * u2;
        values.push((radius * theta.cos()) as f32);
        if values.len() < count {
            values.push((radius * theta.sin()) as f32);
        }
    }

    let data = TensorData::new(values, [count]);
    Tensor::<B, 1>::from_data(data, device)
}

fn unit_from_key(key: EggrollKey, counter: u64) -> f64 {
    let word = philox2x32(key, counter).0 as u64;
    let scaled =
        ((word >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)); // 53 bits -> (0,1)
    scaled.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
}

// Philox2x32 adapted from Random123-style round constants.
fn philox2x32(key: EggrollKey, counter: u64) -> (u32, u32) {
    // Actually implement Threefry2x32 (Random123) for better diffusion.
    let mut x0 = counter as u32;
    let mut x1 = (counter >> 32) as u32;
    const C: u32 = 0x1BD11BDA;
    let ks = [key.words[0], key.words[1], key.words[0] ^ key.words[1] ^ C];

    let rotations = [13u32, 15, 26, 6, 17, 29, 16, 24];

    x0 = x0.wrapping_add(ks[0]);
    x1 = x1.wrapping_add(ks[1]);

    for r in 0..20 {
        let rot = rotations[r % rotations.len()];
        x0 = x0.wrapping_add(x1);
        x1 = x1.rotate_left(rot) ^ x0;

        if (r + 1) % 4 == 0 {
            let k = (r + 1) / 4;
            x0 = x0.wrapping_add(ks[k % 3]);
            x1 = x1.wrapping_add(ks[(k + 1) % 3].wrapping_add(k as u32));
        }
    }

    (x0, x1)
}

// Splitmix64 mixer: cheap and reproducible across devices.
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn same_index_same_value() {
        let key = EggrollKey::from_seed(42);
        let device = <NdArray<f32> as Backend>::Device::default();
        let out1 = normal_f32::<B>(key, 8, &device);
        let out2 = normal_f32::<B>(key, 8, &device);
        assert_eq!(out1.to_data().convert::<f32>(), out2.to_data().convert::<f32>());
    }

    #[test]
    fn different_index_different_value() {
        let key_a = EggrollKey::from_seed(42);
        let key_b = EggrollKey::from_seed(1337);
        let device = <NdArray<f32> as Backend>::Device::default();
        let out_a = normal_f32::<B>(key_a, 8, &device);
        let out_b = normal_f32::<B>(key_b, 8, &device);
        assert_ne!(out_a.to_data().convert::<f32>(), out_b.to_data().convert::<f32>());
    }

    #[test]
    fn order_independence() {
        let key = EggrollKey::from_seed(123);
        let device = <NdArray<f32> as Backend>::Device::default();
        let a_first = normal_f32::<B>(key, 4, &device);
        let b_second = normal_f32::<B>(fold_in(key, 1), 4, &device);

        let b_first = normal_f32::<B>(fold_in(key, 1), 4, &device);
        let a_second = normal_f32::<B>(key, 4, &device);

        assert_eq!(a_first.to_data().convert::<f32>(), a_second.to_data().convert::<f32>());
        assert_eq!(b_first.to_data().convert::<f32>(), b_second.to_data().convert::<f32>());
    }

    #[test]
    fn step_changes_stream() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let key = EggrollKey::from_seed(7);
        let out_step0 = normal_f32_from_offset::<B>(key, 4, 0, &device);
        let out_step1 = normal_f32_from_offset::<B>(key, 4, 1024, &device);
        assert_ne!(out_step0.to_data().convert::<f32>(), out_step1.to_data().convert::<f32>());
    }

    #[test]
    fn cross_loop_invariance() {
        let device = <NdArray<f32> as Backend>::Device::default();
        let key = EggrollKey::from_seed(2025);
        let params = [1u64, 2, 3];
        let threads = [0u32, 5, 9];

        let mut map_a = std::collections::HashMap::new();
        for &p in &params {
            for &t in &threads {
                let k = EsTreeKey::new(key).with_step(0).for_param_thread(ParamId::from(p), t);
                let noise = normal_f32::<B>(k, 4, &device).to_data().convert::<f32>();
                map_a.insert((p, t), noise);
            }
        }

        let mut map_b = std::collections::HashMap::new();
        for &t in threads.iter().rev() {
            for &p in params.iter().rev() {
                let k = EsTreeKey::new(key).with_step(0).for_param_thread(ParamId::from(p), t);
                let noise = normal_f32::<B>(k, 4, &device).to_data().convert::<f32>();
                map_b.insert((p, t), noise);
            }
        }

        assert_eq!(map_a, map_b, "noise differs with reordered loops");
    }
}
