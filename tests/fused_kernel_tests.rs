use std::f32::consts::PI;

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution as TensorDistribution, Int, Tensor, activation};
use burn_dragon_hatchling::kernel::{linear_attention, relu_lowrank};
use burn_dragon_hatchling::{BlockPattern1d, BlockPattern2d, RotaryEmbedding};
use burn_ndarray::NdArray;

type Backend = NdArray<f32>;

const ROW_NORM_EPS: f32 = 1e-6;

fn assert_close(lhs: Tensor<Backend, 4>, rhs: Tensor<Backend, 4>, atol: f32, rtol: f32) {
    let lhs_data = lhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("lhs vec");
    let rhs_data = rhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rhs vec");

    for (a, b) in lhs_data.iter().zip(rhs_data.iter()) {
        let diff = (a - b).abs();
        let tol = atol + rtol * b.abs();
        assert!(
            diff <= tol,
            "difference {diff} exceeds tolerance {tol} (lhs={a}, rhs={b})"
        );
    }
}

#[test]
fn relu_lowrank_fused_matches_reference() {
    let device = <Backend as BackendTrait>::Device::default();
    <Backend as BackendTrait>::seed(&device, 42);

    let input =
        Tensor::<Backend, 4>::random([2, 1, 5, 6], TensorDistribution::Normal(0.0, 1.0), &device);
    let weight =
        Tensor::<Backend, 4>::random([1, 3, 6, 8], TensorDistribution::Normal(0.0, 1.0), &device);
    let bias =
        Tensor::<Backend, 3>::random([3, 1, 8], TensorDistribution::Normal(0.0, 0.5), &device);
    let threshold = 0.25;

    let layout = BlockPattern1d::from_blocks(3, [0usize, 2usize]);
    let fused = relu_lowrank::fused_forward(
        input.clone(),
        weight.clone(),
        Some(bias.clone()),
        threshold,
        &layout,
    );

    let mut reference = input.matmul(weight);
    let bias_dims = bias.shape().dims::<3>();
    let bias_view = bias.reshape([1, bias_dims[0], 1, bias_dims[2]]);
    reference = reference + bias_view;
    reference = reference.sub_scalar(threshold);
    reference = activation::relu(reference);
    let latent_dim = reference.shape().dims::<4>()[3];
    let mask = layout.mask::<Backend>(latent_dim, &device);
    reference = reference * mask;

    assert_close(fused, reference, 1e-5, 1e-5);
}

#[test]
fn fused_attention_matches_reference_when_alibi_disabled() {
    let device = <Backend as BackendTrait>::Device::default();
    <Backend as BackendTrait>::seed(&device, 1337);

    let batch = 1;
    let heads = 2;
    let time = 6;
    let latent = 8;
    let value_dim = 6;

    let query = Tensor::<Backend, 4>::random(
        [batch, heads, time, latent],
        TensorDistribution::Normal(0.0, 1.0),
        &device,
    );
    let value = Tensor::<Backend, 4>::random(
        [batch, 1, time, value_dim],
        TensorDistribution::Normal(0.0, 1.0),
        &device,
    );

    let freqs = build_freqs(latent, 65_536.0, RotaryEmbedding::Rope, &device);

    // Reference attention (no ALiBi, dense computation).
    let positions = Tensor::<Backend, 1, Int>::arange(0..time as i64, &device)
        .float()
        .reshape([1, 1, time, 1]);
    let raw = positions.clone() * freqs.clone();
    let phases = (raw.clone() - raw.floor()) * (2.0 * PI);
    let q_rot = rope(phases.clone(), query.clone());
    let k_rot = q_rot.clone();
    let scores = q_rot.matmul(k_rot.swap_dims(2, 3)).tril(-1);
    let denom = scores.clone().abs().sum_dim(3).add_scalar(ROW_NORM_EPS);
    let scores = scores / denom;
    let repeated_value = value.clone().repeat_dim(1, heads);
    let reference = scores.matmul(repeated_value);

    // Fused attention with block-sparse layout covering all causal blocks.
    let layout = BlockPattern2d::dense(3);
    let fused = linear_attention::fused_state_aligned(
        query.clone(),
        value.clone(),
        freqs,
        None,
        &layout,
        RotaryEmbedding::Rope,
    );

    assert_close(fused, reference, 1e-4, 1e-4);
}

#[test]
fn fused_attention_matches_reference_with_pope() {
    let device = <Backend as BackendTrait>::Device::default();
    <Backend as BackendTrait>::seed(&device, 2024);

    let batch = 1;
    let heads = 2;
    let time = 6;
    let latent = 8;
    let value_dim = 6;

    let query = Tensor::<Backend, 4>::random(
        [batch, heads, time, latent],
        TensorDistribution::Normal(0.0, 1.0),
        &device,
    );
    let value = Tensor::<Backend, 4>::random(
        [batch, 1, time, value_dim],
        TensorDistribution::Normal(0.0, 1.0),
        &device,
    );

    let freqs = build_freqs(latent, 65_536.0, RotaryEmbedding::Pope, &device);

    let positions = Tensor::<Backend, 1, Int>::arange(0..time as i64, &device)
        .float()
        .reshape([1, 1, time, 1]);
    let raw = positions.clone() * freqs.clone();
    let phases = (raw.clone() - raw.floor()) * (2.0 * PI);
    let q_rot = pope(phases.clone(), query.clone());
    let k_rot = q_rot.clone();
    let scores = q_rot.matmul(k_rot.swap_dims(2, 3)).tril(-1);
    let denom = scores.clone().abs().sum_dim(3).add_scalar(ROW_NORM_EPS);
    let scores = scores / denom;
    let repeated_value = value.clone().repeat_dim(1, heads);
    let reference = scores.matmul(repeated_value);

    let layout = BlockPattern2d::dense(3);
    let fused = linear_attention::fused_state_aligned(
        query.clone(),
        value.clone(),
        freqs,
        None,
        &layout,
        RotaryEmbedding::Pope,
    );

    assert_close(fused, reference, 1e-4, 1e-4);
}

fn build_freqs(
    latent: usize,
    theta: f32,
    rotary_embedding: RotaryEmbedding,
    device: &<Backend as BackendTrait>::Device,
) -> Tensor<Backend, 4> {
    let mut data = Vec::with_capacity(latent);
    for idx in 0..latent {
        let exponent = match rotary_embedding {
            RotaryEmbedding::Rope => ((idx as f32 / 2.0).floor() * 2.0) / latent as f32,
            RotaryEmbedding::Pope => idx as f32 / latent as f32,
            RotaryEmbedding::Alibi => 0.0,
        };
        let value = if matches!(rotary_embedding, RotaryEmbedding::Alibi) {
            0.0
        } else {
            1.0 / theta.powf(exponent) / (2.0 * PI)
        };
        data.push(value);
    }
    Tensor::<Backend, 1>::from_floats(data.as_slice(), device).reshape([1, 1, 1, latent])
}

fn rope(phases: Tensor<Backend, 4>, values: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
    let cos = phases.clone().cos();
    let sin = phases.sin();

    let [b, h, t, n] = values.shape().dims();
    let pairs = values.clone().reshape([b, h, t, n / 2, 2]);

    let even = pairs
        .clone()
        .slice_dim(4, 0..1)
        .squeeze_dim::<4>(4);
    let odd = pairs.slice_dim(4, 1..2).squeeze_dim::<4>(4);

    let rotated = Tensor::stack::<5>(vec![odd.clone().neg(), even], 4).reshape([b, h, t, n]);

    values * cos + rotated * sin
}

fn pope(phases: Tensor<Backend, 4>, values: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
    let magnitude = activation::softplus(values, 1.0);
    let cos = phases.clone().cos();
    let sin = phases.sin();
    let real = magnitude.clone() * cos;
    let imag = magnitude * sin;
    Tensor::cat(vec![real, imag], 3)
}
