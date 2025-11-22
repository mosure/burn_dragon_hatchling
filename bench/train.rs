#![recursion_limit = "256"]

use std::time::{Duration, Instant};

use burn::optim::{AdamWConfig, GradientsParams, LearningRate, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_dragon_hatchling::{
    eggroll::{EggrollKey, EggrollNoiser, EsTreeKey},
    bdh_param_specs, language_model_loss,
    BDH, BDHConfig, BdhEsConfig, wgpu::init_runtime,
};
use burn_wgpu::Wgpu;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

#[cfg(feature = "cuda")]
use burn_cuda::Cuda;

#[derive(Clone, Copy)]
struct TrainConfig {
    name: &'static str,
    batch: usize,
    block: usize,
}

const TRAIN_CONFIGS: &[TrainConfig] = &[
    TrainConfig {
        name: "b4_t64",
        batch: 4,
        block: 64,
    },
    TrainConfig {
        name: "b8_t128",
        batch: 8,
        block: 128,
    },
    TrainConfig {
        name: "b16_t256",
        batch: 16,
        block: 256,
    },
    TrainConfig {
        name: "b16_t512",
        batch: 8,
        block: 512,
    },
];

fn training_step_bench(c: &mut Criterion) {
    run_training_backend::<Autodiff<Wgpu<f32>>, _>(c, "wgpu", init_runtime);

    #[cfg(feature = "cuda")]
    run_training_backend::<Autodiff<Cuda<f32>>, _>(c, "cuda", |_| {});
}

fn eggroll_forward_bench(c: &mut Criterion) {
    run_eggroll_forward_backend::<Wgpu<f32>, _>(c, "wgpu", init_runtime);

    #[cfg(feature = "cuda")]
    run_eggroll_forward_backend::<Cuda<f32>, _>(c, "cuda", |_| {});
}

fn run_training_backend<B, Init>(c: &mut Criterion, backend_name: &'static str, init_backend: Init)
where
    B: AutodiffBackend + Clone + 'static,
    Init: Fn(&<B as BackendTrait>::Device),
{
    let device = <B as BackendTrait>::Device::default();
    <B as BackendTrait>::seed(&device, 42);
    init_backend(&device);

    let model_config = BDHConfig::default();
    let base_model = BDH::<B>::new(model_config.clone(), &device);
    let optimizer_config = AdamWConfig::new().with_weight_decay(0.1);
    let lr: LearningRate = 1e-3;

    let max_batch = TRAIN_CONFIGS.iter().map(|cfg| cfg.batch).max().unwrap_or(1);
    let max_block = TRAIN_CONFIGS.iter().map(|cfg| cfg.block).max().unwrap_or(1);
    let max_token_count = max_batch * max_block;

    let mut base_input_tokens = Vec::with_capacity(max_token_count);
    for idx in 0..max_token_count {
        base_input_tokens.push((idx % 255) as i64);
    }
    let base_target_tokens: Vec<i64> = base_input_tokens
        .iter()
        .map(|tok| (*tok + 1) % 255)
        .collect();

    let base_inputs = Tensor::<B, 2, Int>::from_data(
        TensorData::new(base_input_tokens.clone(), [max_batch, max_block]),
        &device,
    );
    let base_targets = Tensor::<B, 2, Int>::from_data(
        TensorData::new(base_target_tokens, [max_batch, max_block]),
        &device,
    );

    let mut group = c.benchmark_group(format!("bdh_single_train_step/{backend_name}"));

    for cfg in TRAIN_CONFIGS {
        let inputs = base_inputs
            .clone()
            .slice_dim(0, 0..cfg.batch)
            .slice_dim(1, 0..cfg.block);
        let targets = base_targets
            .clone()
            .slice_dim(0, 0..cfg.batch)
            .slice_dim(1, 0..cfg.block);

        // Warm-up pass to avoid counting shader compilation and graph building.
        {
            let model = base_model.clone();
            let mut optimizer = optimizer_config.clone().init::<B, BDH<B>>();
            let logits = model.forward(inputs.clone());
            let loss = language_model_loss::<B>(logits, targets.clone());
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            let _ = optimizer.step(lr, model, grads);
        }

        log_theoretical_profile(&model_config, cfg);

        group.throughput(Throughput::Elements(cfg.batch as u64));
        group.bench_with_input(BenchmarkId::from_parameter(cfg.name), cfg, |b, _| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;

                for _ in 0..iters {
                    let model = base_model.clone();
                    let mut optimizer = optimizer_config.clone().init::<B, BDH<B>>();

                    let start = Instant::now();
                    let logits = model.forward(inputs.clone());
                    let loss = language_model_loss::<B>(logits, targets.clone());
                    let grads = loss.backward();
                    let grads = GradientsParams::from_grads(grads, &model);
                    let _ = optimizer.step(lr, model, grads);
                    total += start.elapsed();
                }

                total
            });
        });
    }

    group.finish();
}

fn run_eggroll_forward_backend<B, Init>(
    c: &mut Criterion,
    backend_name: &'static str,
    init_backend: Init,
)
where
    B: BackendTrait + Clone + 'static,
    Init: Fn(&<B as BackendTrait>::Device),
{
    let device = <B as BackendTrait>::Device::default();
    <B as BackendTrait>::seed(&device, 42);
    init_backend(&device);

    let model_config = BDHConfig::default();
    let mut es_cfg = BdhEsConfig::default();
    es_cfg.eggroll.pop_size = 1; // forward_with_noise uses a single noise sample
    es_cfg.eggroll.sigma = 0.01;

    let max_batch = TRAIN_CONFIGS.iter().map(|cfg| cfg.batch).max().unwrap_or(1);
    let max_block = TRAIN_CONFIGS.iter().map(|cfg| cfg.block).max().unwrap_or(1);
    let max_token_count = max_batch * max_block;

    let mut base_input_tokens = Vec::with_capacity(max_token_count);
    for idx in 0..max_token_count {
        base_input_tokens.push((idx % 255) as i64);
    }
    let base_inputs = Tensor::<B, 2, Int>::from_data(
        TensorData::new(base_input_tokens, [max_batch, max_block]),
        &device,
    );

    let mut group =
        c.benchmark_group(format!("bdh_forward_vs_eggroll/{backend_name}"));

    for cfg in TRAIN_CONFIGS {
        let inputs = base_inputs
            .clone()
            .slice_dim(0, 0..cfg.batch)
            .slice_dim(1, 0..cfg.block);

        let model = BDH::<B>::new(model_config.clone(), &device);
        let param_specs = bdh_param_specs(&model, &es_cfg);
        let noiser = EggrollNoiser::new(param_specs, es_cfg.eggroll.clone(), &device);
        let es_key = EsTreeKey::new(EggrollKey::from_seed(es_cfg.eggroll.seed));

        // Warm-up plain and noisy forward
        let _ = model.forward(inputs.clone());
        let _ = model.forward_with_noise(inputs.clone(), &noiser, &es_key, 0);

        log_theoretical_profile(&model_config, cfg);
        group.throughput(Throughput::Elements(cfg.batch as u64));

        group.bench_with_input(
            BenchmarkId::new("plain", cfg.name),
            cfg,
            |b, _| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let start = Instant::now();
                        let _ = model.forward(inputs.clone());
                        total += start.elapsed();
                    }
                    total
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("eggroll", cfg.name),
            cfg,
            |b, _| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for step in 0..iters {
                        let es_step = es_key.clone().with_step(step as u64);
                        let start = Instant::now();
                        let _ = model.forward_with_noise(
                            inputs.clone(),
                            &noiser,
                            &es_step,
                            0,
                        );
                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

fn log_theoretical_profile(config: &BDHConfig, cfg: &TrainConfig) {
    let batch = cfg.batch as u64;
    let time = cfg.block as u64;
    let embed = config.n_embd as u64;
    let latent_per_head = compute_latent_per_head(config) as u64;
    let latent_total = compute_latent_total(config) as u64;
    let heads = config.n_head as u64;
    let bt = batch * time;

    let encoder_matmul = 2 * bt * embed * latent_total;
    let attn_scores = 2 * batch * heads * time * time * latent_per_head;
    let attn_value = 2 * batch * heads * time * time * embed;
    let decoder_matmul = 2 * bt * latent_total * embed;
    let total = encoder_matmul + attn_scores + attn_value + decoder_matmul;

    println!(
        "[train:{name}] approx forward GFLOPs: total={total_gflops:.2}, encoder={enc:.2}, \
         attn_scores={scores:.2}, attn_value={value:.2}, decoder={dec:.2} (backward ~2x forward)",
        name = cfg.name,
        total_gflops = total as f64 / 1e9,
        enc = encoder_matmul as f64 / 1e9,
        scores = attn_scores as f64 / 1e9,
        value = attn_value as f64 / 1e9,
        dec = decoder_matmul as f64 / 1e9,
    );
}

fn compute_latent_per_head(config: &BDHConfig) -> usize {
    (config.mlp_internal_dim_multiplier * config.n_embd) / config.n_head
}

fn compute_latent_total(config: &BDHConfig) -> usize {
    compute_latent_per_head(config) * config.n_head
}

criterion_group!(benches, training_step_bench, eggroll_forward_bench);
criterion_main!(benches);
