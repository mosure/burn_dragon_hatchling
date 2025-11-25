#![recursion_limit = "256"]

use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use burn::optim::{AdamWConfig, GradientsParams, LearningRate, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_dragon_hatchling::{
    BDH, BDHConfig, BdhEsConfig, RecurrentStateStore, StreamHandle, bdh_param_specs,
    eggroll::{EggrollKey, EggrollNoiser, EsTreeKey},
    language_model_loss,
    wgpu::init_runtime,
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

fn stream_retain_bench(c: &mut Criterion) {
    run_stream_retain_backend::<Wgpu<f32>, _>(c, "wgpu", init_runtime);

    #[cfg(feature = "cuda")]
    run_stream_retain_backend::<Cuda<f32>, _>(c, "cuda", |_| {});
}

fn context_mode_bench(c: &mut Criterion) {
    run_context_modes_backend::<Wgpu<f32>, _>(c, "wgpu", init_runtime);

    #[cfg(feature = "cuda")]
    run_context_modes_backend::<Cuda<f32>, _>(c, "cuda", |_| {});
}

fn init_backend_once<B, Init>(backend_name: &str, init_backend: Init, device: &<B as BackendTrait>::Device)
where
    B: BackendTrait,
    Init: Fn(&<B as BackendTrait>::Device),
{
    static INIT_WGPU: Once = Once::new();
    static INIT_CUDA: Once = Once::new();

    match backend_name {
        "wgpu" => INIT_WGPU.call_once(|| init_backend(device)),
        "cuda" => INIT_CUDA.call_once(|| init_backend(device)),
        _ => init_backend(device),
    }
}

fn run_training_backend<B, Init>(c: &mut Criterion, backend_name: &'static str, init_backend: Init)
where
    B: AutodiffBackend + Clone + 'static,
    Init: Fn(&<B as BackendTrait>::Device),
{
    let device = <B as BackendTrait>::Device::default();
    <B as BackendTrait>::seed(&device, 42);
    init_backend_once::<B, _>(backend_name, init_backend, &device);

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
) where
    B: BackendTrait + Clone + 'static,
    Init: Fn(&<B as BackendTrait>::Device),
{
    let device = <B as BackendTrait>::Device::default();
    <B as BackendTrait>::seed(&device, 42);
    init_backend_once::<B, _>(backend_name, init_backend, &device);

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

    let mut group = c.benchmark_group(format!("bdh_forward_vs_eggroll/{backend_name}"));

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

        group.bench_with_input(BenchmarkId::new("plain", cfg.name), cfg, |b, _| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = Instant::now();
                    let _ = model.forward(inputs.clone());
                    total += start.elapsed();
                }
                total
            });
        });

        group.bench_with_input(BenchmarkId::new("eggroll", cfg.name), cfg, |b, _| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for step in 0..iters {
                    let es_step = es_key.clone().with_step(step as u64);
                    let start = Instant::now();
                    let _ = model.forward_with_noise(inputs.clone(), &noiser, &es_step, 0);
                    total += start.elapsed();
                }
                total
            });
        });
    }

    group.finish();
}

fn run_stream_retain_backend<B, Init>(
    c: &mut Criterion,
    backend_name: &'static str,
    init_backend: Init,
) where
    B: BackendTrait + Clone + 'static,
    Init: Fn(&<B as BackendTrait>::Device),
{
    let device = <B as BackendTrait>::Device::default();
    <B as BackendTrait>::seed(&device, 1337);
    init_backend_once::<B, _>(backend_name, init_backend, &device);

    // Keep the benchmark small enough for GPU memory while still exercising attention.
    let cfg = TrainConfig {
        name: "stream_b4_t64",
        batch: 4,
        block: 32,
    };

    let total_tokens = cfg.batch * cfg.block;
    let mut base_input_tokens = Vec::with_capacity(total_tokens);
    for idx in 0..total_tokens {
        base_input_tokens.push((idx % 255) as i64);
    }

    let inputs = Tensor::<B, 2, Int>::from_data(
        TensorData::new(base_input_tokens.clone(), [cfg.batch, cfg.block]),
        &device,
    );

    let model = BDH::<B>::new(BDHConfig::default(), &device);

    let mut group = c.benchmark_group(format!(
        "bdh_stream_retain_pct/{backend_name}/{cfg_name}",
        cfg_name = cfg.name
    ));
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.throughput(Throughput::Elements(cfg.batch as u64));

    let pct_values = [0.0_f32, 0.25];

    for &pct in &pct_values {
        let handles_template = build_stream_handles(&model, cfg.batch, cfg.block, pct);
        let max_ctx = Some(cfg.block); // keep stream states bounded per iter
        let reset_state = true; // ensure per-iteration isolation for fair timing

        // Warm-up once to avoid including shader compilation.
        let _ = forward_streaming(
            &model,
            inputs.clone(),
            handles_template.clone(),
            max_ctx,
            reset_state,
        );

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("pct_{pct:.2}")),
            &pct,
            |b, &_| {
                let handles_template = handles_template.clone();
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let start = Instant::now();
                        let _ = forward_streaming(
                            &model,
                            inputs.clone(),
                            handles_template.clone(),
                            max_ctx,
                            reset_state,
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

#[derive(Clone, Copy)]
struct ContextMode {
    name: &'static str,
    retain_pct: f32,
    max_context: Option<usize>,
    reset_each_iter: bool,
}

fn run_context_modes_backend<B, Init>(
    c: &mut Criterion,
    backend_name: &'static str,
    init_backend: Init,
) where
    B: BackendTrait + Clone + 'static,
    Init: Fn(&<B as BackendTrait>::Device),
{
    let force = std::env::var("BDH_RUN_STREAM_BENCH").unwrap_or_default() == "1";
    if backend_name == "cuda" && !force {
        eprintln!("Skipping cuda context_mode bench (cubecl fast_math divide-by-zero on this device). Set BDH_RUN_STREAM_BENCH=1 to force.");
        return;
    }

    let device = <B as BackendTrait>::Device::default();
    <B as BackendTrait>::seed(&device, 2024);
    init_backend_once::<B, _>(backend_name, init_backend, &device);

    // Mid-size batch/time to stress attention while keeping runtime acceptable.
    let cfg = TrainConfig {
        name: "ctx_b4_t128",
        batch: 4,
        block: 64,
    };

    let total_tokens = cfg.batch * cfg.block;
    let mut base_input_tokens = Vec::with_capacity(total_tokens);
    for idx in 0..total_tokens {
        base_input_tokens.push((idx % 255) as i64);
    }

    let inputs = Tensor::<B, 2, Int>::from_data(
        TensorData::new(base_input_tokens.clone(), [cfg.batch, cfg.block]),
        &device,
    );

    let model = BDH::<B>::new(BDHConfig::default(), &device);
    let modes = [
        ContextMode {
            name: "dense_no_stream",
            retain_pct: 0.0,
            max_context: None,
            reset_each_iter: false,
        },
        ContextMode {
            name: "stream_pct25_reset",
            retain_pct: 0.25,
            max_context: Some(cfg.block / 2),
            reset_each_iter: true,
        },
    ];

    let mut group = c.benchmark_group(format!(
        "bdh_context_modes/{backend_name}/{cfg_name}",
        cfg_name = cfg.name
    ));
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.throughput(Throughput::Elements(cfg.batch as u64));

    for mode in modes {
        let handles_template =
            build_stream_handles(&model, cfg.batch, cfg.block, mode.retain_pct);

        // Warm-up to exclude shader compilation.
        let _ = forward_streaming(
            &model,
            inputs.clone(),
            handles_template.clone(),
            mode.max_context,
            mode.reset_each_iter,
        );

        group.bench_with_input(BenchmarkId::from_parameter(mode.name), &mode, |b, _| {
            let handles_template = handles_template.clone();
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let start = Instant::now();
                    let _ = forward_streaming(
                        &model,
                        inputs.clone(),
                        handles_template.clone(),
                        mode.max_context,
                        mode.reset_each_iter,
                    );
                    total += start.elapsed();
                }
                total
            })
        });
    }

    group.finish();
}

fn build_stream_handles<B: BackendTrait>(
    _model: &BDH<B>,
    batch: usize,
    block: usize,
    retain_pct: f32,
) -> Vec<Option<StreamHandle<B>>> {
    let count = ((batch as f32) * retain_pct).round() as usize;
    let streams = count.min(batch);
    let mut handles = Vec::with_capacity(batch);
    let pool = Arc::new(Mutex::new(RecurrentStateStore::new(streams)));

    for idx in 0..batch {
        if idx < streams {
            handles.push(Some(StreamHandle {
                id: idx as u64,
                offset: idx * block,
                slot: idx,
                pool: Arc::clone(&pool),
            }));
        } else {
            handles.push(None);
        }
    }

    handles
}

fn forward_streaming<B: BackendTrait>(
    model: &BDH<B>,
    inputs: Tensor<B, 2, Int>,
    handles: Vec<Option<StreamHandle<B>>>,
    max_context: Option<usize>,
    reset_state: bool,
) -> Tensor<B, 3> {
    if handles.iter().all(|h| h.is_none()) {
        return model.forward(inputs);
    }

    let batch_size = inputs.shape().dims::<2>()[0];
    let time = inputs.shape().dims::<2>()[1];
    let device = inputs.device();

    let stream_indices: Vec<usize> = handles
        .iter()
        .enumerate()
        .filter_map(|(idx, entry)| entry.as_ref().map(|_| idx))
        .collect();
    let fresh_indices: Vec<usize> = handles
        .iter()
        .enumerate()
        .filter_map(|(idx, entry)| entry.is_none().then_some(idx))
        .collect();

    if stream_indices.is_empty() {
        return model.forward(inputs);
    }

    let pool_handle = handles
        .iter()
        .find_map(|handle| handle.as_ref().map(|handle| Arc::clone(&handle.pool)))
        .expect("stream pool available");

    let contiguous = stream_indices
        .iter()
        .copied()
        .eq(0..stream_indices.len());

    let mut pool_guard = pool_handle.lock().expect("lock stream pool");
    let pool = pool_guard.ensure_pool(|max_streams| model.init_state_pool(max_streams, &device));
    if reset_state {
        for slot in 0..stream_indices.len() {
            pool.reset_slot(slot);
        }
    }
    let state_view = pool.prefix_view(stream_indices.len());
    drop(pool_guard);

    let stream_inputs = if contiguous {
        inputs.clone().slice_dim(0, 0..stream_indices.len())
    } else {
        let mut slices = Vec::with_capacity(stream_indices.len());
        for &idx in &stream_indices {
            slices.push(inputs.clone().slice_dim(0, idx..idx + 1));
        }
        Tensor::cat(slices, 0)
    };

    let (logits_stream, updated_states) =
        model.forward_with_state_pool(stream_inputs, state_view, false);

    let mut logits_ordered: Vec<Option<Tensor<B, 3>>> = vec![None; batch_size];
    for (pos, &idx) in stream_indices.iter().enumerate() {
        logits_ordered[idx] = Some(logits_stream.clone().slice_dim(0, pos..pos + 1));
    }

    let mut pool_guard = pool_handle.lock().expect("lock stream pool");
    let pool = pool_guard.ensure_pool(|max_streams| model.init_state_pool(max_streams, &device));
    pool.write_prefix(stream_indices.len(), &updated_states, max_context, time);
    drop(pool_guard);

    if !fresh_indices.is_empty() {
        let fresh_inputs = if contiguous && stream_indices.len() < batch_size {
            inputs
                .clone()
                .slice_dim(0, stream_indices.len()..batch_size)
        } else if fresh_indices.len() == 1 {
            inputs.clone().slice_dim(0, fresh_indices[0]..fresh_indices[0] + 1)
        } else {
            let mut slices = Vec::with_capacity(fresh_indices.len());
            for &idx in &fresh_indices {
                slices.push(inputs.clone().slice_dim(0, idx..idx + 1));
            }
            Tensor::cat(slices, 0)
        };

        let logits_fresh = model.forward(fresh_inputs);
        for (pos, idx) in fresh_indices.into_iter().enumerate() {
            logits_ordered[idx] = Some(logits_fresh.clone().slice_dim(0, pos..pos + 1));
        }
    }

    let ordered: Vec<_> = logits_ordered
        .into_iter()
        .map(|item| item.expect("logits present"))
        .collect();
    Tensor::cat(ordered, 0)
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

criterion_group!(
    benches,
    training_step_bench,
    eggroll_forward_bench,
    stream_retain_bench,
    context_mode_bench
);
criterion_main!(benches);
