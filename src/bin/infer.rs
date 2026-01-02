#![recursion_limit = "256"]

use std::convert::TryFrom;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};
use serde::Deserialize;

use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend;
use burn_dragon_hatchling::wgpu::init_runtime;
use burn_dragon_hatchling::{
    BDH, ContextStrategy, ContextStrategyConfig, GenerationConfig, ModelOverrides, TrainingConfig,
    build_model_config, generate_text, load_training_config, prefill_state,
    resolve_context_strategy, sample_next_token,
};
use burn_wgpu::Wgpu;

#[cfg(feature = "cuda")]
use burn_cuda::Cuda;

#[cfg(feature = "viz")]
use burn_dragon_hatchling::viz::{self, VizConfig, VizDimensions, VizEncoder};
#[cfg(feature = "viz")]
use std::sync::mpsc;
#[cfg(feature = "viz")]
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[cfg(feature = "viz")]
struct VizRuntime<B: Backend> {
    encoder: VizEncoder<B>,
    sender: viz::VizSender<B>,
    stop: Arc<AtomicBool>,
}

pub fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    let mut config_paths = vec![PathBuf::from("config/base.toml")];
    config_paths.extend(args.config.clone());
    let mut config = load_training_config(&config_paths)?;
    if args.config.is_empty() {
        let backend_name = backend_name(args.backend);
        if let Some(config_path) = resolve_run_config_path(args.checkpoint.as_ref(), backend_name) {
            let contents = fs::read_to_string(&config_path).with_context(|| {
                format!("failed to read run config {}", config_path.display())
            })?;
            let run_config: RunConfigJson = serde_json::from_str(&contents)
                .with_context(|| format!("failed to parse {}", config_path.display()))?;
            apply_run_config(&mut config, &run_config);
        }
    }

    #[cfg(feature = "viz")]
    let use_viz = args.viz;
    #[cfg(not(feature = "viz"))]
    let use_viz = false;

    match args.backend {
        BackendArg::Wgpu => {
            #[cfg(feature = "viz")]
            if use_viz {
                return infer_backend_with_viz::<Wgpu<f32>, _>(
                    &config,
                    &args,
                    "wgpu",
                    init_runtime,
                );
            }
            infer_backend::<Wgpu<f32>, _>(&config, &args, "wgpu", init_runtime)
        }
        BackendArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                if use_viz {
                    return Err(anyhow!(
                        "viz overlay requires the wgpu backend; run with --backend wgpu"
                    ));
                }
                infer_backend::<Cuda<f32>, _>(&config, &args, "cuda", |_| {})
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(anyhow!(
                    "cuda backend selected but this build lacks `cuda` feature; rebuild with `--features cuda`"
                ))
            }
        }
    }
}

fn infer_backend<B, Init>(
    config: &TrainingConfig,
    args: &Args,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: Backend + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    let device = B::Device::default();
    #[cfg(feature = "viz")]
    {
        infer_backend_on_device::<B, Init>(
            config,
            args,
            backend_name,
            device,
            init_backend,
            None,
        )
    }
    #[cfg(not(feature = "viz"))]
    {
        infer_backend_on_device::<B, Init>(config, args, backend_name, device, init_backend)
    }
}

fn infer_backend_on_device<B, Init>(
    config: &TrainingConfig,
    args: &Args,
    backend_name: &str,
    device: B::Device,
    init_backend: Init,
    #[cfg(feature = "viz")] mut viz_runtime: Option<VizRuntime<B>>,
) -> Result<()>
where
    B: Backend + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    B::seed(&device, 1337);
    init_backend(&device);

    let tokenizer_path = config
        .dataset
        .tokenizer
        .storage_path(&config.dataset.cache_dir);
    let tokenizer = if let Some(path) = tokenizer_path {
        config
            .dataset
            .tokenizer
            .load(&path)
            .with_context(|| format!("failed to load tokenizer {}", path.display()))?
    } else {
        config
            .dataset
            .tokenizer
            .fit(std::iter::empty::<&str>())
            .context("failed to initialize tokenizer")?
    };

    let checkpoint_dir = args
        .checkpoint
        .clone()
        .unwrap_or_else(|| default_checkpoint_dir(backend_name));
    let (checkpoint_base, epoch) = resolve_checkpoint_base(&checkpoint_dir, args.epoch)?;

    let mut model_config = build_model_config(&config.model, config.training.block_size);
    model_config.vocab_size = tokenizer.len();
    let mut model = BDH::<B>::new(model_config, &device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load::<<BDH<B> as Module<B>>::Record>(checkpoint_base.clone(), &device)
        .with_context(|| {
            format!(
                "failed to load checkpoint {}",
                format_checkpoint(&checkpoint_base)
            )
        })?;
    model = model.load_record(record);

    let mut generation = config.generation.clone();
    apply_generation_overrides(&mut generation, args, config.training.block_size);

    let status_msg = format!(
        "Loaded epoch {epoch} from {} using {backend_name} backend.",
        format_checkpoint(&checkpoint_base)
    );

    let use_streaming = args.streaming;

    if use_streaming {
        let strategy =
            resolve_context_strategy(&generation.context_strategy, config.training.block_size);
        eprintln!("{status_msg}");

        let mut prompt_ids = tokenizer.encode(&generation.prompt, false, false);
        if let ContextStrategy::Sliding { window } = strategy
            && prompt_ids.len() > window
        {
            prompt_ids = prompt_ids[prompt_ids.len() - window..].to_vec();
        }

        let prompt_tokens: Vec<i64> = prompt_ids.iter().map(|&id| id as i64).collect();
        let prompt_ids_u32: Vec<u32> = prompt_ids.to_vec();

        let mut writer = io::stdout();
        let prompt_text = tokenizer.decode(&prompt_ids_u32);
        writer
            .write_all(prompt_text.as_bytes())
            .context("failed to write prompt to stdout")?;
        writer.flush().context("failed to flush stdout")?;

        let mut generated_ids: Vec<u32> = Vec::new();
        let mut last_print_len = 0usize;
        let mut stream_err: Option<anyhow::Error> = None;

        let (mut state, mut last_logits) = prefill_state::<B>(&model, &prompt_tokens, &device)?;

        if let ContextStrategy::Sliding { window } = strategy
            && window > 0
            && state.position > window
        {
            state.trim(window);
        }

        let max_tokens = normalize_max_tokens(generation.max_tokens);
        let mut generated = 0usize;
        while max_tokens.is_none_or(|max| generated < max) {
            #[cfg(feature = "viz")]
            if let Some(viz) = viz_runtime.as_ref() {
                if viz.stop.load(Ordering::Relaxed) {
                    break;
                }
            }

            let (next, logits) = sample_next_token(
                &model,
                &mut state,
                last_logits,
                generation.temperature,
                generation.top_k,
                &device,
            )?;
            last_logits = logits;
            generated = generated.saturating_add(1);

            if let Ok(token_u32) = u32::try_from(next) {
                generated_ids.push(token_u32);
                let decoded = tokenizer.decode(&generated_ids);

                if decoded.len() > last_print_len {
                    let new_text = &decoded[last_print_len..];
                    if !new_text.is_empty() {
                        if let Err(err) = writer.write_all(new_text.as_bytes()) {
                            stream_err = Some(anyhow!("failed to write streamed token: {err}"));
                            break;
                        }
                        if let Err(err) = writer.flush() {
                            stream_err =
                                Some(anyhow!("failed to flush stdout during streaming: {err}"));
                            break;
                        }
                    }
                    last_print_len = decoded.len();
                }
            }

            #[cfg(feature = "viz")]
            if let Some(viz) = viz_runtime.as_mut() {
                let token_index = state.position.saturating_sub(1);
                if viz.encoder.should_capture(token_index) {
                    let layers = state.take_viz();
                    let frame = viz.encoder.step(&layers, token_index);
                    viz.sender.try_send(frame);
                }
            }

            if stream_err.is_some() {
                break;
            }

            #[cfg(feature = "viz")]
            if let Some(viz) = viz_runtime.as_ref() {
                if viz.stop.load(Ordering::Relaxed) {
                    break;
                }
            }

            if let ContextStrategy::Sliding { window } = strategy
                && window > 0
                && state.position > window
            {
                state.trim(window);
            }
        }

        if let Some(err) = stream_err {
            return Err(err);
        }

        writer
            .write_all(b"\n")
            .context("failed to write trailing newline")?;
        writer.flush().context("failed to flush stdout")?;
    } else {
        let output = generate_text::<B>(
            &model,
            tokenizer.as_ref(),
            &device,
            &config.training,
            &generation,
        )?;

        eprintln!("{status_msg}");
        println!("{output}");
    }

    Ok(())
}

#[cfg(feature = "viz")]
fn infer_backend_with_viz<B, Init>(
    config: &TrainingConfig,
    args: &Args,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: Backend<Device = burn_wgpu::WgpuDevice> + 'static,
    B::Device: Default + Clone + Send + Sync + 'static,
    Init: Fn(&B::Device) + Copy + Send + 'static,
    (): bevy_burn::gpu_burn_to_bevy::BurnBevyPrepare<B>,
{
    let model_config = build_model_config(&config.model, config.training.block_size);
    let dims = VizDimensions {
        layers: model_config.n_layer,
        heads: model_config.n_head,
        latent_per_head: model_config.latent_per_head(),
    };
    let viz_config = VizConfig::default();

    let (exit_tx, exit_rx) = mpsc::channel();
    let overlay = viz::start_overlay_native::<B>(viz_config.clone(), dims, Some(exit_rx));
    let stop_flag = overlay.handle().stop_flag();
    let (viz_handle, mut app) = overlay.split();
    let device = viz_handle.device().clone();
    let sender = viz_handle.sender();

    #[cfg(not(target_arch = "wasm32"))]
    {
        let ctrlc_stop = stop_flag.clone();
        let ctrlc_exit = exit_tx.clone();
        let _ = ctrlc::set_handler(move || {
            ctrlc_stop.store(true, Ordering::Relaxed);
            let _ = ctrlc_exit.send(());
        });
    }

    let backend_name = backend_name.to_string();
    let config = config.clone();
    let args = args.clone();
    let stop_for_thread = stop_flag.clone();
    let infer_thread = std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let viz_runtime = VizRuntime {
                encoder: VizEncoder::new(
                    viz_config,
                    dims.layers,
                    dims.heads,
                    dims.latent_per_head,
                    &device,
                ),
                sender,
                stop: stop_for_thread,
            };
            infer_backend_on_device::<B, Init>(
                &config,
                &args,
                backend_name.as_str(),
                device,
                init_backend,
                Some(viz_runtime),
            )
        }));

        let _ = exit_tx.send(());
        match result {
            Ok(outcome) => outcome,
            Err(_) => Err(anyhow!("inference thread panicked")),
        }
    });

    app.run();
    let result = infer_thread
        .join()
        .map_err(|_| anyhow!("inference thread crashed"))??;
    Ok(result)
}


fn apply_generation_overrides(generation: &mut GenerationConfig, args: &Args, block_size: usize) {
    if let Some(prompt) = &args.prompt {
        generation.prompt = prompt.clone();
    }
    if let Some(max_tokens) = args.max_tokens {
        generation.max_tokens = if max_tokens < 0 {
            None
        } else {
            Some(max_tokens)
        };
    }
    if let Some(temperature) = args.temperature {
        generation.temperature = temperature;
    }
    if let Some(top_k) = args.top_k {
        generation.top_k = Some(top_k);
    }
    if let Some(mode) = args.context_mode {
        generation.context_strategy = match mode {
            ContextModeArg::Infinite => ContextStrategyConfig::Infinite,
            ContextModeArg::Sliding => ContextStrategyConfig::Sliding {
                window: args.context_window.unwrap_or(block_size).max(1),
            },
        };
    }
}

fn resolve_checkpoint_base(path: &Path, epoch: Option<usize>) -> Result<(PathBuf, usize)> {
    if path.is_dir() {
        let target_epoch = epoch.unwrap_or(find_latest_epoch(path)?);
        let base = path.join(format!("model-{target_epoch}"));
        ensure_checkpoint_exists(&base)?;
        return Ok((base, target_epoch));
    }

    let mut base = if path.extension().is_some() {
        let mut without_ext = path.to_path_buf();
        without_ext.set_extension("");
        without_ext
    } else {
        path.to_path_buf()
    };

    let detected_epoch = parse_epoch_from_stem(&base);
    let target_epoch = match (epoch, detected_epoch) {
        (Some(explicit), Some(detected)) if explicit != detected => {
            let parent = base.parent().map(Path::to_path_buf).unwrap_or_default();
            base = parent.join(format!("model-{explicit}"));
            explicit
        }
        (Some(explicit), _) => {
            if detected_epoch.is_none() {
                let parent = base
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("runs").join("checkpoint"));
                base = parent.join(format!("model-{explicit}"));
            }
            explicit
        }
        (None, Some(detected)) => detected,
        (None, None) => {
            return Err(anyhow!(
                "unable to infer checkpoint epoch from {}; provide --epoch",
                path.display()
            ));
        }
    };

    ensure_checkpoint_exists(&base)?;
    Ok((base, target_epoch))
}

fn ensure_checkpoint_exists(base: &Path) -> Result<()> {
    let mut candidate = base.to_path_buf();
    candidate.set_extension("bin");
    if candidate.is_file() {
        return Ok(());
    }

    Err(anyhow!("checkpoint file {}.bin not found", base.display()))
}

fn find_latest_epoch(dir: &Path) -> Result<usize> {
    let mut max_epoch = None;
    for entry in fs::read_dir(dir)
        .with_context(|| format!("failed to read checkpoint directory {}", dir.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let mut base = entry.path();
        base.set_extension("");
        if let Some(epoch) = parse_epoch_from_stem(&base) {
            let updated = max_epoch
                .map(|current: usize| current.max(epoch))
                .unwrap_or(epoch);
            max_epoch = Some(updated);
        }
    }

    max_epoch.ok_or_else(|| anyhow!("no model checkpoints found in {}", dir.display()))
}

fn parse_epoch_from_stem(path: &Path) -> Option<usize> {
    let stem = path.file_name()?.to_string_lossy();
    let stem = stem.strip_suffix(".bin").unwrap_or(&stem);
    let epoch_part = stem.strip_prefix("model-")?;
    epoch_part.parse().ok()
}

fn format_checkpoint(base: &Path) -> String {
    let mut path = base.to_path_buf();
    path.set_extension("bin");
    path.display().to_string()
}

#[derive(Debug, Deserialize, Default)]
struct RunConfigJson {
    #[serde(default)]
    block_size: Option<usize>,
    #[serde(default)]
    overrides: ModelOverrides,
}

fn apply_run_config(config: &mut TrainingConfig, run_config: &RunConfigJson) {
    let block_override = run_config
        .block_size
        .or(run_config.overrides.block_size)
        .map(|value| value.max(1));
    if let Some(block_size) = block_override {
        config.training.block_size = block_size;
    }
    merge_model_overrides(&mut config.model, &run_config.overrides);
}

fn merge_model_overrides(base: &mut ModelOverrides, incoming: &ModelOverrides) {
    if let Some(value) = incoming.n_layer {
        base.n_layer = Some(value);
    }
    if let Some(value) = incoming.n_embd {
        base.n_embd = Some(value);
    }
    if let Some(value) = incoming.n_head {
        base.n_head = Some(value);
    }
    if let Some(value) = incoming.mlp_internal_dim_multiplier {
        base.mlp_internal_dim_multiplier = Some(value);
    }
    if let Some(value) = incoming.relu_threshold {
        base.relu_threshold = Some(value);
    }
    if let Some(value) = incoming.dropout {
        base.dropout = Some(value);
    }
    if let Some(value) = incoming.fused_kernels {
        base.fused_kernels = Some(value);
    }
    if let Some(value) = incoming.block_size {
        base.block_size = Some(value);
    }
    if let Some(value) = incoming.rotary_embedding {
        base.rotary_embedding = Some(value);
    }
}

fn resolve_run_config_path(
    checkpoint: Option<&PathBuf>,
    backend_name: &str,
) -> Option<PathBuf> {
    let checkpoint_path = checkpoint
        .cloned()
        .unwrap_or_else(|| default_checkpoint_dir(backend_name));
    let mut candidates = Vec::new();

    if checkpoint_path.is_dir() {
        candidates.push(checkpoint_path.join("config.json"));
        if checkpoint_path
            .file_name()
            .is_some_and(|name| name == "checkpoint")
            && let Some(parent) = checkpoint_path.parent()
        {
            candidates.push(parent.join("config.json"));
        }
    } else if let Some(parent) = checkpoint_path.parent() {
        candidates.push(parent.join("config.json"));
        if parent
            .file_name()
            .is_some_and(|name| name == "checkpoint")
            && let Some(grandparent) = parent.parent()
        {
            candidates.push(grandparent.join("config.json"));
        }
    }

    candidates.into_iter().find(|path| path.is_file())
}

fn default_checkpoint_dir(backend_name: &str) -> PathBuf {
    resolve_latest_run_dir(backend_name)
        .map(|dir| dir.join("checkpoint"))
        .unwrap_or_else(|| PathBuf::from("runs").join("checkpoint"))
}

fn resolve_latest_run_dir(backend_name: &str) -> Option<PathBuf> {
    let run_root = PathBuf::from("runs");
    resolve_latest_run_dir_from(&run_root).or_else(|| {
        let device_root = run_root.join(backend_name);
        resolve_latest_run_dir_from(&device_root)
    })
}

fn resolve_latest_run_dir_from(run_root: &Path) -> Option<PathBuf> {
    let latest_path = run_root.join("latest");
    let contents = fs::read_to_string(&latest_path).ok()?;
    let name = contents.trim();
    if name.is_empty() {
        return None;
    }
    Some(run_root.join(name))
}

#[derive(Parser, Debug, Clone)]
#[command(
    author,
    version,
    about = "Run inference with a trained Baby Dragon Hatchling model"
)]
struct Args {
    /// Additional configuration files applied in order (later files override earlier ones).
    #[arg(short = 'c', long = "config", value_name = "PATH")]
    config: Vec<PathBuf>,
    /// Backend to use for inference.
    #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
    backend: BackendArg,
    /// Path to the checkpoint directory or file.
    #[arg(long, value_name = "PATH")]
    checkpoint: Option<PathBuf>,
    /// Specific checkpoint epoch to load.
    #[arg(long, value_name = "N")]
    epoch: Option<usize>,
    /// Override the prompt used for generation.
    #[arg(long)]
    prompt: Option<String>,
    /// Override the number of tokens to generate.
    #[arg(long, value_name = "N")]
    max_tokens: Option<i64>,
    /// Override the sampling temperature.
    #[arg(long, value_name = "T")]
    temperature: Option<f32>,
    /// Override the top-k sampling parameter.
    #[arg(long, value_name = "K")]
    top_k: Option<usize>,
    /// Override the context strategy.
    #[arg(long, value_enum)]
    context_mode: Option<ContextModeArg>,
    /// Sliding window size when using `--context-mode=sliding`.
    #[arg(long, value_name = "N")]
    context_window: Option<usize>,
    /// Stream tokens to stdout as they are generated.
    #[arg(long)]
    streaming: bool,
    /// whether or not to spawn visualization app, wgpu only
    #[arg(long)]
    #[cfg(feature = "viz")]
    viz: bool,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendArg {
    Wgpu,
    Cuda,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ContextModeArg {
    Infinite,
    Sliding,
}


fn backend_name(backend: BackendArg) -> &'static str {
    match backend {
        BackendArg::Wgpu => "wgpu",
        BackendArg::Cuda => "cuda",
    }
}

fn normalize_max_tokens(max_tokens: Option<i64>) -> Option<usize> {
    match max_tokens {
        Some(value) if value >= 0 => Some(value as usize),
        _ => None,
    }
}
