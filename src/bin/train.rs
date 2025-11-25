#![recursion_limit = "256"]

use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};

use burn::data::dataloader::{DataLoader, DataLoaderIterator};
use burn::lr_scheduler::{
    LrScheduler,
    cosine::{CosineAnnealingLrScheduler, CosineAnnealingLrSchedulerConfig},
    exponential::{ExponentialLrScheduler, ExponentialLrSchedulerConfig},
    linear::{LinearLrScheduler, LinearLrSchedulerConfig},
    noam::{NoamLrScheduler, NoamLrSchedulerConfig},
    step::{StepLrScheduler, StepLrSchedulerConfig},
};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, LearningRate};
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{Tensor, activation};
use burn_autodiff::Autodiff;
use burn_train::Interrupter;
use burn_train::metric::{
    Adaptor, ItemLazy, LearningRateMetric, LossInput, LossMetric, MetricEntry, MetricId,
    NumericEntry, SerializedEntry,
};
use burn_train::renderer::tui::TuiMetricsRenderer;
use burn_train::renderer::{MetricState, MetricsRendererTraining, TrainingProgress};
use burn_train::{
    LearnerBuilder, LearningStrategy, TrainOutput, TrainStep, TrainingResult, ValidStep,
};
use burn_wgpu::Wgpu;
use tracing::info;

#[cfg(feature = "cuda")]
use burn_cuda::Cuda;

use burn::prelude::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

use burn_dragon_hatchling::wgpu::init_runtime;
use burn_dragon_hatchling::{
    BDH, BDHConfig, BdhEsConfig, ContextStrategyConfig, Dataset, DatasetConfig, DatasetSplit,
    EggrollObjective, EggrollTrainer, LearningRateScheduleConfig, ModelOverrides, OptimizerConfig,
    RandomDataLoader, SequenceBatch, TrainingConfig, TrainingHyperparameters, TrainingMode,
    bdh_param_specs, build_dataset, language_model_loss, load_training_config,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Train the Baby Dragon Hatchling model")]
struct Cli {
    #[command(flatten)]
    train: TrainArgs,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(ClapArgs, Debug)]
struct TrainArgs {
    /// Additional configuration files applied in order (later files override earlier ones).
    #[arg(short = 'c', long = "config", value_name = "PATH", global = true)]
    config: Vec<PathBuf>,
    /// Backend to use for training.
    #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
    backend: BackendArg,
    /// Optional model checkpoint to initialize training from.
    #[arg(long, value_name = "PATH")]
    checkpoint: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Build the character-level vocabulary and exit.
    BuildVocab,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendArg {
    Cuda,
    Wgpu,
}

#[derive(Clone)]
struct LanguageModelOutput<B: BackendTrait> {
    loss: Tensor<B, 1>,
}

impl<B: BackendTrait> LanguageModelOutput<B> {
    fn new(loss: Tensor<B, 1>) -> Self {
        Self { loss }
    }
}

impl<B: BackendTrait> ItemLazy for LanguageModelOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl<B: BackendTrait> Adaptor<LossInput<B>> for LanguageModelOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

struct LanguageModelTrainItem<B: AutodiffBackend> {
    loss: Tensor<B, 1>,
}

impl<B: AutodiffBackend> LanguageModelTrainItem<B> {
    fn new(loss: Tensor<B, 1>) -> Self {
        Self { loss }
    }
}

impl<B: AutodiffBackend> ItemLazy for LanguageModelTrainItem<B> {
    type ItemSync = LanguageModelOutput<B::InnerBackend>;

    fn sync(self) -> Self::ItemSync {
        LanguageModelOutput::new(self.loss.inner())
    }
}

type ValidBackend<B> = <B as AutodiffBackend>::InnerBackend;

fn forward_with_streaming<B: BackendTrait>(
    model: &BDH<B>,
    batch: &SequenceBatch<B>,
) -> Tensor<B, 3> {
    let Some(stream) = &batch.stream else {
        return model.forward(batch.inputs.clone());
    };

    if !stream.has_streams() {
        return model.forward(batch.inputs.clone());
    }

    let batch_size = batch.inputs.shape().dims::<2>()[0];
    let time = batch.inputs.shape().dims::<2>()[1];
    let device = batch.inputs.device();

    let stream_mask: Vec<bool> = stream.entries.iter().map(|entry| entry.is_some()).collect();
    let stream_indices: Vec<usize> = stream
        .entries
        .iter()
        .enumerate()
        .filter_map(|(idx, entry)| entry.as_ref().map(|_| idx))
        .collect();
    if stream_indices.is_empty() {
        return model.forward(batch.inputs.clone());
    }

    let fresh_indices: Vec<usize> = stream
        .entries
        .iter()
        .enumerate()
        .filter_map(|(idx, entry)| entry.is_none().then_some(idx))
        .collect();

    let contiguous_stream = stream_indices
        .iter()
        .copied()
        .eq(0..stream_indices.len())
        && stream_mask
            .iter()
            .take_while(|flag| **flag)
            .count()
            == stream_indices.len();
    debug_assert!(
        contiguous_stream,
        "stream rows should be the leading positions in the batch"
    );

    let stream_inputs = if contiguous_stream {
        batch
            .inputs
            .clone()
            .slice_dim(0, 0..stream_indices.len())
    } else {
        let mut slices = Vec::with_capacity(stream_indices.len());
        for &idx in &stream_indices {
            slices.push(batch.inputs.clone().slice_dim(0, idx..idx + 1));
        }
        Tensor::cat(slices, 0)
    };

    let stream_pool = stream.pool.clone();
    let mut pool_guard = stream_pool.lock().expect("lock stream pool");
    let pool = pool_guard.ensure_pool(|max_streams| model.init_state_pool(max_streams, &device));
    let state_view = pool.prefix_view(stream_indices.len());
    drop(pool_guard);

    let (logits_stream, updated_states) =
        model.forward_with_state_pool(stream_inputs, state_view, false);

    let mut logits_ordered: Vec<Option<Tensor<B, 3>>> = vec![None; batch_size];
    for (pos, &idx) in stream_indices.iter().enumerate() {
        logits_ordered[idx] = Some(logits_stream.clone().slice_dim(0, pos..pos + 1));
    }

    let mut pool_guard = stream_pool.lock().expect("lock stream pool");
    let pool = pool_guard.ensure_pool(|max_streams| model.init_state_pool(max_streams, &device));
    pool.write_prefix(stream_indices.len(), &updated_states, stream.max_context, time);
    drop(pool_guard);

    if !fresh_indices.is_empty() {
        let fresh_inputs = if contiguous_stream && stream_indices.len() < batch_size {
            batch
                .inputs
                .clone()
                .slice_dim(0, stream_indices.len()..batch_size)
        } else if fresh_indices.len() == 1 {
            batch
                .inputs
                .clone()
                .slice_dim(0, fresh_indices[0]..fresh_indices[0] + 1)
        } else {
            let mut slices = Vec::with_capacity(fresh_indices.len());
            for &idx in &fresh_indices {
                slices.push(batch.inputs.clone().slice_dim(0, idx..idx + 1));
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
        .map(|item| item.expect("all logits present"))
        .collect();
    Tensor::cat(ordered, 0)
}

impl<B: AutodiffBackend> TrainStep<SequenceBatch<B>, LanguageModelTrainItem<B>> for BDH<B> {
    fn step(&self, batch: SequenceBatch<B>) -> TrainOutput<LanguageModelTrainItem<B>> {
        let logits = forward_with_streaming(self, &batch);
        let loss = language_model_loss::<B>(logits, batch.targets);
        let grads = loss.backward();

        TrainOutput::new(self, grads, LanguageModelTrainItem::new(loss))
    }
}

impl<B: BackendTrait> ValidStep<SequenceBatch<B>, LanguageModelOutput<B>> for BDH<B> {
    fn step(&self, batch: SequenceBatch<B>) -> LanguageModelOutput<B> {
        let logits = forward_with_streaming(self, &batch);
        let loss = language_model_loss::<B>(logits, batch.targets);
        LanguageModelOutput::new(loss)
    }
}

struct LanguageModelEsObjective<B: BackendTrait> {
    backend: core::marker::PhantomData<B>,
}

impl<B: BackendTrait> Default for LanguageModelEsObjective<B> {
    fn default() -> Self {
        Self {
            backend: core::marker::PhantomData,
        }
    }
}

impl<B: BackendTrait> EggrollObjective<BDH<B>, B> for LanguageModelEsObjective<B> {
    type Batch = SequenceBatch<B>;

    fn evaluate(&mut self, logits: &Tensor<B, 3>, batch: &Self::Batch) -> f32 {
        let loss = language_model_loss::<B>(logits.clone(), batch.targets.clone());
        let data = loss
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap_or_default();
        data.first().copied().unwrap_or(0.0)
    }

    fn evaluate_population(&mut self, logits: &Tensor<B, 4>, batch: &Self::Batch) -> Vec<f32> {
        let [pop, batch_size, time, _vocab] = logits.shape().dims::<4>();
        let targets = batch
            .targets
            .clone()
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, pop);
        let log_probs = activation::log_softmax(logits.clone(), 3);
        let gathered = log_probs.gather(3, targets.unsqueeze_dim::<4>(3));
        let token_loss = gathered.neg().reshape([pop, batch_size * time]);
        let loss_mean = token_loss
            .sum_dim(1)
            .div_scalar((batch_size * time) as f32)
            .reshape([pop]);

        loss_mean
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap_or_default()
    }

    fn forward_population(
        &self,
        model: &BDH<B>,
        batch: &Self::Batch,
        noiser: &burn_dragon_hatchling::eggroll::EggrollNoiser<B>,
        tree_key: &burn_dragon_hatchling::eggroll::EsTreeKey,
        step: u64,
        global_workers: &[u32],
        deterministic: bool,
    ) -> Option<Tensor<B, 4>> {
        Some(model.forward_population_with_noise(
            &batch.inputs,
            noiser,
            tree_key,
            step,
            global_workers,
            deterministic,
        ))
    }

    fn evaluate_with_noise(
        &mut self,
        model: &BDH<B>,
        batch: &Self::Batch,
        noiser: &burn_dragon_hatchling::eggroll::EggrollNoiser<B>,
        es_key: &burn_dragon_hatchling::eggroll::EsTreeKey,
        thread_id: u32,
        deterministic: bool,
    ) -> f32 {
        let logits = model.forward_with_noise_det(
            batch.inputs.clone(),
            noiser,
            es_key,
            thread_id,
            deterministic,
        );
        self.evaluate(&logits, batch)
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Cli::parse();

    let mut config_paths = vec![PathBuf::from("config/base.toml")];
    config_paths.extend(args.train.config.clone());
    let config = load_training_config(&config_paths)?;

    if matches!(args.command, Some(Command::BuildVocab)) {
        build_vocab_only(&config)?;
        return Ok(());
    }

    let dataset = prepare_dataset(&config.dataset, &config.training)?;
    let mode = config.mode.clone();

    match args.train.backend {
        BackendArg::Wgpu => match mode {
            TrainingMode::Eggroll => train_backend_eggroll::<Wgpu<f32>, _>(
                &config,
                Arc::clone(&dataset),
                "wgpu",
                init_runtime,
                args.train.checkpoint.as_ref(),
            ),
            TrainingMode::Backprop => train_backend::<Autodiff<Wgpu<f32>>, _>(
                &config,
                Arc::clone(&dataset),
                "wgpu",
                init_runtime,
            ),
        },
        BackendArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                match mode {
                    TrainingMode::Eggroll => train_backend_eggroll::<Cuda<f32>, _>(
                        &config,
                        dataset,
                        "cuda",
                        |_| {},
                        args.train.checkpoint.as_ref(),
                    ),
                    TrainingMode::Backprop => {
                        train_backend::<Autodiff<Cuda<f32>>, _>(&config, dataset, "cuda", |_| {})
                    }
                }
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

fn train_backend<B, Init>(
    config: &TrainingConfig,
    dataset: Arc<Dataset>,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    let device = B::Device::default();
    B::seed(&device, 42);
    init_backend(&device);

    let training = &config.training;
    let optimizer_cfg = &config.optimizer;

    let mut model_config = build_model_config(&config.model);
    let tokenizer = dataset.tokenizer();
    model_config.vocab_size = tokenizer.len();

    let steps_per_epoch = dataset.steps_per_epoch(DatasetSplit::Train);
    let total_steps = training.max_iters.max(1);
    let total_epochs = usize::max(1, total_steps.div_ceil(steps_per_epoch));
    let stream_context = resolve_training_context_window(training);

    info!(
        "train schedule: steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, epochs={total_epochs}"
    );
    let train_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>> =
        Arc::new(RandomDataLoader::<B>::new(
            Arc::clone(&dataset),
            DatasetSplit::Train,
            &device,
            steps_per_epoch,
            Some(total_steps),
            training.stream_retain_pct,
            stream_context,
        ));

    let val_steps_per_epoch = dataset.steps_per_epoch(DatasetSplit::Val);
    let desired_valid_steps = usize::max(1, total_steps / training.log_frequency.max(1));
    let valid_steps = desired_valid_steps.min(val_steps_per_epoch).max(1);

    let valid_device = device.clone();
    let valid_loader: Arc<dyn DataLoader<ValidBackend<B>, SequenceBatch<ValidBackend<B>>>> =
        Arc::new(RandomDataLoader::<ValidBackend<B>>::new(
            Arc::clone(&dataset),
            DatasetSplit::Val,
            &valid_device,
            valid_steps,
            None,
            0.0,
            None,
        ));

    let mut model = Some(BDH::<B>::new(model_config.clone(), &device));
    let mut optim = Some(
        AdamWConfig::new()
            .with_weight_decay(optimizer_cfg.weight_decay)
            .init::<B, BDH<B>>(),
    );
    let scheduler = resolve_lr_scheduler(optimizer_cfg, total_steps, &model_config)?;

    let run_dir = PathBuf::from("runs").join(backend_name);
    let context = TrainEnvironment {
        run_dir: &run_dir,
        backend_name,
        training,
        model_config: &model_config,
        device: &device,
        train_loader,
        valid_loader,
        epochs: total_epochs,
    };
    let _model = match scheduler {
        ResolvedLrScheduler::Constant(lr) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            lr,
        )?,
        ResolvedLrScheduler::Cosine(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
        ResolvedLrScheduler::Linear(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
        ResolvedLrScheduler::Exponential(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
        ResolvedLrScheduler::Step(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
        ResolvedLrScheduler::Noam(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
    };

    info!("Training complete on {backend_name}");

    Ok(())
}

fn train_backend_eggroll<B, Init>(
    config: &TrainingConfig,
    dataset: Arc<Dataset>,
    backend_name: &str,
    init_backend: Init,
    checkpoint: Option<&PathBuf>,
) -> Result<()>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    let device = B::Device::default();
    B::seed(&device, 42);
    init_backend(&device);

    let training = &config.training;
    let mut model_config = build_model_config(&config.model);
    let tokenizer = dataset.tokenizer();
    model_config.vocab_size = tokenizer.len();

    let total_steps = training.max_iters.max(1);
    let steps_per_epoch = dataset.steps_per_epoch(DatasetSplit::Train);
    let train_loader_base = RandomDataLoader::<B>::new(
        Arc::clone(&dataset),
        DatasetSplit::Train,
        &device,
        steps_per_epoch,
        Some(total_steps),
        0.0,
        None,
    );
    let mut train_loader: Box<dyn DataLoaderIterator<SequenceBatch<B>>> = train_loader_base.iter();

    let eggroll_cfg = resolve_bdh_eggroll(config);
    let mut model = BDH::<B>::new(model_config.clone(), &device);
    if let Some(path) = checkpoint {
        info!("Loading checkpoint from {}", path.display());
        let record =
            BinFileRecorder::<FullPrecisionSettings>::new().load(path.to_path_buf(), &device)?;
        model = model.load_record(record);
    }
    let param_specs = bdh_param_specs(&model, &eggroll_cfg);
    let mut trainer = EggrollTrainer::new(
        model,
        eggroll_cfg.eggroll.clone(),
        param_specs,
        LanguageModelEsObjective::<B>::default(),
    );

    let use_tui = std::io::stdout().is_terminal();
    let interrupter = Interrupter::new();
    let mut tui = use_tui.then(|| TuiMetricsRenderer::new(interrupter.clone(), None));
    let epoch_total = total_steps.div_ceil(steps_per_epoch);
    let run_dir = PathBuf::from("runs").join(backend_name).join("eggroll");
    fs::create_dir_all(&run_dir)?;

    for step_idx in 0..total_steps {
        let step_start = Instant::now();
        let batch = match train_loader.next() {
            Some(batch) => batch,
            None => {
                train_loader = train_loader_base.iter();
                train_loader
                    .next()
                    .ok_or_else(|| anyhow!("data loader yielded no batches"))?
            }
        };

        let _model_ref = trainer.step(&batch);

        if let Some(tui) = tui.as_mut() {
            let items_processed = step_idx + 1;
            let mean = trainer.state.last_fitness_mean.unwrap_or(0.0);
            let std_dev = trainer.state.last_fitness_std.unwrap_or(0.0);
            let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
            let progress = TrainingProgress {
                progress: burn::data::dataloader::Progress {
                    items_processed,
                    items_total: total_steps,
                },
                epoch: step_idx / steps_per_epoch + 1,
                epoch_total,
                iteration: items_processed,
            };
            let metric = MetricEntry::new(
                MetricId::new(Arc::new("fitness_mean".to_string())),
                SerializedEntry::new(format!("{:.4}", mean), format!("{}", mean)),
            );
            let metric_std = MetricEntry::new(
                MetricId::new(Arc::new("fitness_std".to_string())),
                SerializedEntry::new(format!("{:.4}", std_dev), format!("{}", std_dev)),
            );
            let metric_step = MetricEntry::new(
                MetricId::new(Arc::new("step_ms".to_string())),
                SerializedEntry::new(format!("{:.1}", step_ms), format!("{}", step_ms)),
            );
            let metric_pop = MetricEntry::new(
                MetricId::new(Arc::new("pop_size".to_string())),
                SerializedEntry::new(
                    trainer.state.config.pop_size.to_string(),
                    trainer.state.config.pop_size.to_string(),
                ),
            );
            let metric_sigma = MetricEntry::new(
                MetricId::new(Arc::new("sigma".to_string())),
                SerializedEntry::new(
                    format!("{:.4}", trainer.state.config.sigma),
                    format!("{}", trainer.state.config.sigma),
                ),
            );
            let metric_lr = MetricEntry::new(
                MetricId::new(Arc::new("lr".to_string())),
                SerializedEntry::new(
                    format!("{:.5}", trainer.state.config.lr),
                    format!("{}", trainer.state.config.lr),
                ),
            );
            tui.update_train(MetricState::Numeric(
                metric,
                NumericEntry::Value(mean as f64),
            ));
            tui.update_train(MetricState::Numeric(
                metric_std,
                NumericEntry::Value(std_dev as f64),
            ));
            tui.update_train(MetricState::Numeric(
                metric_step,
                NumericEntry::Value(step_ms),
            ));
            tui.update_train(MetricState::Numeric(
                metric_pop,
                NumericEntry::Value(trainer.state.config.pop_size as f64),
            ));
            tui.update_train(MetricState::Numeric(
                metric_sigma,
                NumericEntry::Value(trainer.state.config.sigma as f64),
            ));
            tui.update_train(MetricState::Numeric(
                metric_lr,
                NumericEntry::Value(trainer.state.config.lr as f64),
            ));
            tui.render_train(progress);
            if interrupter.should_stop() {
                info!("Eggroll training interrupted by user");
                break;
            }
        }

        if (step_idx + 1) % training.log_frequency.max(1) == 0 {
            info!(
                "[eggroll:{backend}] step={}/{}",
                step_idx + 1,
                total_steps,
                backend = backend_name
            );
        }
    }

    let checkpoint_path = run_dir.join("checkpoint.bin");
    let record = trainer.model.clone().into_record();
    BinFileRecorder::<FullPrecisionSettings>::new().record(record, checkpoint_path.clone())?;
    info!("Saved eggroll checkpoint to {}", checkpoint_path.display());

    info!("Eggroll training complete on {backend_name}");
    Ok(())
}

enum ResolvedLrScheduler {
    Constant(LearningRate),
    Cosine(CosineAnnealingLrScheduler),
    Linear(LinearLrScheduler),
    Exponential(ExponentialLrScheduler),
    Step(StepLrScheduler),
    Noam(NoamLrScheduler),
}

struct TrainEnvironment<'a, B>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    run_dir: &'a Path,
    backend_name: &'a str,
    training: &'a TrainingHyperparameters,
    model_config: &'a BDHConfig,
    device: &'a B::Device,
    train_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>>,
    valid_loader: Arc<dyn DataLoader<ValidBackend<B>, SequenceBatch<ValidBackend<B>>>>,
    epochs: usize,
}

fn train_with_scheduler<B, S>(
    env: &TrainEnvironment<'_, B>,
    model: BDH<B>,
    optimizer: OptimizerAdaptor<AdamW, BDH<B>, B>,
    scheduler: S,
) -> Result<BDH<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    S: LrScheduler + 'static,
{
    fs::create_dir_all(env.run_dir)?;

    let builder = LearnerBuilder::new(env.run_dir)
        .num_epochs(env.epochs)
        .with_file_checkpointer(BinFileRecorder::<FullPrecisionSettings>::new())
        .metric_train_numeric(LossMetric::<ValidBackend<B>>::new())
        .metric_valid_numeric(LossMetric::<ValidBackend<B>>::new())
        .metric_train_numeric(LearningRateMetric::new())
        .summary();

    let learner = builder.build(
        model,
        optimizer,
        scheduler,
        LearningStrategy::SingleDevice(env.device.clone()),
    );

    let TrainingResult { model, .. } =
        learner.fit(Arc::clone(&env.train_loader), Arc::clone(&env.valid_loader));

    log_theoretical_profile(
        env.model_config,
        env.training.batch_size,
        env.training.block_size,
        env.backend_name,
    );

    Ok(model)
}

fn resolve_training_context_window(training: &TrainingHyperparameters) -> Option<usize> {
    if let Some(max_ctx) = training.stream_max_context {
        if max_ctx == 0 {
            return None;
        }
        return Some(max_ctx.max(1));
    }

    match &training.context_strategy {
        ContextStrategyConfig::Infinite => None,
        ContextStrategyConfig::Sliding { window } => {
            let win = if *window == 0 {
                training.block_size.max(1)
            } else {
                *window
            };
            Some(win)
        }
    }
}

fn resolve_lr_scheduler(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
    model_config: &BDHConfig,
) -> Result<ResolvedLrScheduler> {
    let base_lr = optimizer_cfg.learning_rate;
    let fallback_iters = total_steps.max(1);

    let schedule = match &optimizer_cfg.lr_schedule {
        None => ResolvedLrScheduler::Constant(base_lr),
        Some(LearningRateScheduleConfig::Constant { initial_lr }) => {
            ResolvedLrScheduler::Constant(initial_lr.unwrap_or(base_lr))
        }
        Some(LearningRateScheduleConfig::Cosine {
            initial_lr,
            min_lr,
            num_iters,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = CosineAnnealingLrSchedulerConfig::new(
                init_lr,
                num_iters.unwrap_or(fallback_iters).max(1),
            )
            .with_min_lr(min_lr.unwrap_or(0.0))
            .init()
            .map_err(|err| anyhow!("failed to initialize cosine lr scheduler: {err}"))?;
            ResolvedLrScheduler::Cosine(scheduler)
        }
        Some(LearningRateScheduleConfig::Linear {
            initial_lr,
            final_lr,
            num_iters,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = LinearLrSchedulerConfig::new(
                init_lr,
                *final_lr,
                num_iters.unwrap_or(fallback_iters).max(1),
            )
            .init()
            .map_err(|err| anyhow!("failed to initialize linear lr scheduler: {err}"))?;
            ResolvedLrScheduler::Linear(scheduler)
        }
        Some(LearningRateScheduleConfig::Exponential { initial_lr, gamma }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = ExponentialLrSchedulerConfig::new(init_lr, *gamma)
                .init()
                .map_err(|err| anyhow!("failed to initialize exponential lr scheduler: {err}"))?;
            ResolvedLrScheduler::Exponential(scheduler)
        }
        Some(LearningRateScheduleConfig::Step {
            initial_lr,
            gamma,
            step_size,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler =
                StepLrSchedulerConfig::new(init_lr, step_size.unwrap_or(fallback_iters).max(1))
                    .with_gamma(*gamma)
                    .init()
                    .map_err(|err| anyhow!("failed to initialize step lr scheduler: {err}"))?;
            ResolvedLrScheduler::Step(scheduler)
        }
        Some(LearningRateScheduleConfig::Noam {
            initial_lr,
            warmup_steps,
            model_size,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let mut config = NoamLrSchedulerConfig::new(init_lr);
            config = config.with_warmup_steps(warmup_steps.unwrap_or(fallback_iters).max(1));
            config = config.with_model_size(model_size.unwrap_or(model_config.n_embd).max(1));
            let scheduler = config
                .init()
                .map_err(|err| anyhow!("failed to initialize noam lr scheduler: {err}"))?;
            ResolvedLrScheduler::Noam(scheduler)
        }
    };

    Ok(schedule)
}

fn resolve_bdh_eggroll(config: &TrainingConfig) -> BdhEsConfig {
    let mut base = BdhEsConfig::default();
    if let Some(es) = &config.eggroll {
        base.eggroll = es.clone().into_runtime();
    }
    base
}

fn build_vocab_only(config: &TrainingConfig) -> Result<()> {
    let dataset = prepare_dataset(&config.dataset, &config.training)?;
    let tokenizer = dataset.tokenizer();
    info!(
        "Tokenizer `{}` ready with {} tokens",
        config.dataset.tokenizer.kind_name(),
        tokenizer.len()
    );
    Ok(())
}

fn prepare_dataset(
    dataset_cfg: &DatasetConfig,
    training: &TrainingHyperparameters,
) -> Result<Arc<Dataset>> {
    let tokenizer_path = dataset_cfg.tokenizer.storage_path(&dataset_cfg.cache_dir);
    let tokenizer_preexists = tokenizer_path
        .as_ref()
        .map(|path| path.is_file())
        .unwrap_or(false);

    let (dataset_enum, dataset_summary) = build_dataset(dataset_cfg, training)?;
    let dataset = Arc::new(dataset_enum);

    let tokenizer = dataset.tokenizer();
    match tokenizer_path {
        Some(path) if tokenizer_preexists => info!(
            "Loaded {} tokenizer with {} tokens from {}",
            dataset_cfg.tokenizer.kind_name(),
            tokenizer.len(),
            path.display()
        ),
        Some(path) => info!(
            "Built {} tokenizer with {} tokens at {}",
            dataset_cfg.tokenizer.kind_name(),
            tokenizer.len(),
            path.display()
        ),
        None => info!(
            "Initialized {} tokenizer with {} tokens (no persistence required)",
            dataset_cfg.tokenizer.kind_name(),
            tokenizer.len()
        ),
    };

    info!("{dataset_summary}");

    Ok(dataset)
}

fn build_model_config(overrides: &ModelOverrides) -> BDHConfig {
    let mut model_config = BDHConfig::default();

    if let Some(n_layer) = overrides.n_layer {
        model_config.n_layer = n_layer;
    }
    if let Some(n_embd) = overrides.n_embd {
        model_config.n_embd = n_embd;
    }
    if let Some(n_head) = overrides.n_head {
        model_config.n_head = n_head;
    }
    if let Some(multiplier) = overrides.mlp_internal_dim_multiplier {
        model_config.mlp_internal_dim_multiplier = multiplier;
    }
    if let Some(dropout) = overrides.dropout {
        model_config.dropout = dropout;
    }
    if let Some(enabled) = overrides.fused_kernels {
        model_config.fused_kernels.enabled = enabled;
    }
    if let Some(block) = overrides.block_size {
        model_config.fused_kernels.set_block_sizes(block, block);
    }
    if let Some(use_alibi) = overrides.use_alibi {
        model_config.fused_kernels.set_use_alibi(use_alibi);
        if !use_alibi {
            model_config
                .fused_kernels
                .set_alibi_slopes(vec![0.0; model_config.n_head]);
        }
    }

    model_config
}

fn log_theoretical_profile(config: &BDHConfig, batch: usize, block: usize, backend: &str) {
    let batch = batch as u64;
    let time = block as u64;
    let embed = config.n_embd as u64;
    let latent_per_head = config.latent_per_head() as u64;
    let latent_total = config.latent_total() as u64;
    let heads = config.n_head as u64;
    let bt = batch * time;

    let encoder_matmul = 2 * bt * embed * latent_total;
    let attn_scores = 2 * batch * heads * time * time * latent_per_head;
    let attn_value = 2 * batch * heads * time * time * embed;
    let decoder_matmul = 2 * bt * latent_total * embed;
    let total = encoder_matmul + attn_scores + attn_value + decoder_matmul;

    info!(
        "[train:{backend}] approx forward GFLOPs: total={total_gflops:.2}, encoder={enc:.2}, \
         attn_scores={scores:.2}, attn_value={value:.2}, decoder={dec:.2} (backward ~2x forward)",
        total_gflops = total as f64 / 1e9,
        enc = encoder_matmul as f64 / 1e9,
        scores = attn_scores as f64 / 1e9,
        value = attn_value as f64 / 1e9,
        dec = decoder_matmul as f64 / 1e9,
    );
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{Int, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use burn_dragon_hatchling::{
        ModelState,
        RecurrentStateStore,
        StreamBatchState,
        StreamHandle,
    };
    use std::sync::Mutex;

    type Backend = NdArray<f32>;

    fn assert_close(lhs: Tensor<Backend, 3>, rhs: Tensor<Backend, 3>, atol: f32, rtol: f32) {
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

    fn make_stream_batch(
        pool: Arc<Mutex<RecurrentStateStore<Backend>>>,
        inputs: Tensor<Backend, 2, Int>,
        targets: Tensor<Backend, 2, Int>,
    ) -> SequenceBatch<Backend> {
        let entries = vec![
            Some(StreamHandle {
                id: 0,
                offset: 0,
                slot: 0,
                pool: Arc::clone(&pool),
            }),
            None,
        ];

        let stream_state = StreamBatchState {
            entries,
            max_context: None,
            pool,
        };

        SequenceBatch::with_stream(inputs, targets, stream_state)
    }

    #[test]
    fn pooled_streaming_matches_baseline() {
        let device = <Backend as BackendTrait>::Device::default();
        <Backend as BackendTrait>::seed(&device, 2025);

        let model = BDH::<Backend>::new(BDHConfig::default(), &device);

        let batch = 2;
        let time = 4;

        let mut input_step1 = Vec::new();
        let mut target_step1 = Vec::new();
        let mut input_step2 = Vec::new();
        let mut target_step2 = Vec::new();
        for idx in 0..(batch * time) {
            input_step1.push((idx % 17) as i64);
            target_step1.push(((idx + 1) % 17) as i64);
            input_step2.push(((idx + 3) % 23) as i64);
            target_step2.push(((idx + 4) % 23) as i64);
        }

        let inputs1 = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(input_step1, [batch, time]),
            &device,
        );
        let targets1 = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(target_step1, [batch, time]),
            &device,
        );

        let inputs2 = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(input_step2, [batch, time]),
            &device,
        );
        let targets2 = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(target_step2, [batch, time]),
            &device,
        );

        let pool = Arc::new(Mutex::new(RecurrentStateStore::new(1)));

        let batch1 = make_stream_batch(Arc::clone(&pool), inputs1.clone(), targets1);
        let streamed_logits = forward_with_streaming(&model, &batch1);

        let mut baseline_state = model.init_recurrent_state();
        let logits_stream = model.forward_with_state(
            inputs1.clone().slice_dim(0, 0..1),
            &mut baseline_state,
        );
        let logits_fresh = model.forward(inputs1.clone().slice_dim(0, 1..2));
        let baseline = Tensor::cat(vec![logits_stream.clone(), logits_fresh.clone()], 0);

        assert_close(streamed_logits.clone(), baseline.clone(), 1e-4, 1e-4);

        let batch2 = make_stream_batch(pool, inputs2.clone(), targets2);
        let streamed_logits_step2 = forward_with_streaming(&model, &batch2);

        let logits_stream_step2 = model.forward_with_state(
            inputs2.clone().slice_dim(0, 0..1),
            &mut baseline_state,
        );
        let logits_fresh_step2 = model.forward(inputs2.clone().slice_dim(0, 1..2));
        let baseline_step2 = Tensor::cat(vec![logits_stream_step2, logits_fresh_step2], 0);

        assert_close(streamed_logits_step2, baseline_step2, 1e-4, 1e-4);
    }

    fn make_stream_batch_with_mask(
        pool: Arc<Mutex<RecurrentStateStore<Backend>>>,
        inputs: Tensor<Backend, 2, Int>,
        targets: Tensor<Backend, 2, Int>,
        mask: &[bool],
        max_context: Option<usize>,
    ) -> SequenceBatch<Backend> {
        let mut entries = Vec::with_capacity(mask.len());
        let mut slot = 0;
        for (idx, &is_stream) in mask.iter().enumerate() {
            if is_stream {
                entries.push(Some(StreamHandle {
                    id: idx as u64,
                    offset: idx * inputs.shape().dims::<2>()[1],
                    slot,
                    pool: Arc::clone(&pool),
                }));
                slot += 1;
            } else {
                entries.push(None);
            }
        }

        let stream_state = StreamBatchState {
            entries,
            max_context,
            pool,
        };

        SequenceBatch::with_stream(inputs, targets, stream_state)
    }

    fn baseline_logits(
        model: &BDH<Backend>,
        inputs: Tensor<Backend, 2, Int>,
        mask: &[bool],
        states: &mut [ModelState<Backend>],
        max_context: Option<usize>,
    ) -> Tensor<Backend, 3> {
        let batch = inputs.shape().dims::<2>()[0];
        let mut fresh_indices = Vec::new();
        let mut fresh_slices = Vec::new();
        let mut logits_ordered: Vec<Option<Tensor<Backend, 3>>> = vec![None; batch];

        let mut stream_pos = 0;
        for (idx, is_stream) in mask.iter().enumerate() {
            if *is_stream {
                let slice = inputs.clone().slice_dim(0, idx..idx + 1);
                let state = states
                    .get_mut(stream_pos)
                    .expect("state per stream available");
                let logits = model.forward_with_state(slice, state);
                if let Some(limit) = max_context {
                    state.trim(limit);
                }
                logits_ordered[idx] = Some(logits);
                stream_pos += 1;
            } else {
                fresh_indices.push(idx);
                fresh_slices.push(inputs.clone().slice_dim(0, idx..idx + 1));
            }
        }

        if !fresh_slices.is_empty() {
            let fresh_inputs = if fresh_slices.len() == 1 {
                fresh_slices.pop().expect("single fresh input")
            } else {
                Tensor::cat(fresh_slices, 0)
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

    #[test]
    fn pooled_streaming_handles_mixed_stream_and_fresh_rows() {
        let device = <Backend as BackendTrait>::Device::default();
        <Backend as BackendTrait>::seed(&device, 7);

        let mut model_config = BDHConfig::default();
        model_config.dropout = 0.0;
        let model = BDH::<Backend>::new(model_config, &device);

        let batch = 4;
        let time = 3;
        let mask = [true, true, false, false];

        let mut inputs_step1 = Vec::new();
        let mut targets_step1 = Vec::new();
        let mut inputs_step2 = Vec::new();
        let mut targets_step2 = Vec::new();
        for idx in 0..(batch * time) {
            inputs_step1.push((idx % 31) as i64);
            targets_step1.push(((idx + 1) % 31) as i64);
            inputs_step2.push(((idx + 5) % 37) as i64);
            targets_step2.push(((idx + 6) % 37) as i64);
        }

        let inputs1 = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(inputs_step1, [batch, time]),
            &device,
        );
        let targets1 = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(targets_step1, [batch, time]),
            &device,
        );
        let inputs2 = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(inputs_step2, [batch, time]),
            &device,
        );
        let targets2 = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(targets_step2, [batch, time]),
            &device,
        );

        let pool = Arc::new(Mutex::new(RecurrentStateStore::new(mask.iter().filter(|m| **m).count())));
        let batch1 = make_stream_batch_with_mask(Arc::clone(&pool), inputs1.clone(), targets1, &mask, None);
        let batch2 = make_stream_batch_with_mask(pool, inputs2.clone(), targets2, &mask, None);

        let mut stream_states: Vec<ModelState<Backend>> = (0..mask.iter().filter(|m| **m).count())
            .map(|_| model.init_recurrent_state())
            .collect();

        let baseline1 =
            baseline_logits(&model, inputs1.clone(), &mask, &mut stream_states, None);
        let streamed1 = forward_with_streaming(&model, &batch1);
        assert_close(streamed1, baseline1, 1e-4, 1e-4);

        let baseline2 =
            baseline_logits(&model, inputs2.clone(), &mask, &mut stream_states, None);
        let streamed2 = forward_with_streaming(&model, &batch2);
        assert_close(streamed2, baseline2, 1e-4, 1e-4);
    }

    #[test]
    fn pooled_streaming_respects_max_context() {
        let device = <Backend as BackendTrait>::Device::default();
        <Backend as BackendTrait>::seed(&device, 11);

        let mut model_config = BDHConfig::default();
        model_config.dropout = 0.0;
        let model = BDH::<Backend>::new(model_config, &device);

        let batch = 2;
        let time = 4;
        let mask = [true, true];
        let max_ctx = Some(2);

        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        for idx in 0..(batch * time) {
            inputs.push((idx % 19) as i64);
            targets.push(((idx + 1) % 19) as i64);
        }

        let inputs1 = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(inputs.clone(), [batch, time]),
            &device,
        );
        let targets1 = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(targets.clone(), [batch, time]),
            &device,
        );

        let inputs2 = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(
                inputs.iter().map(|v| (v + 3) % 19).collect::<Vec<_>>(),
                [batch, time],
            ),
            &device,
        );
        let targets2 = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(
                targets.iter().map(|v| (v + 3) % 19).collect::<Vec<_>>(),
                [batch, time],
            ),
            &device,
        );

        let pool = Arc::new(Mutex::new(RecurrentStateStore::new(mask.len())));
        let batch1 =
            make_stream_batch_with_mask(Arc::clone(&pool), inputs1.clone(), targets1, &mask, max_ctx);
        let batch2 = make_stream_batch_with_mask(pool, inputs2.clone(), targets2, &mask, max_ctx);

        let mut stream_states: Vec<ModelState<Backend>> =
            (0..mask.len()).map(|_| model.init_recurrent_state()).collect();

        let baseline1 =
            baseline_logits(&model, inputs1.clone(), &mask, &mut stream_states, max_ctx);
        let streamed1 = forward_with_streaming(&model, &batch1);
        assert_close(streamed1, baseline1, 1e-4, 1e-4);

        let baseline2 =
            baseline_logits(&model, inputs2.clone(), &mask, &mut stream_states, max_ctx);
        let streamed2 = forward_with_streaming(&model, &batch2);
        assert_close(streamed2, baseline2, 1e-4, 1e-4);
    }

    #[test]
    #[should_panic]
    fn streaming_panics_on_non_leading_stream_rows() {
        let device = <Backend as BackendTrait>::Device::default();
        <Backend as BackendTrait>::seed(&device, 13);

        let mut model_config = BDHConfig::default();
        model_config.dropout = 0.0;
        let model = BDH::<Backend>::new(model_config, &device);

        let batch = 3;
        let time = 2;
        let mask = [false, true, true];

        let inputs = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(vec![1, 2, 3, 4, 5, 6], [batch, time]),
            &device,
        );
        let targets = inputs.clone();

        let pool =
            Arc::new(Mutex::new(RecurrentStateStore::new(mask.iter().filter(|m| **m).count())));
        let batch = make_stream_batch_with_mask(pool, inputs, targets, &mask, None);

        let _ = forward_with_streaming(&model, &batch);
    }
}
