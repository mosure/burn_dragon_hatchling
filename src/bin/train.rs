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
use burn::tensor::{Tensor, activation};
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn_autodiff::Autodiff;
use burn_train::metric::{Adaptor, ItemLazy, LearningRateMetric, LossInput, LossMetric, MetricEntry, NumericEntry};
use burn_train::renderer::{MetricState, MetricsRendererTraining, TrainingProgress};
use burn_train::renderer::tui::TuiMetricsRenderer;
use burn_train::Interrupter;
use burn_train::{
    LearnerBuilder,
    LearningStrategy,
    TrainingResult,
    TrainOutput,
    TrainStep,
    ValidStep,
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

    let mut logits_by_index: Vec<(usize, Tensor<B, 3>)> =
        Vec::with_capacity(batch.inputs.shape().dims::<2>()[0]);
    let mut fresh_indices = Vec::new();

    for (idx, entry) in stream.entries.iter().enumerate() {
        if let Some(handle) = entry {
            let mut guard = handle.state.lock().expect("lock stream state");
            if guard.is_none() {
                *guard = Some(model.init_compressed_state());
            }
            let state = guard.as_mut().expect("stream state initialized");
            let input_slice = batch.inputs.clone().slice_dim(0, idx..idx + 1);
            let logits = model.forward_with_state(input_slice, state);
            logits_by_index.push((idx, logits));
        } else {
            fresh_indices.push(idx);
        }
    }

    if !fresh_indices.is_empty() {
        let mut fresh_inputs = Vec::with_capacity(fresh_indices.len());
        for &idx in &fresh_indices {
            fresh_inputs.push(batch.inputs.clone().slice_dim(0, idx..idx + 1));
        }
        let fresh_inputs = if fresh_inputs.len() == 1 {
            fresh_inputs.pop().expect("single fresh slice")
        } else {
            Tensor::cat(fresh_inputs, 0)
        };
        let logits_fresh = model.forward(fresh_inputs);
        for (pos, &idx) in fresh_indices.iter().enumerate() {
            let slice = logits_fresh.clone().slice_dim(0, pos..pos + 1);
            logits_by_index.push((idx, slice));
        }
    }

    logits_by_index.sort_by_key(|(idx, _)| *idx);
    let ordered: Vec<_> = logits_by_index.into_iter().map(|(_, logits)| logits).collect();
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

    fn evaluate_population(
        &mut self,
        logits: &Tensor<B, 4>,
        batch: &Self::Batch,
    ) -> Vec<f32> {
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
                    TrainingMode::Eggroll => {
                        train_backend_eggroll::<Cuda<f32>, _>(
                            &config,
                            dataset,
                            "cuda",
                            |_| {},
                            args.train.checkpoint.as_ref(),
                        )
                    }
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
    let mut train_loader: Box<dyn DataLoaderIterator<SequenceBatch<B>>> =
        train_loader_base.iter();

    let eggroll_cfg = resolve_bdh_eggroll(config);
    let mut model = BDH::<B>::new(model_config.clone(), &device);
    if let Some(path) = checkpoint {
        info!("Loading checkpoint from {}", path.display());
        let record = BinFileRecorder::<FullPrecisionSettings>::new()
            .load(path.to_path_buf(), &device)?;
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
    let run_dir = PathBuf::from("runs")
        .join(backend_name)
        .join("eggroll");
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
                Arc::new("fitness_mean".to_string()),
                format!("{:.4}", mean),
                format!("{}", mean),
            );
            let metric_std = MetricEntry::new(
                Arc::new("fitness_std".to_string()),
                format!("{:.4}", std_dev),
                format!("{}", std_dev),
            );
            let metric_step = MetricEntry::new(
                Arc::new("step_ms".to_string()),
                format!("{:.1}", step_ms),
                format!("{}", step_ms),
            );
            let metric_pop = MetricEntry::new(
                Arc::new("pop_size".to_string()),
                trainer.state.config.pop_size.to_string(),
                trainer.state.config.pop_size.to_string(),
            );
            let metric_sigma = MetricEntry::new(
                Arc::new("sigma".to_string()),
                format!("{:.4}", trainer.state.config.sigma),
                format!("{}", trainer.state.config.sigma),
            );
            let metric_lr = MetricEntry::new(
                Arc::new("lr".to_string()),
                format!("{:.5}", trainer.state.config.lr),
                format!("{}", trainer.state.config.lr),
            );
            tui.update_train(MetricState::Numeric(metric, NumericEntry::Value(mean as f64)));
            tui.update_train(MetricState::Numeric(metric_std, NumericEntry::Value(std_dev as f64)));
            tui.update_train(MetricState::Numeric(metric_step, NumericEntry::Value(step_ms)));
            tui.update_train(MetricState::Numeric(metric_pop, NumericEntry::Value(trainer.state.config.pop_size as f64)));
            tui.update_train(MetricState::Numeric(metric_sigma, NumericEntry::Value(trainer.state.config.sigma as f64)));
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
        .learning_strategy(LearningStrategy::SingleDevice(env.device.clone()))
        .with_file_checkpointer(BinFileRecorder::<FullPrecisionSettings>::new())
        .metric_train_numeric(LossMetric::<ValidBackend<B>>::new())
        .metric_valid_numeric(LossMetric::<ValidBackend<B>>::new())
        .metric_train_numeric(LearningRateMetric::new())
        .summary();

    let learner = builder.build(model, optimizer, scheduler);

    let TrainingResult { model, .. } = learner.fit(
        Arc::clone(&env.train_loader),
        Arc::clone(&env.valid_loader),
    );

    log_theoretical_profile(
        env.model_config,
        env.training.batch_size,
        env.training.block_size,
        env.backend_name,
    );

    Ok(model)
}

fn resolve_training_context_window(
    training: &TrainingHyperparameters,
) -> Option<usize> {
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
