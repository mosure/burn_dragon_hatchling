#![recursion_limit = "256"]

pub mod config;
pub mod dataset;
pub mod generation;
pub mod kernel;
pub mod model;
pub mod tokenizer;
pub mod viz;
pub mod wgpu;
pub mod eggroll;

pub use config::{
    ContextStrategyConfig, DatasetConfig, GenerationConfig, LearningRateScheduleConfig,
    ModelOverrides, OptimizerConfig, TrainingConfig, TrainingHyperparameters, TrainingMode,
    load_training_config,
};
pub use dataset::{
    Dataset, DatasetSplit, HuggingFaceDataset, RandomDataLoader, SequenceBatch, ShakespeareBatch,
    ShakespeareDataset, ShakespeareRandomDataLoader, ShakespeareSplit, StreamBatchState,
    StreamHandle, TokenSequenceDataset, build_dataset,
};
pub use generation::{
    ContextStrategy, GenerationSettings, generate_text, generate_tokens, prefill_state,
    resolve_context_strategy, sample_next_token,
};
pub use kernel::{BlockPattern1d, BlockPattern2d, BlockSparseConfig};
pub use model::{BDH, BDHConfig, ModelState, language_model_loss};
pub use tokenizer::char_vocab::CharVocab;
pub use eggroll::{
    EggrollConfig,
    EggrollKey,
    EggrollObjective,
    EggrollTrainer,
    EggrollParamSpec,
};
pub use eggroll::bdh_integration::{BdhEsConfig, BdhEsTarget, BdhEsTargetConfig, bdh_param_specs};
