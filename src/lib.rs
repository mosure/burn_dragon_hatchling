#![recursion_limit = "256"]

pub mod config;
#[cfg(feature = "train")]
pub mod dataset;
pub mod generation;
pub mod inference;
pub mod kernel;
pub mod model;
pub mod positional;
pub mod tokenizer;
pub mod viz;
#[cfg(feature = "cli")]
pub mod wgpu;
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub mod web;

pub use config::{ContextStrategyConfig, GenerationConfig, ModelOverrides, TrainingHyperparameters};
#[cfg(feature = "train")]
pub use config::{
    DatasetConfig, DatasetSourceConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    LearningRateScheduleConfig, OptimizerConfig, TrainingConfig, load_training_config,
};
#[cfg(feature = "train")]
pub use dataset::{
    Dataset, DatasetSplit, HuggingFaceDataset, RandomDataLoader, SequenceBatch, ShakespeareBatch,
    ShakespeareDataset, ShakespeareRandomDataLoader, ShakespeareSplit, TokenSequenceDataset,
    build_dataset,
};
pub use generation::{
    ContextStrategy, GenerationSettings, generate_text, generate_tokens, prefill_state,
    resolve_context_strategy, sample_next_token,
};
pub use inference::build_model_config;
pub use kernel::{BlockPattern1d, BlockPattern2d, BlockSparseConfig};
pub use model::{BDH, BDHConfig, ModelState, language_model_loss};
pub use positional::RotaryEmbedding;
pub use tokenizer::char_vocab::CharVocab;
