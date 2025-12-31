use serde::{Deserialize, Serialize};

use crate::positional::RotaryEmbedding;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TrainingHyperparameters {
    pub block_size: usize,
    pub batch_size: usize,
    pub max_iters: usize,
    pub log_frequency: usize,
    #[serde(default)]
    pub fast_train: bool,
    #[serde(default = "default_context_strategy")]
    pub context_strategy: ContextStrategyConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct GenerationConfig {
    pub prompt: String,
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default = "default_context_strategy")]
    pub context_strategy: ContextStrategyConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContextStrategyConfig {
    #[default]
    Infinite,
    Sliding {
        window: usize,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
pub struct ModelOverrides {
    pub n_layer: Option<usize>,
    pub n_embd: Option<usize>,
    pub n_head: Option<usize>,
    pub mlp_internal_dim_multiplier: Option<usize>,
    pub dropout: Option<f64>,
    pub fused_kernels: Option<bool>,
    pub block_size: Option<usize>,
    pub rotary_embedding: Option<RotaryEmbedding>,
}

fn default_context_strategy() -> ContextStrategyConfig {
    ContextStrategyConfig::Infinite
}

fn default_temperature() -> f32 {
    1.0
}
