use burn::tensor::backend::Backend;

use super::noiser::{EggrollConfig, EggrollParamSpec};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BdhEsTarget {
    Embedding,
    Decoder,
    LmHead,
    Encoder,
    EncoderV,
}

#[derive(Clone, Debug)]
pub struct BdhEsTargetConfig {
    pub enabled: bool,
    pub rank: usize,
    pub sigma_scale: f32,
}

#[derive(Clone, Debug)]
pub struct BdhEsConfig {
    pub eggroll: EggrollConfig,
    pub embedding: BdhEsTargetConfig,
    pub decoder: BdhEsTargetConfig,
    pub lm_head: BdhEsTargetConfig,
    pub encoder: BdhEsTargetConfig,
    pub encoder_v: BdhEsTargetConfig,
}

impl Default for BdhEsConfig {
    fn default() -> Self {
        let default_rank = EggrollConfig::default().rank;
        let on = |rank| BdhEsTargetConfig {
            enabled: true,
            rank,
            sigma_scale: 1.0,
        };
        let off = |rank| BdhEsTargetConfig {
            enabled: false,
            rank,
            sigma_scale: 0.0,
        };
        Self {
            eggroll: EggrollConfig::default(),
            embedding: on(default_rank),
            decoder: on(default_rank),
            lm_head: on(default_rank),
            encoder: on(default_rank),
            encoder_v: on(default_rank),
        }
    }
}

pub fn bdh_param_specs<B: Backend>(
    model: &crate::model::BDH<B>,
    cfg: &BdhEsConfig,
) -> Vec<EggrollParamSpec> {
    model.es_param_specs(cfg)
}
