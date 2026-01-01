use crate::{BDHConfig, ModelOverrides};

/// Build a model configuration by applying training overrides.
pub fn build_model_config(overrides: &ModelOverrides, training_block_size: usize) -> BDHConfig {
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
    if let Some(relu_threshold) = overrides.relu_threshold {
        model_config.fused_kernels.relu_threshold = relu_threshold;
    }
    if let Some(dropout) = overrides.dropout {
        model_config.dropout = dropout;
    }
    if let Some(enabled) = overrides.fused_kernels {
        model_config.fused_kernels.enabled = enabled;
    }
    let block = overrides
        .block_size
        .unwrap_or(training_block_size)
        .max(1);
    model_config.fused_kernels.set_block_sizes(block, block);
    if let Some(rotary_embedding) = overrides.rotary_embedding {
        model_config
            .fused_kernels
            .set_rotary_embedding(rotary_embedding);
    }

    model_config
}
