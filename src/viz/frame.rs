use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub const LAYER_GAP: usize = 20;
pub const VIZ_MAX_RES: usize = 8192;

pub fn clamp_history(history: usize) -> usize {
    history.max(1).min(VIZ_MAX_RES)
}

pub fn clamp_layers(layers: usize, latent_total: usize) -> usize {
    let layers = layers.max(1);
    let latent_total = latent_total.max(1);
    let step = latent_total.saturating_add(LAYER_GAP).max(1);
    let max_layers = VIZ_MAX_RES.saturating_add(LAYER_GAP) / step;
    layers.min(max_layers.max(1))
}

pub fn units_height(layers: usize, latent_total: usize) -> usize {
    let layers = layers.max(1);
    let latent_total = latent_total.max(1);
    latent_total
        .saturating_mul(layers)
        .saturating_add(LAYER_GAP.saturating_mul(layers.saturating_sub(1)))
        .max(1)
}

#[derive(Clone, Debug)]
pub struct VizConfig {
    pub history: usize,
    pub layer_focus: usize,
    pub stride_tokens: usize,
    pub gain_x: f32,
    pub gain_xy: f32,
}

impl Default for VizConfig {
    fn default() -> Self {
        Self {
            history: 256,
            layer_focus: 0,
            stride_tokens: 1,
            gain_x: 1.0,
            gain_xy: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VizFrame<B: Backend> {
    pub units_x: Tensor<B, 3>,
    pub units_y: Tensor<B, 3>,
    pub units_xy: Tensor<B, 3>,
    pub units_rho: Tensor<B, 3>,
    pub cursor: usize,
    pub token_index: usize,
}
