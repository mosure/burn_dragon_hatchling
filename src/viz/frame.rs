use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub const LAYER_GAP: usize = 20;

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
