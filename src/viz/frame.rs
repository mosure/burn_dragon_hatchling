use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

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
    pub overview_activity: Tensor<B, 3>,
    pub overview_writes: Tensor<B, 3>,
    pub units_activity: Tensor<B, 3>,
    pub units_writes: Tensor<B, 3>,
    pub cursor: usize,
    pub token_index: usize,
}
