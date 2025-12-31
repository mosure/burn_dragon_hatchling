use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Debug, Clone)]
pub struct LayerState<B: Backend> {
    pub rho: Option<Tensor<B, 4>>,
    #[cfg(feature = "viz")]
    pub viz: Option<LayerVizState<B>>,
}

#[derive(Debug, Clone)]
pub struct ModelState<B: Backend> {
    pub layers: Vec<LayerState<B>>,
    pub position: usize,
}

#[cfg(feature = "viz")]
#[derive(Debug, Clone)]
pub struct LayerVizState<B: Backend> {
    pub attn_rows: Vec<Tensor<B, 1>>,
    pub neurons: Tensor<B, 1>,
    pub synapses: Tensor<B, 2>,
}

impl<B: Backend> ModelState<B> {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerState {
                    rho: None,
                    #[cfg(feature = "viz")]
                    viz: None,
                })
                .collect(),
            position: 0,
        }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.rho = None;
        }
        self.position = 0;
    }

    pub fn len(&self) -> usize {
        self.position
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn trim(&mut self, max_len: usize) {
        let _ = max_len;
    }

    #[cfg(feature = "viz")]
    pub fn take_viz(&mut self) -> Vec<Option<LayerVizState<B>>> {
        self.layers
            .iter_mut()
            .map(|layer| layer.viz.take())
            .collect()
    }

    #[cfg(feature = "viz")]
    pub fn clear_viz(&mut self) {
        for layer in &mut self.layers {
            layer.viz = None;
        }
    }
}

#[cfg(feature = "viz")]
impl<B: Backend> LayerState<B> {
    pub fn take_viz(&mut self) -> Option<LayerVizState<B>> {
        self.viz.take()
    }
}
