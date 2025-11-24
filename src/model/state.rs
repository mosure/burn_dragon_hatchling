#[cfg(feature = "viz")]
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use super::attention::AttentionCache;

#[derive(Debug, Clone)]
pub struct LayerState<B: Backend> {
    pub attention: AttentionCache<B>,
    #[cfg(feature = "viz")]
    pub viz: Option<LayerVizState<B>>,
}

#[derive(Debug, Clone)]
pub struct ModelState<B: Backend> {
    pub layers: Vec<LayerState<B>>,
    pub position: usize,
    pub compressed: bool,
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
                    attention: AttentionCache::new(),
                    #[cfg(feature = "viz")]
                    viz: None,
                })
                .collect(),
            position: 0,
            compressed: false,
        }
    }

    pub fn new_compressed(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerState {
                    attention: AttentionCache::new(),
                    #[cfg(feature = "viz")]
                    viz: None,
                })
                .collect(),
            position: 0,
            compressed: true,
        }
    }

    pub fn new_recurrent(num_layers: usize) -> Self {
        Self::new_compressed(num_layers)
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.attention.reset();
        }
        self.position = 0;
    }

    pub fn len(&self) -> usize {
        self.layers
            .first()
            .map(|layer| layer.attention.len())
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn trim(&mut self, max_len: usize) {
        for layer in &mut self.layers {
            layer.attention.retain_last(max_len);
        }
        self.position = self.len().min(max_len);
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

    pub fn try_stack(mut states: Vec<Self>) -> Option<Self> {
        let position = states.first().map(|s| s.position)?;
        let compressed = states.first().map(|s| s.compressed)?;
        let layer_len = states.first().map(|s| s.layers.len())?;

        if states.iter().any(|s| s.compressed != compressed) {
            return None;
        }

        if states.len() == 1 {
            return states.pop();
        }

        if states.iter().any(|s| s.layers.len() != layer_len) {
            return None;
        }

        let mut layers = Vec::with_capacity(layer_len);
        for layer_idx in 0..layer_len {
            let mut caches = Vec::with_capacity(states.len());
            for state in states.iter_mut() {
                let cache = state.layers.get_mut(layer_idx)?.attention.clone();
                caches.push(cache);
            }
            let stacked = AttentionCache::try_stack(caches)?;
            layers.push(LayerState {
                attention: stacked,
                #[cfg(feature = "viz")]
                viz: None,
            });
        }

        Some(Self {
            layers,
            position,
            compressed,
        })
    }

    pub fn split(self, parts: usize) -> Vec<Self> {
        if parts == 0 {
            return Vec::new();
        }

        let mut per_part = vec![
            Self {
                layers: Vec::with_capacity(self.layers.len()),
                position: self.position,
                compressed: self.compressed,
            };
            parts
        ];

        for layer in self.layers {
            let split = layer.attention.split(parts);
            for (idx, cache) in split.into_iter().enumerate() {
                per_part[idx].layers.push(LayerState {
                    attention: cache,
                    #[cfg(feature = "viz")]
                    viz: None,
                });
            }
        }

        per_part
    }
}

#[cfg(feature = "viz")]
impl<B: Backend> LayerState<B> {
    pub fn take_viz(&mut self) -> Option<LayerVizState<B>> {
        self.viz.take()
    }
}
