use burn::tensor::{
    Int,
    Tensor,
    TensorData,
    backend::Backend,
};

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

#[derive(Debug)]
pub struct RecurrentStateStore<B: Backend> {
    pool: Option<ModelStatePool<B>>,
    max_streams: usize,
}

impl<B: Backend> RecurrentStateStore<B> {
    pub fn new(max_streams: usize) -> Self {
        Self {
            pool: None,
            max_streams,
        }
    }

    pub fn max_streams(&self) -> usize {
        self.max_streams
    }

    pub fn ensure_pool<F>(&mut self, init: F) -> &mut ModelStatePool<B>
    where
        F: FnOnce(usize) -> ModelStatePool<B>,
    {
        if self.pool.is_none() {
            self.pool = Some(init(self.max_streams));
        }
        self.pool
            .as_mut()
            .expect("state pool initialized")
    }

    pub fn pool(&self) -> Option<&ModelStatePool<B>> {
        self.pool.as_ref()
    }

    pub fn reset_slot(&mut self, slot: usize) {
        if let Some(pool) = &mut self.pool {
            pool.reset_slot(slot);
        }
    }
}

#[derive(Debug)]
pub struct RecurrentStateView<B: Backend> {
    pub positions: Tensor<B, 1, Int>,
    pub positions_host: Vec<usize>,
    pub layers: Vec<Tensor<B, 4>>,
}

#[derive(Debug)]
pub struct ModelStatePool<B: Backend> {
    layers: Vec<Tensor<B, 4>>,
    steps: Vec<usize>,
    max_streams: usize,
    heads: usize,
    latent: usize,
    dim: usize,
    device: B::Device,
}

impl<B: Backend> ModelStatePool<B> {
    pub fn new(
        num_layers: usize,
        max_streams: usize,
        heads: usize,
        latent: usize,
        dim: usize,
        device: &B::Device,
    ) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(Tensor::<B, 4>::zeros(
                [max_streams, heads, latent, dim],
                device,
            ));
        }
        Self {
            layers,
            steps: vec![0; max_streams],
            max_streams,
            heads,
            latent,
            dim,
            device: device.clone(),
        }
    }

    pub fn prefix_view(&self, count: usize) -> RecurrentStateView<B> {
        let count = count.min(self.max_streams);
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(layer.clone().slice_dim(0, 0..count).detach());
        }
        let positions_host: Vec<usize> = self.steps.iter().take(count).cloned().collect();
        let positions = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                positions_host
                    .iter()
                    .map(|value| *value as i64)
                    .collect::<Vec<_>>(),
                [count],
            ),
            &self.device,
        );
        RecurrentStateView {
            positions,
            positions_host,
            layers,
        }
    }

    pub fn write_prefix(
        &mut self,
        count: usize,
        updated: &[Tensor<B, 4>],
        max_context: Option<usize>,
        advance: usize,
    ) {
        let count = count.min(self.max_streams);
        if count == 0 {
            return;
        }

        let mut reset_mask = vec![1.0_f32; count];
        if let Some(limit) = max_context {
            for idx in 0..count {
                let new_steps = self.steps.get(idx).copied().unwrap_or(0).saturating_add(advance);
                if new_steps > limit {
                    reset_mask[idx] = 0.0;
                    self.steps[idx] = 0;
                } else {
                    self.steps[idx] = new_steps;
                }
            }
        } else {
            for idx in 0..count {
                let new_steps = self.steps.get(idx).copied().unwrap_or(0).saturating_add(advance);
                self.steps[idx] = new_steps;
            }
        }

        let mask_tensor = Tensor::<B, 1>::from_floats(reset_mask.as_slice(), &self.device)
            .reshape([count, 1, 1, 1]);

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let mut updated_rows = updated
                .get(layer_idx)
                .cloned()
                .expect("updated layer state matches layout");
            // Reuse cached states without carrying the previous graph forward.
            updated_rows = updated_rows.detach();
            if reset_mask.iter().any(|value| *value == 0.0) {
                updated_rows = updated_rows * mask_tensor.clone();
            }

            if count < self.max_streams {
                let tail = layer.clone().slice_dim(0, count..self.max_streams);
                *layer = Tensor::cat(vec![updated_rows, tail], 0);
            } else {
                *layer = updated_rows;
            }
        }
    }

    pub fn reset_slot(&mut self, slot: usize) {
        if slot >= self.max_streams {
            return;
        }
        self.steps[slot] = 0;
        let zero = Tensor::<B, 4>::zeros([1, self.heads, self.latent, self.dim], &self.device);
        for layer in &mut self.layers {
            if slot + 1 >= self.max_streams {
                *layer = Tensor::cat(
                    vec![layer.clone().slice_dim(0, 0..slot), zero.clone()],
                    0,
                );
            } else if slot == 0 {
                let tail = layer.clone().slice_dim(0, 1..self.max_streams);
                *layer = Tensor::cat(vec![zero.clone(), tail], 0);
            } else {
                let head = layer.clone().slice_dim(0, 0..slot);
                let tail = layer.clone().slice_dim(0, slot + 1..self.max_streams);
                *layer = Tensor::cat(vec![head, zero.clone(), tail], 0);
            }
        }
    }
}
