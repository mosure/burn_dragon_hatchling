use std::f32::consts::PI;

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, activation};

use super::config::FusedKernelConfig;
use crate::kernel::{BlockPattern2d, linear_attention};
use crate::positional::RotaryEmbedding;

#[derive(Default, Debug, Clone)]
pub struct AttentionCache<B: Backend> {
    q_rot: Option<Tensor<B, 4>>,
    value: Option<Tensor<B, 4>>,
    #[cfg(feature = "viz")]
    last_attention: Option<Tensor<B, 3>>,
}

impl<B: Backend> AttentionCache<B> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.q_rot
            .as_ref()
            .map(|tensor| tensor.shape().dims::<4>()[2])
            .unwrap_or(0)
    }

    pub fn reset(&mut self) {
        self.q_rot = None;
        self.value = None;
        #[cfg(feature = "viz")]
        {
            self.last_attention = None;
        }
    }

    pub fn append(&mut self, q_rot: Tensor<B, 4>, value: Tensor<B, 4>) {
        self.q_rot = Some(match self.q_rot.take() {
            Some(prev) => Tensor::cat(vec![prev, q_rot], 2),
            None => q_rot,
        });
        self.value = Some(match self.value.take() {
            Some(prev) => Tensor::cat(vec![prev, value], 2),
            None => value,
        });
        #[cfg(feature = "viz")]
        {
            self.last_attention = None;
        }
    }

    pub fn retain_last(&mut self, max_len: usize) {
        if max_len == 0 {
            self.reset();
            return;
        }

        if let Some(existing) = self.q_rot.take() {
            let time = existing.shape().dims::<4>()[2];
            let trimmed = if time > max_len {
                let start = time - max_len;
                existing.slice_dim(2, start..time)
            } else {
                existing
            };
            self.q_rot = Some(trimmed);
        }

        if let Some(existing) = self.value.take() {
            let time = existing.shape().dims::<4>()[2];
            let trimmed = if time > max_len {
                let start = time - max_len;
                existing.slice_dim(2, start..time)
            } else {
                existing
            };
            self.value = Some(trimmed);
        }
        #[cfg(feature = "viz")]
        {
            self.last_attention = None;
        }
    }

    #[cfg(feature = "viz")]
    pub fn set_last_attention(&mut self, row: Tensor<B, 3>) {
        self.last_attention = Some(row);
    }

    #[cfg(feature = "viz")]
    pub fn last_attention(&self) -> Option<Tensor<B, 3>> {
        self.last_attention.clone()
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    freqs: Tensor<B, 4>,
    n_head: usize,
    fused: bool,
    block_pattern: BlockPattern2d,
    use_alibi: bool,
    alibi_slopes: Tensor<B, 1>,
    rotary_embedding: RotaryEmbedding,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        latent: usize,
        n_head: usize,
        device: &B::Device,
        kernel: &FusedKernelConfig,
    ) -> Self {
        let freqs = Self::build_freqs(latent, kernel.rope_theta, kernel.rotary_embedding, device);
        let (use_alibi, alibi_slopes) = if kernel.enabled && kernel.use_alibi {
            let slopes = kernel
                .alibi_slopes
                .clone()
                .unwrap_or_else(|| linear_attention::default_alibi_slopes(n_head));
            (true, Tensor::<B, 1>::from_floats(slopes.as_slice(), device))
        } else {
            (false, Tensor::<B, 1>::zeros([n_head], device))
        };

        Self {
            freqs,
            n_head,
            fused: kernel.enabled,
            block_pattern: kernel.block_sparse.time.clone(),
            use_alibi,
            alibi_slopes,
            rotary_embedding: kernel.rotary_embedding,
        }
    }

    pub fn forward(&self, query: Tensor<B, 4>, value: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.fused {
            return linear_attention::fused_state_aligned(
                query,
                value,
                self.freqs.clone(),
                self.use_alibi.then_some(self.alibi_slopes.clone()),
                &self.block_pattern,
                self.rotary_embedding,
            );
        }

        let q_rot = self.rotate(query, 0);
        let k_rot = q_rot.clone();

        let scores = q_rot.matmul(k_rot.swap_dims(2, 3)).tril(-1);
        let value = value.repeat_dim(1, self.n_head);

        scores.matmul(value)
    }

    pub fn forward_cached(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        cache: &mut AttentionCache<B>,
    ) -> Tensor<B, 4> {
        let time_new = query.shape().dims::<4>()[2];
        let position = cache.len();

        let q_rot = self.rotate(query, position);
        let k_rot = q_rot.clone();
        let value_rep = value.repeat_dim(1, self.n_head);

        #[cfg(feature = "viz")]
        let mut attn_row: Option<Tensor<B, 3>> = None;

        let context = if let (Some(prev_q), Some(prev_v)) = (&cache.q_rot, &cache.value) {
            let scores_prev = q_rot.clone().matmul(prev_q.clone().swap_dims(2, 3));
            let mut scores_self = q_rot.clone().matmul(k_rot.clone().swap_dims(2, 3)).tril(-1);

            let scores_prev = if self.use_alibi {
                let device = q_rot.device();
                let slopes = self.alibi_slopes.clone().reshape([1, self.n_head, 1, 1]);
                let prev_len = position;

                let pos_row = Tensor::<B, 1, Int>::arange(
                    position as i64..(position + time_new) as i64,
                    &device,
                )
                .float()
                .reshape([1, 1, time_new, 1]);

                let pos_prev = Tensor::<B, 1, Int>::arange(0..prev_len as i64, &device)
                    .float()
                    .reshape([1, 1, 1, prev_len]);
                let alibi_prev = slopes.clone() * (pos_prev - pos_row.clone());

                let pos_new = Tensor::<B, 1, Int>::arange(
                    position as i64..(position + time_new) as i64,
                    &device,
                )
                .float()
                .reshape([1, 1, 1, time_new]);
                let alibi_self = slopes * (pos_new - pos_row).tril(-1);

                scores_self = scores_self + alibi_self;
                scores_prev + alibi_prev
            } else {
                scores_prev
            };

            let scores = Tensor::cat(vec![scores_prev, scores_self], 3);

            #[cfg(feature = "viz")]
            {
                let dims = scores.shape().dims::<4>();
                if dims[2] > 0 {
                    let row = scores
                        .clone()
                        .slice_dim(2, (dims[2] - 1)..dims[2])
                        .reshape([dims[0], dims[1], dims[3]]);
                    attn_row = Some(row);
                }
            }
            let value_all = Tensor::cat(vec![prev_v.clone(), value_rep.clone()], 2);
            scores.matmul(value_all)
        } else {
            let mut scores = q_rot.clone().matmul(k_rot.clone().swap_dims(2, 3)).tril(-1);
            if self.use_alibi {
                let device = q_rot.device();
                let slopes = self.alibi_slopes.clone().reshape([1, self.n_head, 1, 1]);
                let pos_row = Tensor::<B, 1, Int>::arange(
                    position as i64..(position + time_new) as i64,
                    &device,
                )
                .float()
                .reshape([1, 1, time_new, 1]);
                let pos_new = Tensor::<B, 1, Int>::arange(
                    position as i64..(position + time_new) as i64,
                    &device,
                )
                .float()
                .reshape([1, 1, 1, time_new]);
                let alibi = slopes * (pos_new - pos_row).tril(-1);
                scores = scores + alibi;
            }
            #[cfg(feature = "viz")]
            {
                let dims = scores.shape().dims::<4>();
                if dims[2] > 0 {
                    let row = scores
                        .clone()
                        .slice_dim(2, (dims[2] - 1)..dims[2])
                        .reshape([dims[0], dims[1], dims[3]]);
                    attn_row = Some(row);
                }
            }
            scores.matmul(value_rep.clone())
        };

        cache.append(k_rot.clone(), value_rep.clone());

        #[cfg(feature = "viz")]
        if let Some(row) = attn_row {
            cache.set_last_attention(row);
        }

        context
    }

    fn rope(&self, phases: Tensor<B, 4>, values: Tensor<B, 4>) -> Tensor<B, 4> {
        let cos = phases.clone().cos();
        let sin = phases.sin();

        let [b, h, t, n] = values.shape().dims();
        let pairs = values.clone().reshape([b, h, t, n / 2, 2]);

        let even = pairs
            .clone()
            .slice_dim(4, 0..1)
            .squeeze_dim::<4>(4);
        let odd = pairs.slice_dim(4, 1..2).squeeze_dim::<4>(4);

        let rotated = Tensor::stack::<5>(vec![odd.clone().neg(), even], 4).reshape([b, h, t, n]);

        values * cos + rotated * sin
    }

    fn pope(&self, phases: Tensor<B, 4>, values: Tensor<B, 4>) -> Tensor<B, 4> {
        let magnitude = activation::softplus(values, 1.0);
        let cos = phases.clone().cos();
        let sin = phases.sin();
        let real = magnitude.clone() * cos;
        let imag = magnitude * sin;
        Tensor::cat(vec![real, imag], 3)
    }

    fn rotate(&self, values: Tensor<B, 4>, start: usize) -> Tensor<B, 4> {
        let time = values.shape().dims::<4>()[2];
        let device = values.device();
        let positions = Tensor::<B, 1, Int>::arange(start as i64..(start + time) as i64, &device)
            .float()
            .reshape([1, 1, time, 1]);

        let raw = positions * self.freqs.clone();
        let phases = (raw.clone() - raw.floor()) * (2.0 * PI);
        match self.rotary_embedding {
            RotaryEmbedding::Rope => self.rope(phases, values),
            RotaryEmbedding::Pope => self.pope(phases, values),
        }
    }

    fn build_freqs(
        latent: usize,
        theta: f32,
        rotary_embedding: RotaryEmbedding,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let mut data = Vec::with_capacity(latent);
        for idx in 0..latent {
            let exponent = match rotary_embedding {
                RotaryEmbedding::Rope => ((idx as f32 / 2.0).floor() * 2.0) / latent as f32,
                RotaryEmbedding::Pope => idx as f32 / latent as f32,
            };
            let value = 1.0 / theta.powf(exponent) / (2.0 * PI);
            data.push(value);
        }
        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([1, 1, 1, latent])
    }
}
