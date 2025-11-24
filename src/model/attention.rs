use std::f32::consts::PI;

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use super::config::FusedKernelConfig;
use crate::kernel::{BlockPattern2d, linear_attention};

#[derive(Default, Debug, Clone)]
pub struct AttentionCache<B: Backend> {
    q_rot: Option<Tensor<B, 4>>,
    value: Option<Tensor<B, 4>>,
    linear_kv: Option<Tensor<B, 4>>,
    steps: usize,
    #[cfg(feature = "viz")]
    last_attention: Option<Tensor<B, 3>>,
}

impl<B: Backend> AttentionCache<B> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        if self.steps > 0 {
            return self.steps;
        }
        self.q_rot
            .as_ref()
            .map(|tensor| tensor.shape().dims::<4>()[2])
            .unwrap_or(0)
    }

    pub fn reset(&mut self) {
        self.q_rot = None;
        self.value = None;
        self.linear_kv = None;
        self.steps = 0;
        #[cfg(feature = "viz")]
        {
            self.last_attention = None;
        }
    }

    pub fn append(&mut self, q_rot: Tensor<B, 4>, value: Tensor<B, 4>) {
        let time = q_rot.shape().dims::<4>()[2];
        self.q_rot = Some(match self.q_rot.take() {
            Some(prev) => Tensor::cat(vec![prev, q_rot], 2),
            None => q_rot,
        });
        self.value = Some(match self.value.take() {
            Some(prev) => Tensor::cat(vec![prev, value], 2),
            None => value,
        });
        self.steps = self.steps.saturating_add(time);
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

        // Recurrent / linear attention state cannot be partially trimmed;
        // drop it entirely if the requested window is smaller than the
        // number of cached steps to avoid inconsistent positional phases.
        if self.linear_kv.is_some() && self.q_rot.is_none() && self.value.is_none() {
            if self.steps > max_len {
                self.linear_kv = None;
                self.steps = 0;
            }
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

        if max_len == 0 {
            self.linear_kv = None;
        }

        self.steps = self
            .q_rot
            .as_ref()
            .map(|tensor| tensor.shape().dims::<4>()[2])
            .unwrap_or(0);
    }

    pub fn clear_linear(&mut self) {
        self.linear_kv = None;
    }

    #[cfg(feature = "viz")]
    pub fn set_last_attention(&mut self, row: Tensor<B, 3>) {
        self.last_attention = Some(row);
    }

    #[cfg(feature = "viz")]
    pub fn last_attention(&self) -> Option<Tensor<B, 3>> {
        self.last_attention.clone()
    }

    pub fn try_stack(caches: &[Self]) -> Option<Self> {
        let first = caches.first()?;
        let steps = first.steps;

        if caches.iter().any(|cache| cache.steps != steps) {
            return None;
        }

        if caches
            .iter()
            .any(|cache| cache.q_rot.is_some() || cache.value.is_some())
        {
            return None;
        }

        let mut linear_parts: Vec<Tensor<B, 4>> = Vec::new();
        for cache in caches {
            if let Some(kv) = cache.linear_kv.clone() {
                linear_parts.push(kv);
            } else if !linear_parts.is_empty() {
                return None;
            }
        }

        let linear_kv = if linear_parts.is_empty() {
            None
        } else {
            let dims_first = linear_parts
                .first()
                .map(|kv| kv.shape().dims::<4>())
                .unwrap_or([0, 0, 0, 0]);
            let consistent = linear_parts
                .iter()
                .all(|kv| kv.shape().dims::<4>()[1..] == dims_first[1..]);
            if !consistent {
                return None;
            }
            Some(Tensor::cat(linear_parts, 0))
        };

        Some(Self {
            q_rot: None,
            value: None,
            linear_kv,
            steps,
            #[cfg(feature = "viz")]
            last_attention: None,
        })
    }

    pub fn split(self, parts: usize) -> Vec<Self> {
        if parts == 0 {
            return Vec::new();
        }

        if let Some(kv) = self.linear_kv {
            let [batch, _, _, _] = kv.shape().dims::<4>();
            let mut out = Vec::with_capacity(parts);
            let stride = batch.div_ceil(parts);
            for idx in 0..parts {
                let start = idx * stride;
                if start >= batch {
                    out.push(Self {
                        q_rot: None,
                        value: None,
                        linear_kv: None,
                        steps: self.steps,
                        #[cfg(feature = "viz")]
                        last_attention: None,
                    });
                    continue;
                }
                let end = usize::min(start + stride, batch);
                let slice = kv.clone().slice_dim(0, start..end);
                out.push(Self {
                    q_rot: None,
                    value: None,
                    linear_kv: Some(slice),
                    steps: self.steps,
                    #[cfg(feature = "viz")]
                    last_attention: None,
                });
            }
            out
        } else {
            vec![
                Self {
                    q_rot: None,
                    value: None,
                    linear_kv: None,
                    steps: self.steps,
                    #[cfg(feature = "viz")]
                    last_attention: None,
                };
                parts
            ]
        }
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
}

impl<B: Backend> Attention<B> {
    pub fn new(
        latent: usize,
        n_head: usize,
        device: &B::Device,
        kernel: &FusedKernelConfig,
    ) -> Self {
        let freqs = Self::build_freqs(latent, kernel.rope_theta, device);
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

    pub fn forward_linear_cached(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        cache: &mut AttentionCache<B>,
    ) -> Tensor<B, 4> {
        let position = cache.len();
        let q_rot = self.rotate(query, position);
        let k_rot = q_rot.clone();
        let value_rep = value.repeat_dim(1, self.n_head);

        let [batch, heads, time, latent] = k_rot.shape().dims::<4>();
        let n_embd = value_rep.shape().dims::<4>()[3];
        let device = k_rot.device();

        let mut synaptic = cache
            .linear_kv
            .take()
            .unwrap_or_else(|| Tensor::<B, 4>::zeros([batch, heads, latent, n_embd], &device));

        let mut outputs: Vec<Tensor<B, 4>> = Vec::with_capacity(time);

        for t in 0..time {
            let q_t = q_rot.clone().slice_dim(2, t..t + 1);
            let ctx_t = q_t.clone().matmul(synaptic.clone());
            outputs.push(ctx_t);

            let k_t = k_rot.clone().slice_dim(2, t..t + 1);
            let v_t = value_rep.clone().slice_dim(2, t..t + 1);
            let outer = k_t.swap_dims(2, 3).matmul(v_t);
            synaptic = synaptic + outer;
        }

        cache.linear_kv = Some(synaptic);
        cache.steps = cache.steps.saturating_add(time);
        cache.q_rot = None;
        cache.value = None;

        if outputs.is_empty() {
            return Tensor::<B, 4>::zeros([batch, heads, 0, n_embd], &device);
        }

        Tensor::cat(outputs, 2)
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

    fn rotate(&self, values: Tensor<B, 4>, start: usize) -> Tensor<B, 4> {
        let time = values.shape().dims::<4>()[2];
        let device = values.device();
        let positions = Tensor::<B, 1, Int>::arange(start as i64..(start + time) as i64, &device)
            .float()
            .reshape([1, 1, time, 1]);

        let raw = positions * self.freqs.clone();
        let phases = (raw.clone() - raw.floor()) * (2.0 * PI);
        self.rope(phases, values)
    }

    fn build_freqs(latent: usize, theta: f32, device: &B::Device) -> Tensor<B, 4> {
        let mut data = Vec::with_capacity(latent);
        for idx in 0..latent {
            let quantized = (idx as f32 / 2.0).floor() * 2.0;
            let exponent = quantized / latent as f32;
            let value = 1.0 / theta.powf(exponent) / (2.0 * PI);
            data.push(value);
        }
        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([1, 1, 1, latent])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{Distribution as TensorDistribution, Tensor};
    use burn_ndarray::NdArray;

    type Backend = NdArray<f32>;

    fn assert_close(lhs: Tensor<Backend, 4>, rhs: Tensor<Backend, 4>, atol: f32, rtol: f32) {
        let lhs_data = lhs
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("lhs vec");
        let rhs_data = rhs
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("rhs vec");

        for (a, b) in lhs_data.iter().zip(rhs_data.iter()) {
            let diff = (a - b).abs();
            let tol = atol + rtol * b.abs();
            assert!(
                diff <= tol,
                "difference {diff} exceeds tolerance {tol} (lhs={a}, rhs={b})"
            );
        }
    }

    #[test]
    fn linear_cached_matches_full_attention() {
        let device = <Backend as BackendTrait>::Device::default();
        <Backend as BackendTrait>::seed(&device, 7);

        let batch = 2;
        let heads = 2;
        let time = 5;
        let latent = 4;
        let n_embd = 6;

        let attention = Attention::new(latent, heads, &device, &FusedKernelConfig::default());

        let query = Tensor::<Backend, 4>::random(
            [batch, heads, time, latent],
            TensorDistribution::Normal(0.0, 1.0),
            &device,
        );
        let value = Tensor::<Backend, 4>::random(
            [batch, 1, time, n_embd],
            TensorDistribution::Normal(0.0, 1.0),
            &device,
        );

        let full = attention.forward(query.clone(), value.clone());

        let mut cache = AttentionCache::new();
        let linear = attention.forward_linear_cached(query, value, &mut cache);

        assert_close(linear, full, 1e-4, 1e-4);
    }

    #[test]
    fn linear_cached_streaming_matches_full_attention() {
        let device = <Backend as BackendTrait>::Device::default();
        <Backend as BackendTrait>::seed(&device, 99);

        let batch = 1;
        let heads = 3;
        let time = 6;
        let latent = 8;
        let n_embd = 5;

        let attention = Attention::new(latent, heads, &device, &FusedKernelConfig::default());

        let query = Tensor::<Backend, 4>::random(
            [batch, heads, time, latent],
            TensorDistribution::Normal(0.0, 1.0),
            &device,
        );
        let value = Tensor::<Backend, 4>::random(
            [batch, 1, time, n_embd],
            TensorDistribution::Normal(0.0, 1.0),
            &device,
        );

        let full = attention.forward(query.clone(), value.clone());

        let mut cache = AttentionCache::new();
        let split = 2;
        let q_first = query.clone().slice_dim(2, 0..split);
        let v_first = value.clone().slice_dim(2, 0..split);
        let out_first = attention.forward_linear_cached(q_first, v_first, &mut cache);

        let q_second = query.clone().slice_dim(2, split..time);
        let v_second = value.clone().slice_dim(2, split..time);
        let out_second = attention.forward_linear_cached(q_second, v_second, &mut cache);

        let streaming = Tensor::cat(vec![out_first, out_second], 2);

        assert_close(streaming, full, 1e-4, 1e-4);
    }
}
