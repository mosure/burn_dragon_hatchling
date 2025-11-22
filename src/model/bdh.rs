use burn::module::{Module, Param};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution as TensorDistribution, Int, Tensor, TensorData, activation};
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use std::cmp::Ordering;

use crate::kernel::{BlockPattern1d, relu_lowrank};
use crate::eggroll::{EggrollNoiser, EsTreeKey, EggrollParamSpec};
use crate::eggroll::bdh_integration::BdhEsConfig;

use super::attention::Attention;
use super::config::{BDHConfig, FusedKernelConfig};
#[cfg(feature = "viz")]
use super::state::LayerVizState;
use super::state::ModelState;

const LAYER_NORM_EPS: f32 = 1e-5;

#[derive(Module, Debug)]
pub struct BDH<B: Backend> {
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    mlp_internal_dim_multiplier: usize,
    vocab_size: usize,
    kernel: FusedKernelConfig,
    embed: Embedding<B>,
    dropout: Dropout,
    attention: Attention<B>,
    decoder_x: Param<Tensor<B, 3>>,
    decoder_y: Param<Tensor<B, 3>>,
    encoder: Param<Tensor<B, 2>>,
    lm_head: Param<Tensor<B, 2>>,
}

impl<B: Backend> BDH<B> {
    pub fn new(config: BDHConfig, device: &B::Device) -> Self {
        let embed = EmbeddingConfig::new(config.vocab_size, config.n_embd).init(device);
        let dropout = DropoutConfig::new(config.dropout).init();

        let latent_per_head = config.latent_per_head();
        let latent_total = config.latent_total();
        let attention = Attention::new(
            latent_per_head,
            config.n_head,
            device,
            &config.fused_kernels,
        );

        let weight_init = |shape: [usize; 2]| {
            Tensor::<B, 2>::random(shape, TensorDistribution::Normal(0.0, 0.02), device)
        };

        let decoder_x = Param::from_tensor(Tensor::<B, 3>::random(
            [config.n_head, config.n_embd, latent_per_head],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        ));

        let decoder_y = Param::from_tensor(Tensor::<B, 3>::random(
            [config.n_head, config.n_embd, latent_per_head],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        ));

        let encoder = Param::from_tensor(weight_init([latent_total, config.n_embd]));
        let lm_head = Param::from_tensor(weight_init([config.n_embd, config.vocab_size]));

        Self {
            n_layer: config.n_layer,
            n_embd: config.n_embd,
            n_head: config.n_head,
            mlp_internal_dim_multiplier: config.mlp_internal_dim_multiplier,
            vocab_size: config.vocab_size,
            kernel: config.fused_kernels,
            embed,
            dropout,
            attention,
            decoder_x,
            decoder_y,
            encoder,
            lm_head,
        }
    }

    fn layer_norm<const D: usize>(&self, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let (var, mean) = tensor.clone().var_mean_bias(D - 1);
        tensor.sub(mean).div(var.add_scalar(LAYER_NORM_EPS).sqrt())
    }

    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward_inner(tokens, None, None, None, false)
    }

    pub fn forward_with_noise(
        &self,
        tokens: Tensor<B, 2, Int>,
        noiser: &EggrollNoiser<B>,
        es_key: &EsTreeKey,
        thread_id: u32,
    ) -> Tensor<B, 3> {
        self.forward_with_noise_det(tokens, noiser, es_key, thread_id, true)
    }

    pub fn forward_with_noise_det(
        &self,
        tokens: Tensor<B, 2, Int>,
        noiser: &EggrollNoiser<B>,
        es_key: &EsTreeKey,
        thread_id: u32,
        deterministic: bool,
    ) -> Tensor<B, 3> {
        if noiser.params.config.sigma == 0.0 {
            return self.forward(tokens);
        }
        self.forward_inner(
            tokens,
            Some(noiser),
            Some(es_key),
            Some(thread_id),
            deterministic,
        )
    }

    fn forward_inner(
        &self,
        tokens: Tensor<B, 2, Int>,
        noiser: Option<&EggrollNoiser<B>>,
        es_key: Option<&EsTreeKey>,
        thread_id: Option<u32>,
        deterministic: bool,
    ) -> Tensor<B, 3> {
        let embed_out = if let (Some(n), Some(k), Some(tid)) = (noiser, es_key, thread_id) {
            n.do_emb(&self.embed.weight.val(), &tokens, self.embed.weight.id, k, tid)
        } else {
            self.embed.forward(tokens)
        };
        let mut state = embed_out.unsqueeze_dim::<4>(1);
        state = self.layer_norm(state);

        let decoder_x = self.decoder_x.val();
        let decoder_y = self.decoder_y.val();
        let encoder = self.encoder.val();
        let fused = self.kernel.enabled && noiser.is_none();
        let latent_pattern: &BlockPattern1d = &self.kernel.block_sparse.latent;

        for _ in 0..self.n_layer {
            let x_sparse = if fused {
                relu_lowrank::fused_forward(
                    state.clone(),
                    decoder_x.clone().unsqueeze_dim::<4>(0),
                    None,
                    self.kernel.relu_threshold,
                    latent_pattern,
                )
            } else {
                let mut x_latent =
                    if let (Some(n), Some(k), Some(tid)) = (noiser, es_key, thread_id) {
                        n.do_stack_tmm(state.clone(), &decoder_x, self.decoder_x.id, k, tid)
                    } else {
                        state.clone().matmul(decoder_x.clone().unsqueeze_dim::<4>(0))
                    };
                if self.kernel.relu_threshold != 0.0 {
                    x_latent = x_latent.sub_scalar(self.kernel.relu_threshold);
                }
                activation::relu(x_latent)
            };

            let attn = self.attention.forward(x_sparse.clone(), state.clone());
            let attn = self.layer_norm(attn);

            let y_sparse = if fused {
                relu_lowrank::fused_forward(
                    attn.clone(),
                    decoder_y.clone().unsqueeze_dim::<4>(0),
                    None,
                    self.kernel.relu_threshold,
                    latent_pattern,
                )
            } else {
                let mut y_latent =
                    if let (Some(n), Some(k), Some(tid)) = (noiser, es_key, thread_id) {
                        n.do_stack_tmm(attn.clone(), &decoder_y, self.decoder_y.id, k, tid)
                    } else {
                        attn.matmul(decoder_y.clone().unsqueeze_dim::<4>(0))
                    };
                if self.kernel.relu_threshold != 0.0 {
                    y_latent = y_latent.sub_scalar(self.kernel.relu_threshold);
                }
                activation::relu(y_latent)
            };
            let xy_sparse = x_sparse * y_sparse;
            let xy_sparse = if deterministic {
                xy_sparse
            } else {
                self.dropout.forward(xy_sparse)
            };

            let mixed = xy_sparse.clone().swap_dims(1, 2);
            let [batch, time, heads, latent] = mixed.shape().dims();
            let mixed_flat = mixed.reshape([batch * time, heads * latent]);

            let mlp_flat = if let (Some(n), Some(k), Some(tid)) = (noiser, es_key, thread_id) {
                n.do_tmm(mixed_flat.clone(), &encoder, self.encoder.id, k, tid)
            } else {
                mixed_flat.matmul(encoder.clone())
            };
            let mlp_out = mlp_flat
                .reshape([batch, time, self.n_embd])
                .unsqueeze_dim::<4>(1);
            let mlp_out = self.layer_norm(mlp_out);
            state = self.layer_norm(state + mlp_out);
        }

        let [batch, _, time, dim] = state.shape().dims();
        let logits_in = state.reshape([batch * time, dim]);
        let logits = if let (Some(n), Some(k), Some(tid)) = (noiser, es_key, thread_id) {
            n.do_mm(logits_in, &self.lm_head.val(), self.lm_head.id, k, tid)
        } else {
            logits_in.matmul(self.lm_head.val())
        };
        logits.reshape([batch, time, self.vocab_size])
    }

    pub fn generate(
        &self,
        mut indices: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
    ) -> Tensor<B, 2, Int> {
        let [batch, _] = indices.shape().dims();
        assert_eq!(batch, 1, "generation currently supports batch size 1");

        let mut state = self.init_state();
        let mut logits = self.forward_with_state(indices.clone(), &mut state);
        let [_, mut time, vocab] = logits.shape().dims();
        assert_eq!(time, indices.shape().dims::<2>()[1]);

        let mut last_logits = logits
            .slice_dim(1, (time - 1)..time)
            .reshape([vocab])
            .div_scalar(temperature);

        for _ in 0..max_new_tokens {
            let mut logits_values = last_logits
                .clone()
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("logits to vec");

            if let Some(k) = top_k
                && k > 0
                && k < vocab
            {
                let mut sorted = logits_values.clone();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
                let threshold = sorted[k - 1];
                for value in logits_values.iter_mut() {
                    if *value < threshold {
                        *value = f32::NEG_INFINITY;
                    }
                }
            }

            let max_logit = logits_values
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = logits_values
                .iter()
                .map(|value| (value - max_logit).exp())
                .collect();
            let sum: f32 = probs.iter().sum();
            if sum == 0.0 || sum.is_nan() {
                let uniform = 1.0 / vocab as f32;
                for p in probs.iter_mut() {
                    *p = uniform;
                }
            } else {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            }

            let dist = WeightedIndex::new(&probs).expect("valid probability distribution");
            let mut rng = thread_rng();
            let next = dist.sample(&mut rng) as i64;

            let next_token = Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![next], [1, 1]),
                &indices.device(),
            );
            indices = Tensor::cat(vec![indices, next_token.clone()], 1);

            logits = self.forward_with_state(next_token, &mut state);
            let [_, new_time, _] = logits.shape().dims();
            time = new_time;
            last_logits = logits
                .slice_dim(1, (time - 1)..time)
                .reshape([vocab])
                .div_scalar(temperature);
        }

        indices
    }

    pub fn init_state(&self) -> ModelState<B> {
        ModelState::new(self.n_layer)
    }

    pub fn forward_with_state(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        self.forward_with_state_inner(tokens, state, None, None, None, false)
    }

    pub fn forward_with_state_and_noise(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
        noiser: &EggrollNoiser<B>,
        es_key: &EsTreeKey,
        thread_id: u32,
    ) -> Tensor<B, 3> {
        self.forward_with_state_and_noise_det(
            tokens,
            state,
            noiser,
            es_key,
            thread_id,
            true,
        )
    }

    pub fn forward_with_state_and_noise_det(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
        noiser: &EggrollNoiser<B>,
        es_key: &EsTreeKey,
        thread_id: u32,
        deterministic: bool,
    ) -> Tensor<B, 3> {
        self.forward_with_state_inner(
            tokens,
            state,
            Some(noiser),
            Some(es_key),
            Some(thread_id),
            deterministic,
        )
    }

    fn forward_with_state_inner(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
        noiser: Option<&EggrollNoiser<B>>,
        es_key: Option<&EsTreeKey>,
        thread_id: Option<u32>,
        deterministic: bool,
    ) -> Tensor<B, 3> {
        assert_eq!(
            state.layers.len(),
            self.n_layer,
            "model state layers mismatch"
        );
        let embed_out = if let (Some(n), Some(k), Some(tid)) = (noiser, es_key, thread_id) {
            n.do_emb(&self.embed.weight.val(), &tokens, self.embed.weight.id, k, tid)
        } else {
            self.embed.forward(tokens)
        };
        let mut current = embed_out.unsqueeze_dim::<4>(1);
        current = self.layer_norm(current);

        let decoder_x = self.decoder_x.val();
        let decoder_y = self.decoder_y.val();
        let encoder = self.encoder.val();
        let fused = self.kernel.enabled && noiser.is_none();
        let latent_pattern: &BlockPattern1d = &self.kernel.block_sparse.latent;

        for layer_state in &mut state.layers {
            let x_sparse = if fused {
                relu_lowrank::fused_forward(
                    current.clone(),
                    decoder_x.clone().unsqueeze_dim::<4>(0),
                    None,
                    self.kernel.relu_threshold,
                    latent_pattern,
                )
            } else {
                let mut x_latent =
                    if let (Some(n), Some(k), Some(tid)) = (noiser, es_key, thread_id) {
                        n.do_stack_tmm(current.clone(), &decoder_x, self.decoder_x.id, k, tid)
                    } else {
                        current.clone().matmul(decoder_x.clone().unsqueeze_dim::<4>(0))
                    };
                if self.kernel.relu_threshold != 0.0 {
                    x_latent = x_latent.sub_scalar(self.kernel.relu_threshold);
                }
                activation::relu(x_latent)
            };

            let attn = self.attention.forward_cached(
                x_sparse.clone(),
                current.clone(),
                &mut layer_state.attention,
            );
            let attn = self.layer_norm(attn);

            let y_sparse = if fused {
                relu_lowrank::fused_forward(
                    attn.clone(),
                    decoder_y.clone().unsqueeze_dim::<4>(0),
                    None,
                    self.kernel.relu_threshold,
                    latent_pattern,
                )
            } else {
                let mut y_latent =
                    if let (Some(n), Some(k), Some(tid)) = (noiser, es_key, thread_id) {
                        n.do_stack_tmm(attn.clone(), &decoder_y, self.decoder_y.id, k, tid)
                    } else {
                        attn.matmul(decoder_y.clone().unsqueeze_dim::<4>(0))
                    };
                if self.kernel.relu_threshold != 0.0 {
                    y_latent = y_latent.sub_scalar(self.kernel.relu_threshold);
                }
                activation::relu(y_latent)
            };

            let xy_sparse = x_sparse * y_sparse;
            let xy_sparse = if deterministic {
                xy_sparse
            } else {
                self.dropout.forward(xy_sparse)
            };

            let mixed = xy_sparse.clone().swap_dims(1, 2);
            let [batch, time, heads, latent] = mixed.shape().dims();

            #[cfg(feature = "viz")]
            if time > 0 {
                let attn_rows = layer_state
                    .attention
                    .last_attention()
                    .map(|tensor| {
                        let dims = tensor.shape().dims::<3>();
                        let context = dims[2];
                        (0..dims[1])
                            .map(|head_idx| {
                                tensor
                                    .clone()
                                    .slice_dim(0, 0..1)
                                    .slice_dim(1, head_idx..head_idx + 1)
                                    .reshape([context])
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();

                let neurons = mixed
                    .clone()
                    .slice_dim(1, (time - 1)..time)
                    .reshape([batch, heads * latent])
                    .slice_dim(0, 0..1)
                    .reshape([heads * latent]);

                let synapses = xy_sparse
                    .clone()
                    .slice_dim(2, (time - 1)..time)
                    .reshape([batch, heads, latent])
                    .slice_dim(0, 0..1)
                    .reshape([heads, latent]);

                layer_state.viz = Some(LayerVizState {
                    attn_rows,
                    neurons,
                    synapses,
                });
            }

            let mixed_flat = mixed.reshape([batch * time, heads * latent]);

            let mlp_flat = if let (Some(n), Some(k), Some(tid)) = (noiser, es_key, thread_id) {
                n.do_tmm(mixed_flat.clone(), &encoder, self.encoder.id, k, tid)
            } else {
                mixed_flat.matmul(encoder.clone())
            };
            let mlp_out = mlp_flat
                .reshape([batch, time, self.n_embd])
                .unsqueeze_dim::<4>(1);
            let mlp_out = self.layer_norm(mlp_out);
            current = self.layer_norm(current + mlp_out);
        }

        state.position = state.len();

        let [batch, _, time, dim] = current.shape().dims();
        let logits_in = current.reshape([batch * time, dim]);
        let logits = if let (Some(n), Some(k), Some(tid)) = (noiser, es_key, thread_id) {
            n.do_mm(logits_in, &self.lm_head.val(), self.lm_head.id, k, tid)
        } else {
            logits_in.matmul(self.lm_head.val())
        };
        logits.reshape([batch, time, self.vocab_size])
    }

    pub fn es_param_specs(&self, cfg: &BdhEsConfig) -> Vec<EggrollParamSpec> {
        let mut specs = Vec::new();
        if cfg.embedding.enabled {
            let dims = self.embed.weight.shape().dims::<2>();
            specs.push(EggrollParamSpec {
                id: self.embed.weight.id,
                path: "embedding".into(),
                shape: (dims[0], dims[1]),
                rank: cfg.embedding.rank,
                sigma_scale: cfg.embedding.sigma_scale,
                stack: None,
            });
        }
        if cfg.decoder_x.enabled {
            let dims = self.decoder_x.shape().dims::<3>();
            specs.push(EggrollParamSpec {
                id: self.decoder_x.id,
                path: "decoder_x".into(),
                shape: (dims[1], dims[2]),
                rank: cfg.decoder_x.rank,
                sigma_scale: cfg.decoder_x.sigma_scale,
                stack: Some(dims[0]),
            });
        }
        if cfg.decoder_y.enabled {
            let dims = self.decoder_y.shape().dims::<3>();
            specs.push(EggrollParamSpec {
                id: self.decoder_y.id,
                path: "decoder_y".into(),
                shape: (dims[1], dims[2]),
                rank: cfg.decoder_y.rank,
                sigma_scale: cfg.decoder_y.sigma_scale,
                stack: Some(dims[0]),
            });
        }
        if cfg.encoder.enabled {
            let dims = self.encoder.shape().dims::<2>();
            specs.push(EggrollParamSpec {
                id: self.encoder.id,
                path: "encoder".into(),
                shape: (dims[0], dims[1]),
                rank: cfg.encoder.rank,
                sigma_scale: cfg.encoder.sigma_scale,
                stack: None,
            });
        }
        if cfg.lm_head.enabled {
            let dims = self.lm_head.shape().dims::<2>();
            specs.push(EggrollParamSpec {
                id: self.lm_head.id,
                path: "lm_head".into(),
                shape: (dims[0], dims[1]),
                rank: cfg.lm_head.rank,
                sigma_scale: cfg.lm_head.sigma_scale,
                stack: None,
            });
        }
        specs
    }
}
