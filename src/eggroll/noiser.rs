use std::collections::HashMap;

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn::tensor::module::embedding as emb_lookup;

use super::rng::{normal_f32_from_offset, EggrollKey, EsTreeKey};

#[derive(Clone, Debug)]
pub struct EggrollConfig {
    pub pop_size: usize,
    pub pop_chunk_size: usize,
    pub rank: usize,
    pub sigma: f32,
    pub lr: f32,
    pub weight_decay: f32,
    pub seed: u64,
    pub max_param_norm: Option<f32>,
    pub pop_vectorized: bool,
    pub antithetic: bool,
}

impl Default for EggrollConfig {
    fn default() -> Self {
        Self {
            pop_size: 1024,
            pop_chunk_size: 16,
            rank: 4,
            sigma: 0.01,
            lr: 1e-3,
            weight_decay: 0.0,
            seed: 42,
            max_param_norm: None,
            pop_vectorized: true,
            antithetic: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EggrollParamSpec {
    pub id: ParamId,
    pub path: String,
    pub shape: (usize, usize),
    pub rank: usize,
    pub sigma_scale: f32,
    pub stack: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct EggrollState<M, B>
where
    M: Module<B>,
    B: Backend,
{
    pub mean_record: M::Record,
    pub config: EggrollConfig,
    pub param_specs: Vec<EggrollParamSpec>,
    pub step: u64,
    pub es_key: EsTreeKey,
    pub base_key: EggrollKey,
    pub last_fitness_mean: Option<f32>,
    pub last_fitness_std: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct EggrollNoiserFrozen {
    pub param_specs: Vec<EggrollParamSpec>,
}

#[derive(Clone, Debug)]
pub struct EggrollNoiserParams<B: Backend> {
    pub config: EggrollConfig,
    pub device: B::Device,
}

#[derive(Clone, Debug)]
pub enum EggrollNoiseTensor<B: Backend> {
    D2(Tensor<B, 2>),
    D3(Tensor<B, 3>),
}

pub enum EggrollFactors<B: Backend> {
    /// 2D weight: A [pop, rows, rank], B [pop, cols, rank]
    D2 { a: Tensor<B, 3>, b: Tensor<B, 3> },
    /// Stacked 3D weight: A [pop, stack, rows, rank], B [pop, stack, cols, rank]
    D3 { a: Tensor<B, 4>, b: Tensor<B, 4> },
}

enum WorkerFactors<B: Backend> {
    /// 2D weight: A [rows, rank], B [cols, rank]
    D2 { a: Tensor<B, 2>, b: Tensor<B, 2> },
    /// Stacked 3D weight: A [stack, rows, rank], B [stack, cols, rank]
    D3 { a: Tensor<B, 3>, b: Tensor<B, 3> },
}

pub struct EggrollNoiser<B: Backend> {
    pub frozen: EggrollNoiserFrozen,
    pub params: EggrollNoiserParams<B>,
    spec_map: HashMap<ParamId, EggrollParamSpec>,
}

impl<B: Backend> EggrollNoiser<B> {
    pub fn new(
        param_specs: Vec<EggrollParamSpec>,
        config: EggrollConfig,
        device: &B::Device,
    ) -> Self {
        let spec_map = param_specs
            .iter()
            .cloned()
            .map(|spec| (spec.id, spec))
            .collect::<HashMap<_, _>>();

        Self {
            frozen: EggrollNoiserFrozen {
                param_specs: param_specs.clone(),
            },
            params: EggrollNoiserParams {
                config,
                device: device.clone(),
            },
            spec_map,
        }
    }

    pub fn param_spec(&self, id: ParamId) -> Option<&EggrollParamSpec> {
        self.spec_map.get(&id)
    }

    pub fn noise_block(
        &self,
        spec: &EggrollParamSpec,
        es_tree_key: &EsTreeKey,
        thread_id: u32,
    ) -> EggrollNoiseTensor<B> {
        if spec.stack.is_some() {
            EggrollNoiseTensor::D3(self.noise_block_3d(spec, es_tree_key, thread_id))
        } else {
            EggrollNoiseTensor::D2(self.noise_block_2d(spec, es_tree_key, thread_id))
        }
    }

    fn noise_block_2d(
        &self,
        spec: &EggrollParamSpec,
        es_tree_key: &EsTreeKey,
        thread_id: u32,
    ) -> Tensor<B, 2> {
        let rows_plus_cols = spec.shape.0 + spec.shape.1;
        let rank = spec.rank.max(1);
        let key = es_tree_key.for_axis(spec.id, thread_id, 0);
        let counter_offset = es_tree_key.step * 10_000_000 + (thread_id as u64) * 1_000;
        let samples = normal_f32_from_offset::<B>(
            key,
            rows_plus_cols * rank,
            counter_offset,
            &self.params.device,
        );
        samples.reshape([rows_plus_cols, rank])
    }

    fn noise_block_3d(
        &self,
        spec: &EggrollParamSpec,
        es_tree_key: &EsTreeKey,
        thread_id: u32,
    ) -> Tensor<B, 3> {
        let Some(stack) = spec.stack else {
            panic!("noise_block_3d requires stacked param spec");
        };
        let rows_plus_cols = spec.shape.0 + spec.shape.1;
        let rank = spec.rank.max(1);
        let key = es_tree_key.for_axis(spec.id, thread_id, 0);
        let counter_offset = es_tree_key.step * 10_000_000 + (thread_id as u64) * 1_000;
        let samples = normal_f32_from_offset::<B>(
            key,
            stack * rows_plus_cols * rank,
            counter_offset,
            &self.params.device,
        );
        samples.reshape([stack, rows_plus_cols, rank])
    }

    fn scale(&self, spec: &EggrollParamSpec) -> f32 {
        let rank = spec.rank.max(1) as f32;
        self.params.config.sigma * spec.sigma_scale / rank.sqrt()
    }

    pub fn low_rank_delta(
        &self,
        spec: &EggrollParamSpec,
        es_tree_key: &EsTreeKey,
        thread_id: u32,
    ) -> EggrollNoiseTensor<B> {
        if let Some(stack) = spec.stack {
            let noise = self.noise_block_3d(spec, es_tree_key, thread_id);
            let rank = spec.rank.max(1);
            let rows = spec.shape.0;
            let cols = spec.shape.1;
            let a = noise
                .clone()
                .slice_dim(1, 0..rows)
                .reshape([stack, rows, rank]);
            let b = noise
                .slice_dim(1, rows..(rows + cols))
                .reshape([stack, cols, rank]);
            let corr = a.matmul(b.swap_dims(1, 2)).mul_scalar(self.scale(spec));
            EggrollNoiseTensor::D3(corr)
        } else {
            let noise = self.noise_block_2d(spec, es_tree_key, thread_id);
            let rank = spec.rank.max(1);
            let n = spec.shape.1;
            let a = noise
                .clone()
                .slice_dim(0, 0..spec.shape.0)
                .reshape([spec.shape.0, rank]);
            let b = noise.slice_dim(0, spec.shape.0..(spec.shape.0 + n));
            let b = b.reshape([n, rank]);
            let ab_t = a.matmul(b.swap_dims(0, 1)).mul_scalar(self.scale(spec));
            EggrollNoiseTensor::D2(ab_t)
        }
    }

    pub fn low_rank_delta_batch(
        &self,
        spec: &EggrollParamSpec,
        es_tree_key: &EsTreeKey,
        worker_ids: &[u32],
    ) -> EggrollNoiseTensor<B> {
        let tid = worker_ids.first().copied().unwrap_or(0);
        self.low_rank_delta(spec, es_tree_key, tid)
    }

    pub fn low_rank_factors_batch(
        &self,
        spec: &EggrollParamSpec,
        es_tree_key: &EsTreeKey,
        worker_ids: &[u32],
    ) -> EggrollFactors<B> {
        if let Some(stack) = spec.stack {
            let mut a_list = Vec::with_capacity(worker_ids.len());
            let mut b_list = Vec::with_capacity(worker_ids.len());
            for &tid in worker_ids {
                let noise = self.noise_block_3d(spec, es_tree_key, tid);
                let rows = spec.shape.0;
                let cols = spec.shape.1;
                let rank = spec.rank.max(1);
                let a = noise
                    .clone()
                    .slice_dim(1, 0..rows)
                    .reshape([stack, rows, rank])
                    .unsqueeze_dim::<4>(0);
                let b = noise
                    .slice_dim(1, rows..(rows + cols))
                    .reshape([stack, cols, rank])
                    .unsqueeze_dim::<4>(0);
                a_list.push(a);
                b_list.push(b);
            }
            let a = if a_list.is_empty() {
                Tensor::<B, 4>::zeros([0, stack, spec.shape.0, spec.rank.max(1)], &self.params.device)
            } else {
                Tensor::cat(a_list, 0)
            };
            let b = if b_list.is_empty() {
                Tensor::<B, 4>::zeros([0, stack, spec.shape.1, spec.rank.max(1)], &self.params.device)
            } else {
                Tensor::cat(b_list, 0)
            };
            EggrollFactors::D3 { a, b }
        } else {
            let mut a_list = Vec::with_capacity(worker_ids.len());
            let mut b_list = Vec::with_capacity(worker_ids.len());
            for &tid in worker_ids {
                let noise = self.noise_block_2d(spec, es_tree_key, tid);
                let rank = spec.rank.max(1);
                let a = noise
                    .clone()
                    .slice_dim(0, 0..spec.shape.0)
                    .reshape([spec.shape.0, rank])
                    .unsqueeze_dim::<3>(0);
                let b = noise
                    .slice_dim(0, spec.shape.0..(spec.shape.0 + spec.shape.1))
                    .reshape([spec.shape.1, rank])
                    .unsqueeze_dim::<3>(0);
                a_list.push(a);
                b_list.push(b);
            }
            let a = if a_list.is_empty() {
                Tensor::<B, 3>::zeros([0, spec.shape.0, spec.rank.max(1)], &self.params.device)
            } else {
                Tensor::cat(a_list, 0)
            };
            let b = if b_list.is_empty() {
                Tensor::<B, 3>::zeros([0, spec.shape.1, spec.rank.max(1)], &self.params.device)
            } else {
                Tensor::cat(b_list, 0)
            };
            EggrollFactors::D2 { a, b }
        }
    }

    fn low_rank_factors_worker(
        &self,
        spec: &EggrollParamSpec,
        es_tree_key: &EsTreeKey,
        worker_id: u32,
    ) -> WorkerFactors<B> {
        let (key_worker, sign) = if self.params.config.antithetic {
            let base = worker_id / 2;
            let sign = if worker_id.is_multiple_of(2) { 1.0 } else { -1.0 };
            (base, sign)
        } else {
            (worker_id, 1.0)
        };
        if let Some(stack) = spec.stack {
            let noise = self.noise_block_3d(spec, es_tree_key, key_worker);
            let rows = spec.shape.0;
            let cols = spec.shape.1;
            let rank = spec.rank.max(1);
            let a = noise
                .clone()
                .slice_dim(1, 0..rows)
                .reshape([stack, rows, rank])
                .mul_scalar(sign);
            let b = noise
                .slice_dim(1, rows..(rows + cols))
                .reshape([stack, cols, rank]);
            WorkerFactors::D3 { a, b }
        } else {
            let noise = self.noise_block_2d(spec, es_tree_key, key_worker);
            let rows = spec.shape.0;
            let cols = spec.shape.1;
            let rank = spec.rank.max(1);
            let a = noise
                .clone()
                .slice_dim(0, 0..rows)
                .reshape([rows, rank])
                .mul_scalar(sign);
            let b = noise
                .slice_dim(0, rows..(rows + cols))
                .reshape([cols, rank]);
            WorkerFactors::D2 { a, b }
        }
    }

    pub fn compute_updates_from_population(
        &self,
        es_tree_key: &EsTreeKey,
        step: u64,
        worker_ids: &[u32],
        scores: &[f32],
    ) -> HashMap<ParamId, EggrollNoiseTensor<B>> {
        let mut updates = HashMap::new();
        if self.params.config.sigma == 0.0 || worker_ids.is_empty() {
            return updates;
        }
        let es_step = es_tree_key.clone().with_step(step);

        for spec in &self.frozen.param_specs {
            let pop = worker_ids.len() as f32;
            let sigma = self.params.config.sigma.max(1e-8);
            let noise_scale = self.scale(spec);
            let scale = noise_scale / (pop * sigma);

            match (spec.stack, spec.shape) {
                (Some(stack), (rows, cols)) => {
                    let mut acc =
                        Tensor::<B, 3>::zeros([stack, rows, cols], &self.params.device);
                    for (worker_id, score) in worker_ids.iter().copied().zip(scores.iter().copied())
                    {
                        let WorkerFactors::D3 { a, b } =
                            self.low_rank_factors_worker(spec, &es_step, worker_id)
                        else {
                            unreachable!();
                        };
                        let delta = a.matmul(b.swap_dims(2, 1));
                        acc = acc + delta.mul_scalar(score * scale);
                    }
                    if let Some(max_norm) = self.params.config.max_param_norm && max_norm > 0.0 {
                        let norm_val = acc
                            .clone()
                            .powf_scalar(2.0)
                            .sum()
                            .sqrt()
                            .to_data()
                            .convert::<f32>()
                            .into_vec::<f32>()
                            .unwrap_or_default()
                            .first()
                            .copied()
                            .unwrap_or(0.0);
                        if norm_val > max_norm && norm_val > 0.0 {
                            let factor = max_norm / norm_val;
                            acc = acc.mul_scalar(factor);
                        }
                    }
                    updates.insert(spec.id, EggrollNoiseTensor::D3(acc));
                }
                (None, (rows, cols)) => {
                    let mut acc = Tensor::<B, 2>::zeros([rows, cols], &self.params.device);
                    for (worker_id, score) in worker_ids.iter().copied().zip(scores.iter().copied())
                    {
                        let WorkerFactors::D2 { a, b } =
                            self.low_rank_factors_worker(spec, &es_step, worker_id)
                        else {
                            unreachable!();
                        };
                        let delta = a.matmul(b.swap_dims(0, 1));
                        acc = acc + delta.mul_scalar(score * scale);
                    }
                    if let Some(max_norm) = self.params.config.max_param_norm && max_norm > 0.0 {
                        let norm_val = acc
                            .clone()
                            .powf_scalar(2.0)
                            .sum()
                            .sqrt()
                            .to_data()
                            .convert::<f32>()
                            .into_vec::<f32>()
                            .unwrap_or_default()
                            .first()
                            .copied()
                            .unwrap_or(0.0);
                        if norm_val > max_norm && norm_val > 0.0 {
                            let factor = max_norm / norm_val;
                            acc = acc.mul_scalar(factor);
                        }
                    }
                    updates.insert(spec.id, EggrollNoiseTensor::D2(acc));
                }
            }
        }

        updates
    }

    pub fn do_mm(
        &self,
        x: Tensor<B, 2>,
        w: &Tensor<B, 2>,
        param_id: ParamId,
        es_tree_key: &EsTreeKey,
        thread_id: u32,
    ) -> Tensor<B, 2> {
        let base = x.clone().matmul(w.clone());
        let Some(spec) = self.param_spec(param_id) else {
            return base;
        };
        if self.params.config.sigma == 0.0 || spec.sigma_scale == 0.0 {
            return base;
        }
        if spec.stack.is_some() {
            return base;
        }

        let noise = self.noise_block_2d(spec, es_tree_key, thread_id);
        let rank = spec.rank.max(1);
        let rows = spec.shape.0;
        let cols = spec.shape.1;
        let a = noise.clone().slice_dim(0, 0..rows).reshape([rows, rank]);
        let b = noise
            .slice_dim(0, rows..(rows + cols))
            .reshape([cols, rank]);
        let corr = x.matmul(a).matmul(b.swap_dims(0, 1));
        base + corr.mul_scalar(self.scale(spec))
    }

    pub fn do_mm_pop(
        &self,
        x: Tensor<B, 2>,
        w: &Tensor<B, 2>,
        param_id: ParamId,
        es_tree_key: &EsTreeKey,
        worker_ids: &[u32],
    ) -> Tensor<B, 3> {
        let base = x.clone().matmul(w.clone());
        let Some(spec) = self.param_spec(param_id) else {
            return base.unsqueeze_dim::<3>(0).repeat_dim(0, worker_ids.len());
        };
        if self.params.config.sigma == 0.0 || spec.sigma_scale == 0.0 || spec.stack.is_some() {
            return base.unsqueeze_dim::<3>(0).repeat_dim(0, worker_ids.len());
        }

        let factors = self.low_rank_factors_batch(spec, es_tree_key, worker_ids);
        let EggrollFactors::D2 { a, b } = factors else {
            unreachable!();
        };
        let mut outs = Vec::with_capacity(worker_ids.len());
        for (idx, _) in worker_ids.iter().enumerate() {
            let a_i = a.clone().slice_dim(0, idx..idx + 1).reshape([spec.shape.0, spec.rank.max(1)]);
            let b_i = b.clone().slice_dim(0, idx..idx + 1).reshape([spec.shape.1, spec.rank.max(1)]);
            let corr = x.clone().matmul(a_i.clone()).matmul(b_i.swap_dims(0, 1));
            let noisy = base.clone() + corr.mul_scalar(self.scale(spec));
            outs.push(noisy.unsqueeze_dim::<3>(0));
        }
        Tensor::cat(outs, 0)
    }

    pub fn do_tmm(
        &self,
        x: Tensor<B, 2>,
        w: &Tensor<B, 2>,
        param_id: ParamId,
        es_tree_key: &EsTreeKey,
        thread_id: u32,
    ) -> Tensor<B, 2> {
        let base = x.clone().matmul(w.clone());
        let Some(spec) = self.param_spec(param_id) else {
            return base;
        };
        if self.params.config.sigma == 0.0 || spec.sigma_scale == 0.0 {
            return base;
        }
        if spec.stack.is_some() {
            return base;
        }
        let noise = self.noise_block_2d(spec, es_tree_key, thread_id);
        let rank = spec.rank.max(1);
        let m = spec.shape.0;
        let a = noise.clone().slice_dim(0, 0..m).reshape([m, rank]);
        let b = noise
            .slice_dim(0, m..(m + spec.shape.1))
            .reshape([spec.shape.1, rank]);
        let corr = x.matmul(a).matmul(b.swap_dims(0, 1));
        base + corr.mul_scalar(self.scale(spec))
    }

    pub fn do_tmm_pop(
        &self,
        x: Tensor<B, 2>,
        w: &Tensor<B, 2>,
        param_id: ParamId,
        es_tree_key: &EsTreeKey,
        worker_ids: &[u32],
    ) -> Tensor<B, 3> {
        let base = x.clone().matmul(w.clone());
        let Some(spec) = self.param_spec(param_id) else {
            return base.unsqueeze_dim::<3>(0).repeat_dim(0, worker_ids.len());
        };
        if self.params.config.sigma == 0.0 || spec.sigma_scale == 0.0 || spec.stack.is_some() {
            return base.unsqueeze_dim::<3>(0).repeat_dim(0, worker_ids.len());
        }

        let factors = self.low_rank_factors_batch(spec, es_tree_key, worker_ids);
        let EggrollFactors::D2 { a, b } = factors else {
            unreachable!();
        };
        let mut outs = Vec::with_capacity(worker_ids.len());
        for (idx, _) in worker_ids.iter().enumerate() {
            let a_i = a.clone().slice_dim(0, idx..idx + 1).reshape([spec.shape.0, spec.rank.max(1)]);
            let b_i = b.clone().slice_dim(0, idx..idx + 1).reshape([spec.shape.1, spec.rank.max(1)]);
            let corr = x.clone().matmul(a_i.clone()).matmul(b_i.swap_dims(0, 1));
            let noisy = base.clone() + corr.mul_scalar(self.scale(spec));
            outs.push(noisy.unsqueeze_dim::<3>(0));
        }
        Tensor::cat(outs, 0)
    }

    pub fn do_stack_tmm(
        &self,
        x: Tensor<B, 4>,
        w: &Tensor<B, 3>,
        param_id: ParamId,
        es_tree_key: &EsTreeKey,
        thread_id: u32,
    ) -> Tensor<B, 4> {
        let base = x.clone().matmul(w.clone().unsqueeze_dim::<4>(0));
        let Some(spec) = self.param_spec(param_id) else {
            return base;
        };
        if self.params.config.sigma == 0.0 || spec.sigma_scale == 0.0 {
            return base;
        }
        let Some(stack) = spec.stack else {
            return base;
        };

        let noise = self.noise_block_3d(spec, es_tree_key, thread_id);
        let rank = spec.rank.max(1);
        let rows = spec.shape.0;
        let cols = spec.shape.1;

        let a = noise
            .clone()
            .slice_dim(1, 0..rows)
            .reshape([stack, rows, rank])
            .unsqueeze_dim::<4>(0);
        let b = noise
            .slice_dim(1, rows..(rows + cols))
            .reshape([stack, cols, rank])
            .swap_dims(2, 1)
            .unsqueeze_dim::<4>(0);

        let mut x_expanded = x.clone();
        let dims = x_expanded.shape().dims::<4>();
        if dims[1] != stack {
            let repeat = stack.div_ceil(dims[1]).max(1);
            x_expanded = x_expanded.repeat_dim(1, repeat).slice_dim(1, 0..stack);
        }

        let corr = x_expanded.matmul(a).matmul(b);
        base + corr.mul_scalar(self.scale(spec))
    }

    pub fn do_stack_tmm_pop(
        &self,
        x: Tensor<B, 4>,
        w: &Tensor<B, 3>,
        param_id: ParamId,
        es_tree_key: &EsTreeKey,
        worker_ids: &[u32],
    ) -> Tensor<B, 5> {
        let base = x.clone().matmul(w.clone().unsqueeze_dim::<4>(0));
        let Some(spec) = self.param_spec(param_id) else {
            return base.unsqueeze_dim::<5>(0).repeat_dim(0, worker_ids.len());
        };
        if self.params.config.sigma == 0.0 || spec.sigma_scale == 0.0 {
            return base.unsqueeze_dim::<5>(0).repeat_dim(0, worker_ids.len());
        }
        let Some(stack) = spec.stack else {
            return base.unsqueeze_dim::<5>(0).repeat_dim(0, worker_ids.len());
        };

        let factors = self.low_rank_factors_batch(spec, es_tree_key, worker_ids);
        let EggrollFactors::D3 { a, b } = factors else {
            unreachable!();
        };

        let mut outs = Vec::with_capacity(worker_ids.len());
        for (idx, _) in worker_ids.iter().enumerate() {
            let a_i = a
                .clone()
                .slice_dim(0, idx..idx + 1)
                .reshape([stack, spec.shape.0, spec.rank.max(1)])
                .unsqueeze_dim::<4>(0);
            let b_i = b
                .clone()
                .slice_dim(0, idx..idx + 1)
                .reshape([stack, spec.shape.1, spec.rank.max(1)])
                .swap_dims(2, 1)
                .unsqueeze_dim::<4>(0);

            let mut x_expanded = x.clone();
            let dims = x_expanded.shape().dims::<4>();
            if dims[1] != stack {
                let repeat = stack.div_ceil(dims[1]).max(1);
                x_expanded =
                    x_expanded.repeat_dim(1, repeat).slice_dim(1, 0..stack);
            }
            let corr = x_expanded.matmul(a_i).matmul(b_i);
            let noisy = base.clone() + corr.mul_scalar(self.scale(spec));
            outs.push(noisy.unsqueeze_dim::<5>(0));
        }
        Tensor::cat(outs, 0)
    }

    pub fn do_emb(
        &self,
        w: &Tensor<B, 2>,
        indices: &Tensor<B, 2, Int>,
        param_id: ParamId,
        es_tree_key: &EsTreeKey,
        thread_id: u32,
    ) -> Tensor<B, 3> {
        let base = emb_lookup(w.clone(), indices.clone());
        let Some(spec) = self.param_spec(param_id) else {
            return base;
        };
        if self.params.config.sigma == 0.0 || spec.sigma_scale == 0.0 {
            return base;
        }
        if spec.stack.is_some() {
            return base;
        }

        let noise = self.noise_block_2d(spec, es_tree_key, thread_id);
        let rank = spec.rank.max(1);
        let vocab = spec.shape.0;
        let a = noise
            .clone()
            .slice_dim(0, 0..vocab)
            .reshape([vocab, rank]);
        let b = noise
            .slice_dim(0, vocab..(vocab + spec.shape.1))
            .reshape([spec.shape.1, rank]);

        let [batch, seq] = indices.shape().dims();
        let gathered_a = emb_lookup(a, indices.clone()).reshape([batch * seq, rank]);
        let corr = gathered_a
            .matmul(b.swap_dims(0, 1))
            .reshape([batch, seq, spec.shape.1]);
        base + corr.mul_scalar(self.scale(spec))
    }

    pub fn do_emb_pop(
        &self,
        w: &Tensor<B, 2>,
        indices: &Tensor<B, 2, Int>,
        param_id: ParamId,
        es_tree_key: &EsTreeKey,
        worker_ids: &[u32],
    ) -> Tensor<B, 4> {
        let base = emb_lookup(w.clone(), indices.clone());
        let Some(spec) = self.param_spec(param_id) else {
            return base.unsqueeze_dim::<4>(0).repeat_dim(0, worker_ids.len());
        };
        if self.params.config.sigma == 0.0 || spec.sigma_scale == 0.0 || spec.stack.is_some() {
            return base.unsqueeze_dim::<4>(0).repeat_dim(0, worker_ids.len());
        }

        let factors = self.low_rank_factors_batch(spec, es_tree_key, worker_ids);
        let EggrollFactors::D2 { a, b } = factors else {
            unreachable!();
        };

        let [batch, seq] = indices.shape().dims();
        let mut outs = Vec::with_capacity(worker_ids.len());
        for (idx, _) in worker_ids.iter().enumerate() {
            let a_i = a
                .clone()
                .slice_dim(0, idx..idx + 1)
                .reshape([spec.shape.0, spec.rank.max(1)]);
            let b_i = b
                .clone()
                .slice_dim(0, idx..idx + 1)
                .reshape([spec.shape.1, spec.rank.max(1)]);

            let gathered_a = emb_lookup(a_i, indices.clone()); // [batch, seq, rank]
            let gathered_flat = gathered_a.reshape([batch * seq, spec.rank.max(1)]);
            let corr_flat = gathered_flat.matmul(b_i.swap_dims(0, 1)); // [batch*seq, cols]
            let corr = corr_flat.reshape([batch, seq, spec.shape.1]);
            let noisy = base.clone() + corr.mul_scalar(self.scale(spec));
            outs.push(noisy.unsqueeze_dim::<4>(0));
        }
        Tensor::cat(outs, 0)
    }

    pub fn build_noise_map(
        &self,
        es_tree_key: &EsTreeKey,
        thread_id: u32,
    ) -> HashMap<ParamId, EggrollNoiseTensor<B>> {
        self.frozen
            .param_specs
            .iter()
            .map(|spec| (spec.id, self.low_rank_delta(spec, es_tree_key, thread_id)))
            .collect()
    }

    pub fn apply_noise<M>(
        &self,
        model: &M,
        noise: &HashMap<ParamId, EggrollNoiseTensor<B>>,
    ) -> M
    where
        M: Module<B> + Clone,
    {
        struct Mapper<'a, B: Backend> {
            noise: &'a HashMap<ParamId, EggrollNoiseTensor<B>>,
        }

        impl<'a, B: Backend> ModuleMapper<B> for Mapper<'a, B> {
            fn map_float<const D: usize>(
                &mut self,
                param: Param<Tensor<B, D>>,
            ) -> Param<Tensor<B, D>> {
                let (id, tensor, mapper) = param.consume();
                if D == 2 {
                    if let Some(EggrollNoiseTensor::D2(delta)) = self.noise.get(&id) {
                        let delta = Tensor::<B, D>::from_data(
                            delta.clone().to_data(),
                            &tensor.device(),
                        );
                        let updated = tensor + delta;
                        return Param::from_mapped_value(id, updated, mapper);
                    }
                } else if D == 3 && let Some(EggrollNoiseTensor::D3(delta)) = self.noise.get(&id) {
                    let delta = Tensor::<B, D>::from_data(delta.clone().to_data(), &tensor.device());
                    let updated = tensor + delta;
                    return Param::from_mapped_value(id, updated, mapper);
                }
                Param::from_mapped_value(id, tensor, mapper)
            }
        }

        let mut mapper = Mapper { noise };
        model.clone().map(&mut mapper)
    }

    /// Apply noise deterministically from the ES tree key without materializing a noise map.
    pub fn apply_noise_from_key<M>(
        &self,
        model: &M,
        es_tree_key: &EsTreeKey,
        thread_id: u32,
    ) -> M
    where
        M: Module<B> + Clone,
    {
        struct Mapper<'a, B: Backend> {
            noiser: &'a EggrollNoiser<B>,
            es_tree_key: &'a EsTreeKey,
            thread_id: u32,
        }

        impl<'a, B: Backend> ModuleMapper<B> for Mapper<'a, B> {
            fn map_float<const D: usize>(
                &mut self,
                param: Param<Tensor<B, D>>,
            ) -> Param<Tensor<B, D>> {
                let (id, tensor, mapper) = param.consume();
                let Some(spec) = self.noiser.param_spec(id) else {
                    return Param::from_mapped_value(id, tensor, mapper);
                };
                let delta = self
                    .noiser
                    .low_rank_delta(spec, self.es_tree_key, self.thread_id);

                match (D, delta) {
                    (2, EggrollNoiseTensor::D2(delta)) => {
                        let delta = Tensor::<B, D>::from_data(delta.to_data(), &tensor.device());
                        let updated = tensor + delta;
                        Param::from_mapped_value(id, updated, mapper)
                    }
                    (3, EggrollNoiseTensor::D3(delta)) => {
                        let delta = Tensor::<B, D>::from_data(delta.to_data(), &tensor.device());
                        let updated = tensor + delta;
                        Param::from_mapped_value(id, updated, mapper)
                    }
                    _ => Param::from_mapped_value(id, tensor, mapper),
                }
            }
        }

        let mut mapper = Mapper {
            noiser: self,
            es_tree_key,
            thread_id,
        };
        model.clone().map(&mut mapper)
    }
}

pub fn discover_param_specs<M, B>(
    module: &M,
    default_rank: usize,
) -> Vec<EggrollParamSpec>
where
    M: Module<B>,
    B: Backend,
{
    struct Visitor {
        specs: Vec<EggrollParamSpec>,
        default_rank: usize,
    }

    impl<B: Backend> ModuleVisitor<B> for Visitor {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            let dims = param.val().shape().dims::<D>();
            match D {
                2 => self.specs.push(EggrollParamSpec {
                    id: param.id,
                    path: String::new(),
                    shape: (dims[0], dims[1]),
                    rank: self.default_rank,
                    sigma_scale: 1.0,
                    stack: None,
                }),
                3 => self.specs.push(EggrollParamSpec {
                    id: param.id,
                    path: String::new(),
                    shape: (dims[1], dims[2]),
                    rank: self.default_rank,
                    sigma_scale: 1.0,
                    stack: Some(dims[0]),
                }),
                _ => {}
            }
        }

        fn visit_float_with_path<const D: usize>(
            &mut self,
            path: &[String],
            id: ParamId,
            tensor: &Tensor<B, D>,
        ) {
            let dims = tensor.shape().dims::<D>();
            let path = path.join(".");
            match D {
                2 => self.specs.push(EggrollParamSpec {
                    id,
                    path,
                    shape: (dims[0], dims[1]),
                    rank: self.default_rank,
                    sigma_scale: 1.0,
                    stack: None,
                }),
                3 => self.specs.push(EggrollParamSpec {
                    id,
                    path,
                    shape: (dims[1], dims[2]),
                    rank: self.default_rank,
                    sigma_scale: 1.0,
                    stack: Some(dims[0]),
                }),
                _ => {}
            }
        }
    }

    let mut visitor = Visitor {
        specs: Vec::new(),
        default_rank,
    };
    module.visit(&mut visitor);
    visitor.specs
}
