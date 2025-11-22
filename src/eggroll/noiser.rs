use std::collections::HashMap;

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn::tensor::module::embedding as emb_lookup;

use super::rng::{normal_f32_from_offset, EggrollKey, EsTreeKey};

#[derive(Clone, Debug)]
pub struct EggrollConfig {
    pub pop_size: usize,
    pub rank: usize,
    pub sigma: f32,
    pub lr: f32,
    pub weight_decay: f32,
    pub seed: u64,
    pub max_param_norm: Option<f32>,
}

impl Default for EggrollConfig {
    fn default() -> Self {
        Self {
            pop_size: 1024,
            rank: 4,
            sigma: 0.01,
            lr: 1e-3,
            weight_decay: 0.0,
            seed: 42,
            max_param_norm: None,
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

    pub fn do_mm(
        &self,
        x: Tensor<B, 2>,
        w: &Tensor<B, 2>,
        param_id: ParamId,
        es_tree_key: &EsTreeKey,
        thread_id: u32,
    ) -> Tensor<B, 2> {
        let base = x.clone().matmul(w.clone().swap_dims(0, 1));
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
        let n = spec.shape.1;
        let b = noise.clone().slice_dim(0, 0..n).reshape([n, rank]);
        let a = noise.slice_dim(0, n..(spec.shape.0 + n)).reshape([spec.shape.0, rank]);
        let corr = x.matmul(b).matmul(a.swap_dims(0, 1));
        base + corr.mul_scalar(self.scale(spec))
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
            let repeat = ((stack + dims[1] - 1) / dims[1]).max(1);
            x_expanded = x_expanded.repeat_dim(1, repeat).slice_dim(1, 0..stack);
        }

        let corr = x_expanded.matmul(a).matmul(b);
        base + corr.mul_scalar(self.scale(spec))
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
                } else if D == 3 {
                    if let Some(EggrollNoiseTensor::D3(delta)) = self.noise.get(&id) {
                        let delta = Tensor::<B, D>::from_data(
                            delta.clone().to_data(),
                            &tensor.device(),
                        );
                        let updated = tensor + delta;
                        return Param::from_mapped_value(id, updated, mapper);
                    }
                }
                Param::from_mapped_value(id, tensor, mapper)
            }
        }

        let mut mapper = Mapper { noise };
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
