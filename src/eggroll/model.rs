use std::collections::HashMap;

use burn::module::{Module, ModuleMapper, Param, ParamId};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::noiser::{
    EggrollConfig, EggrollNoiseTensor, EggrollNoiser, EggrollParamSpec, EggrollState,
};
use super::rng::{EggrollKey, EsTreeKey};

pub trait EggrollObjective<M, B: Backend> {
    type Batch;
    type PopLogits;

    fn evaluate(&self, model: &M, batch: &Self::Batch) -> f32;

    /// Evaluate using a noisy forward path without cloning the base model.
    fn evaluate_with_noise(
        &self,
        model: &M,
        batch: &Self::Batch,
        noiser: &EggrollNoiser<B>,
        es_key: &EsTreeKey,
        thread_id: u32,
        deterministic: bool,
    ) -> f32;

    /// Optional population path: run a batched noisy forward once and return per-population logits.
    fn forward_population(
        &self,
        _model: &M,
        _batch: &Self::Batch,
        _noiser: &EggrollNoiser<B>,
        _es_key: &EsTreeKey,
        _pop: usize,
        _deterministic: bool,
    ) -> Option<Self::PopLogits> {
        None
    }

    /// Optional population evaluation: compute fitness per population member from batched logits.
    fn evaluate_population(
        &self,
        _logits: &Self::PopLogits,
        _batch: &Self::Batch,
    ) -> Option<Vec<f32>> {
        None
    }
}

pub struct EggrollTrainer<M, B: Backend, Obj>
where
    M: Module<B> + Clone,
    Obj: EggrollObjective<M, B>,
{
    pub model: M,
    pub objective: Obj,
    pub noiser: EggrollNoiser<B>,
    pub state: EggrollState<M, B>,
}

impl<M, B, Obj> EggrollTrainer<M, B, Obj>
where
    M: Module<B> + Clone,
    B: Backend,
    Obj: EggrollObjective<M, B>,
{
    pub fn new(
        model: M,
        config: EggrollConfig,
        param_specs: Vec<EggrollParamSpec>,
        objective: Obj,
    ) -> Self {
        let devices = model.devices();
        let device = devices
            .get(0)
            .cloned()
            .unwrap_or_else(|| B::Device::default());
        let noiser = EggrollNoiser::new(param_specs.clone(), config.clone(), &device);
        let base_key = EggrollKey::from_seed(config.seed);
        let es_key = EsTreeKey::new(base_key);
        let mean_record = model.clone().into_record();

        Self {
            model,
            objective,
            noiser,
            state: EggrollState {
                mean_record,
                config,
                param_specs,
                step: 0,
                es_key,
                base_key,
            },
        }
    }
}

impl<M, B, Obj> EggrollTrainer<M, B, Obj>
where
    M: Module<B> + Clone,
    B: Backend,
    Obj: EggrollObjective<M, B>,
{
    pub fn step(&mut self, batch: &Obj::Batch) -> &M {
        let pop = self.state.config.pop_size;
        if pop == 0 {
            return &self.model;
        }

        let es_key = self.state.es_key.clone().with_step(self.state.step);

        let fitness: Vec<f32> = if let Some(pop_logits) = self.objective.forward_population(
            &self.model,
            batch,
            &self.noiser,
            &es_key,
            pop,
            true,
        ) {
            if let Some(scores) = self.objective.evaluate_population(&pop_logits, batch) {
                scores
            } else {
                (0..pop)
                    .map(|tid| {
                        self.objective.evaluate_with_noise(
                            &self.model,
                            batch,
                            &self.noiser,
                            &es_key,
                            tid as u32,
                            true,
                        )
                    })
                    .collect()
            }
        } else {
            let mut scores = Vec::with_capacity(pop);
            for thread_id in 0..pop {
                let tid = thread_id as u32;
                let score = self.objective.evaluate_with_noise(
                    &self.model,
                    batch,
                    &self.noiser,
                    &es_key,
                    tid,
                    true,
                );
                scores.push(score);
            }
            scores
        };

        let mean_f = fitness.iter().copied().sum::<f32>() / pop as f32;
        let var = fitness
            .iter()
            .copied()
            .map(|f| (f - mean_f) * (f - mean_f))
            .sum::<f32>()
            / (pop as f32).max(1.0);
        let std = var.sqrt().max(1e-8);
        let fitness_norm: Vec<f32> = fitness.iter().map(|&f| (f - mean_f) / std).collect();

        let mut updates: HashMap<ParamId, EggrollNoiseTensor<B>> = HashMap::new();
        let sigma = self
            .state
            .config
            .sigma
            .max(f32::MIN_POSITIVE);
        for spec in &self.state.param_specs {
            if let Some(stack) = spec.stack {
                let mut acc = Tensor::<B, 3>::zeros(
                    [stack, spec.shape.0, spec.shape.1],
                    &self.noiser.params.device,
                );
                for (k, score) in fitness_norm.iter().copied().enumerate() {
                    let delta = match self
                        .noiser
                        .low_rank_delta(spec, &es_key, k as u32)
                    {
                        EggrollNoiseTensor::D3(delta) => delta,
                        EggrollNoiseTensor::D2(_) => continue,
                    };
                    let scaled = delta.mul_scalar(score / (pop as f32 * sigma));
                    acc = acc + scaled;
                }
                updates.insert(
                    spec.id,
                    EggrollNoiseTensor::D3(acc.mul_scalar(self.state.config.lr)),
                );
            } else {
                let mut acc =
                    Tensor::<B, 2>::zeros([spec.shape.0, spec.shape.1], &self.noiser.params.device);
                for (k, score) in fitness_norm.iter().copied().enumerate() {
                    let delta = match self
                        .noiser
                        .low_rank_delta(spec, &es_key, k as u32)
                    {
                        EggrollNoiseTensor::D2(delta) => delta,
                        EggrollNoiseTensor::D3(_) => continue,
                    };
                    // E_k already scaled by sigma/sqrt(rank)
                    let scaled = delta.mul_scalar(score / (pop as f32 * sigma));
                    acc = acc + scaled;
                }
                updates.insert(
                    spec.id,
                    EggrollNoiseTensor::D2(acc.mul_scalar(self.state.config.lr)),
                );
            }
        }

        let weight_decay = self.state.config.weight_decay;
        let lr = self.state.config.lr;

        struct Mapper<'a, B: Backend> {
            updates: &'a HashMap<ParamId, EggrollNoiseTensor<B>>,
            weight_decay: f32,
            lr: f32,
        }

        impl<'a, B: Backend> ModuleMapper<B> for Mapper<'a, B> {
            fn map_float<const D: usize>(
                &mut self,
                param: Param<Tensor<B, D>>,
            ) -> Param<Tensor<B, D>> {
                let (id, tensor, mapper) = param.consume();
                let mut updated = tensor.clone();
                match (D, self.updates.get(&id)) {
                    (2, Some(EggrollNoiseTensor::D2(delta))) => {
                        let delta = Tensor::<B, D>::from_data(
                            delta.clone().to_data(),
                            &tensor.device(),
                        );
                        updated = updated + delta;
                    }
                    (3, Some(EggrollNoiseTensor::D3(delta))) => {
                        let delta = Tensor::<B, D>::from_data(
                            delta.clone().to_data(),
                            &tensor.device(),
                        );
                        updated = updated + delta;
                    }
                    _ => {}
                };
                if self.weight_decay != 0.0 && (D == 2 || D == 3) {
                    updated = updated - tensor.clone().mul_scalar(self.weight_decay * self.lr);
                }
                Param::from_mapped_value(id, updated, mapper)
            }
        }

        let mut mapper = Mapper {
            updates: &updates,
            weight_decay,
            lr,
        };
        self.model = self.model.clone().map(&mut mapper);
        self.state.mean_record = self.model.clone().into_record();
        self.state.step += 1;
        &self.model
    }
}
