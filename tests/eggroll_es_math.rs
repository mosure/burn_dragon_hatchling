use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use burn_dragon_hatchling::eggroll::{
    EggrollConfig, EggrollNoiseTensor, EggrollObjective, EggrollParamSpec, EggrollNoiser,
    EggrollKey, EsTreeKey,
};
use burn::module::Module;

#[derive(Module, Debug)]
struct ScalarModel<B: Backend> {
    w: burn::module::Param<Tensor<B, 2>>,
}

impl<B: Backend> ScalarModel<B> {
    fn new(val: f32, device: &B::Device) -> Self {
        let w = Tensor::<B, 2>::from_floats([[val]], device);
        Self {
            w: burn::module::Param::from_tensor(w),
        }
    }

    // fn forward(&self) -> Tensor<B, 2> {
    //     self.w.val()
    // }
}

#[derive(Clone)]
struct LinearObjective;

impl<B: Backend> EggrollObjective<ScalarModel<B>, B> for LinearObjective {
    type Batch = ();

    fn evaluate(&mut self, logits: &Tensor<B, 3>, _batch: &Self::Batch) -> f32 {
        logits
            .clone()
            .reshape([1])
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap()[0]
    }

    fn evaluate_with_noise(
        &mut self,
        model: &ScalarModel<B>,
        _batch: &Self::Batch,
        noiser: &EggrollNoiser<B>,
        es_key: &EsTreeKey,
        thread_id: u32,
        _deterministic: bool,
    ) -> f32 {
        let spec = EggrollParamSpec {
            id: model.w.id,
            path: "w".into(),
            shape: (1, 1),
            rank: 1,
            sigma_scale: 1.0,
            stack: None,
        };
        let delta = match noiser.low_rank_delta(&spec, es_key, thread_id) {
            EggrollNoiseTensor::D2(d) => d,
            EggrollNoiseTensor::D3(_) => unreachable!(),
        };
        let noisy = model.w.val() + delta;
        noisy.to_data().convert::<f32>().into_vec::<f32>().unwrap()[0]
    }
}

#[test]
fn pop_update_matches_unbiased_estimator() {
    type B = NdArray<f32>;
    let device = <B as Backend>::Device::default();
    let spec = EggrollParamSpec {
        id: burn::module::ParamId::new(),
        path: "w".into(),
        shape: (4, 4),
        rank: 1,
        sigma_scale: 1.0,
        stack: None,
    };
    let config = EggrollConfig {
        pop_size: 4,
        pop_chunk_size: 2,
        rank: 1,
        sigma: 0.5,
        lr: 0.1,
        weight_decay: 0.0,
        seed: 7,
        max_param_norm: None,
        pop_vectorized: false,
        antithetic: false,
    };
    let noiser = EggrollNoiser::new(vec![spec.clone()], config.clone(), &device);
    let tree_key = EsTreeKey::new(EggrollKey::from_seed(config.seed));
    let step = 0;
    let worker_ids: Vec<u32> = (0..config.pop_size as u32).collect();
    let scores: Vec<f32> = vec![0.4, -0.2, 0.1, -0.1];

    let updates = noiser.compute_updates_from_population(
        &tree_key,
        step,
        &worker_ids,
        &scores,
    );
    let delta_pop = match updates.get(&spec.id) {
        Some(EggrollNoiseTensor::D2(t)) => t.clone(),
        _ => panic!("expected 2D update"),
    };

    let mut delta_ref = Tensor::<B, 2>::zeros([spec.shape.0, spec.shape.1], &device);
    let es_step = tree_key.clone().with_step(step);
    for (idx, &score) in scores.iter().enumerate() {
        let tid = worker_ids[idx];
        let delta = match noiser.low_rank_delta(&spec, &es_step, tid) {
            EggrollNoiseTensor::D2(d) => d,
            EggrollNoiseTensor::D3(_) => panic!("unexpected 3D delta"),
        };
        let scale = score / (config.pop_size as f32 * config.sigma);
        delta_ref = delta_ref + delta.mul_scalar(scale);
    }

    let diff = (delta_pop - delta_ref)
        .abs()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    let max_diff = diff.iter().copied().fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "population update should match unbiased estimator, max_diff={}",
        max_diff
    );
}

#[test]
fn antithetic_pairs_cancel_when_scores_match() {
    type B = NdArray<f32>;
    let device = <B as Backend>::Device::default();
    let spec = EggrollParamSpec {
        id: burn::module::ParamId::new(),
        path: "w".into(),
        shape: (2, 2),
        rank: 1,
        sigma_scale: 1.0,
        stack: None,
    };
    let config = EggrollConfig {
        pop_size: 4,
        pop_chunk_size: 2,
        rank: 1,
        sigma: 0.2,
        lr: 0.1,
        weight_decay: 0.0,
        seed: 11,
        max_param_norm: None,
        pop_vectorized: false,
        antithetic: true,
    };
    let noiser = EggrollNoiser::new(vec![spec.clone()], config.clone(), &device);
    let tree_key = EsTreeKey::new(EggrollKey::from_seed(config.seed));
    let worker_ids: Vec<u32> = (0..config.pop_size as u32).collect();
    // Pair 0/1 share base noise with opposite sign and same score; should cancel.
    let scores = vec![1.0, 1.0, 0.0, 0.0];

    let updates = noiser.compute_updates_from_population(
        &tree_key,
        0,
        &worker_ids,
        &scores,
    );
    let delta_pop = match updates.get(&spec.id) {
        Some(EggrollNoiseTensor::D2(t)) => t.clone(),
        _ => panic!("expected 2D update"),
    };

    let norm = delta_pop
        .clone()
        .abs()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap()
        .into_iter()
        .fold(0.0_f32, f32::max);
    assert!(
        norm < 1e-4,
        "antithetic pair with equal scores should cancel; norm={norm}"
    );
}

#[test]
fn es_gradient_aligns_with_finite_difference() {
    type B = NdArray<f32>;
    let device = <B as Backend>::Device::default();

    let model = ScalarModel::<B>::new(0.5, &device);
    let target = 1.0;
    let _objective = LinearObjective;

    let spec = EggrollParamSpec {
        id: model.w.id,
        path: "w".into(),
        shape: (1, 1),
        rank: 1,
        sigma_scale: 1.0,
        stack: None,
    };

    let mut moved = false;
    for seed in 1..=5 {
        let config = EggrollConfig {
            pop_size: 512,
            pop_chunk_size: 32,
            rank: 1,
            sigma: 0.5,
            lr: 10.0,
            weight_decay: 0.0,
            seed,
            max_param_norm: None,
            pop_vectorized: true,
            antithetic: false,
        };

        let base_w = model.w.val();
        let mut fitness = Vec::new();
        let mut deltas = Vec::new();
        let es_key = EsTreeKey::new(EggrollKey::from_seed(config.seed)).with_step(0);
        for tid in 0..config.pop_size {
            let delta: Tensor<B, 2> = match EggrollNoiser::new(
                vec![spec.clone()],
                config.clone(),
                &device,
            )
            .low_rank_delta(&spec, &es_key, tid as u32)
            {
                EggrollNoiseTensor::D2(delta) => delta,
                EggrollNoiseTensor::D3(_) => panic!("expected 2D delta"),
            };
            let candidate = base_w.clone() + delta.clone();
            let val = candidate.to_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
            let fit = -(val - target) * (val - target);
            fitness.push(fit);
            deltas.push(delta);
        }

        let mean = fitness.iter().copied().sum::<f32>() / fitness.len() as f32;
        let var = fitness
            .iter()
            .map(|f| (f - mean) * (f - mean))
            .sum::<f32>()
            / (fitness.len() as f32).max(1.0);
        let std = var.sqrt().max(1e-8);
        let fitness_norm: Vec<f32> = fitness.iter().map(|f| (f - mean) / std).collect();

        let mut update = Tensor::<B, 2>::zeros([1, 1], &device);
        for (score, delta) in fitness_norm.iter().copied().zip(deltas.clone().into_iter()) {
            update = update + delta.mul_scalar(score / (config.pop_size as f32 * config.sigma));
        }
        update = update.mul_scalar(config.lr);

        let new_w = base_w.clone() + update;
        let start = base_w.to_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
        let end = new_w.to_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
        let delta_norm: f32 = deltas
            .iter()
            .map(|d| d.clone().abs().sum().to_data().convert::<f32>().into_vec::<f32>().unwrap()[0])
            .sum();
        if delta_norm > 0.0 && end > start {
            moved = true;
            break;
        }
    }

    assert!(moved, "ES update did not move toward higher fitness for any tested seed");
}
