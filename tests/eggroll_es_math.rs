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

    fn forward(&self) -> Tensor<B, 2> {
        self.w.val()
    }
}

#[derive(Clone)]
struct LinearObjective;

impl<B: Backend> EggrollObjective<ScalarModel<B>, B> for LinearObjective {
    type Batch = ();
    type PopLogits = ();

    fn evaluate(&self, model: &ScalarModel<B>, _batch: &Self::Batch) -> f32 {
        let val = model.forward();
        val.to_data().convert::<f32>().into_vec::<f32>().unwrap()[0]
    }

    fn evaluate_with_noise(
        &self,
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
            rank: 1,
            sigma: 0.5,
            lr: 10.0,
            weight_decay: 0.0,
            seed,
            max_param_norm: None,
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
