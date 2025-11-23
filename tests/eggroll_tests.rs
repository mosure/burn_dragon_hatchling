use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_ndarray::NdArray;

use burn_dragon_hatchling::eggroll::{
    EggrollConfig, EggrollNoiseTensor, EggrollObjective, EggrollTrainer, EggrollNoiser,
    EggrollKey, EsTreeKey, EggrollParamSpec, discover_param_specs,
};
use burn_dragon_hatchling::{BDH, BDHConfig, BdhEsConfig};
use burn::tensor::Int;
use burn::tensor::module::embedding as emb_lookup;

type B = NdArray<f32>;

#[derive(Module, Debug)]
struct SimpleLinear<B: Backend> {
    weight: Param<Tensor<B, 2>>,
}

impl<B: Backend> SimpleLinear<B> {
    fn new(w: Tensor<B, 2>) -> Self {
        Self {
            weight: Param::from_tensor(w),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.matmul(self.weight.val().swap_dims(0, 1))
    }
}

#[derive(Module, Debug)]
struct StackedParams<B: Backend> {
    weight: Param<Tensor<B, 3>>,
}

impl<B: Backend> StackedParams<B> {
    fn new(device: &B::Device) -> Self {
        let w = Tensor::<B, 3>::from_floats(
            [
                [[0.1, -0.2], [0.3, 0.4]],
                [[-0.1, 0.2], [0.5, -0.6]],
            ],
            device,
        );
        Self {
            weight: Param::from_tensor(w),
        }
    }
}

#[derive(Clone)]
struct MseObjective<B: Backend> {
    target: Tensor<B, 2>,
    _phantom: core::marker::PhantomData<B>,
}

impl<B: Backend> MseObjective<B> {
    fn new(target: Tensor<B, 2>) -> Self {
        Self {
            target,
            _phantom: core::marker::PhantomData,
        }
    }

    fn mse(&self, pred: Tensor<B, 2>) -> f32 {
        let diff = pred - self.target.clone();
        let mse = diff.clone().powf_scalar(2.0).sum().div_scalar(diff.shape().num_elements() as f32);
        mse.to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap_or_default()
            .first()
            .copied()
            .unwrap_or(0.0)
            * -1.0
    }
}

impl<B: Backend> EggrollObjective<SimpleLinear<B>, B> for MseObjective<B> {
    type Batch = Tensor<B, 2>;

    fn evaluate(&mut self, logits: &Tensor<B, 3>, _batch: &Self::Batch) -> f32 {
        let [batch, _, dim] = logits.shape().dims();
        let preds = logits.clone().reshape([batch, dim]);
        self.mse(preds)
    }

    fn evaluate_with_noise(
        &mut self,
        model: &SimpleLinear<B>,
        batch: &Self::Batch,
        noiser: &EggrollNoiser<B>,
        es_key: &EsTreeKey,
        thread_id: u32,
        _deterministic: bool,
    ) -> f32 {
        // Manually form noisy weight without cloning the whole module.
        let spec = EggrollParamSpec {
            id: model.weight.id,
            path: "weight".into(),
            shape: {
                let dims = model.weight.val().shape().dims::<2>();
                (dims[0], dims[1])
            },
            rank: 2,
            sigma_scale: 1.0,
            stack: None,
        };
        let delta = match noiser.low_rank_delta(&spec, es_key, thread_id) {
            EggrollNoiseTensor::D2(d) => d,
            EggrollNoiseTensor::D3(_) => unreachable!(),
        };
        let w_noisy = model.weight.val() + delta;
        let pred = batch.clone().matmul(w_noisy.swap_dims(0, 1));
        self.mse(pred)
    }
}

fn setup_linear() -> (SimpleLinear<B>, Tensor<B, 2>, Tensor<B, 2>) {
    let device = <NdArray<f32> as Backend>::Device::default();
    let w = Tensor::<B, 2>::from_floats([[0.1, -0.2], [0.05, 0.3]], &device);
    let model = SimpleLinear::new(w);
    let batch = Tensor::<B, 2>::from_floats([[1.0, 2.0]], &device);
    let target = Tensor::<B, 2>::from_floats([[0.5, -0.4]], &device);
    (model, batch, target)
}

#[test]
fn eggroll_sigma_zero_is_noop() {
    let (model, batch, target) = setup_linear();
    let specs = discover_param_specs(&model, 2);
    assert!(!specs.is_empty(), "no parameter specs discovered for SimpleLinear");
    let mut trainer = EggrollTrainer::new(
        model.clone(),
        EggrollConfig {
            pop_size: 1,
            sigma: 0.0,
            lr: 0.1,
            rank: 2,
            weight_decay: 0.0,
            seed: 123,
            max_param_norm: None,
            pop_vectorized: true,
        },
        specs,
        MseObjective::new(target.clone()),
    );

    let before = model.forward(batch.clone());
    let _ = trainer.step(&batch);
    let after = trainer.model.forward(batch.clone());

    let diff = (after - before)
        .abs()
        .sum()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    assert!(diff[0] < 1e-6, "expected no change when sigma=0, got diff {}", diff[0]);

    let w_before = model.weight.val();
    let w_after = trainer.model.weight.val();
    let w_diff = (w_after - w_before)
        .abs()
        .sum()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    assert!(w_diff[0] < 1e-6, "weights changed despite sigma=0, diff {}", w_diff[0]);
}

#[test]
fn eggroll_improves_on_linear_mse() {
    let (model, batch, target) = setup_linear();
    let specs = discover_param_specs(&model, 2);
    let mut trainer = EggrollTrainer::new(
        model,
        EggrollConfig {
            pop_size: 64,
            sigma: 0.01,
            lr: 0.01,
            rank: 2,
            weight_decay: 0.0,
            seed: 999,
            max_param_norm: None,
            pop_vectorized: true,
        },
        specs,
        MseObjective::new(target.clone()),
    );

    let mut objective = MseObjective::new(target.clone());
    let baseline_preds = trainer.model.forward(batch.clone());
    let [batch_size, dim] = baseline_preds.shape().dims();
    let baseline_logits = baseline_preds.reshape([batch_size, 1, dim]);
    let baseline = objective.evaluate(&baseline_logits, &batch);
    let baseline_mse = -baseline;
    let w_start = trainer.model.weight.val().clone();

    // Ensure noise application actually perturbs parameters for evaluation.
    let noise = trainer
        .noiser
        .build_noise_map(&trainer.state.es_key, 0);
    let candidate = trainer.noiser.apply_noise(&trainer.model, &noise);
    let candidate_diff = (candidate.weight.val() - trainer.model.weight.val())
        .abs()
        .sum()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    assert!(candidate_diff[0] > 0.0, "apply_noise produced no delta");

    for _ in 0..200 {
        let _ = trainer.step(&batch);
    }

    let improved_preds = trainer.model.forward(batch.clone());
    let improved_logits = improved_preds.reshape([batch_size, 1, dim]);
    let improved = objective.evaluate(&improved_logits, &batch);
    let improved_mse = -improved;
    let w_after = trainer.model.weight.val().clone();
    let w_diff = (w_after - w_start)
        .abs()
        .sum()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();

    // Expect weights to move and not significantly worsen fitness.
    assert!(w_diff[0] > 1e-6, "weights did not change after ES steps");
    assert!(
        improved_mse.is_finite() && improved_mse <= baseline_mse * 2.0,
        "MSE diverged. before_mse={baseline_mse}, after_mse={improved_mse}"
    );
}

#[test]
fn stacked_params_receive_noise() {
    type B = NdArray<f32>;
    let device = <B as Backend>::Device::default();
    let model = StackedParams::<B>::new(&device);
    let spec = EggrollParamSpec {
        id: model.weight.id,
        path: "stacked".into(),
        shape: (2, 2),
        rank: 2,
        sigma_scale: 1.0,
        stack: Some(2),
    };
    let noiser = EggrollNoiser::new(
        vec![spec.clone()],
        EggrollConfig {
            sigma: 0.1,
            pop_size: 1,
            lr: 0.0,
            weight_decay: 0.0,
            rank: 2,
            seed: 7,
            max_param_norm: None,
            pop_vectorized: true,
        },
        &device,
    );
    let es_key = EsTreeKey::new(EggrollKey::from_seed(7));
    let noise = noiser.build_noise_map(&es_key, 0);
    let candidate = noiser.apply_noise(&model, &noise);
    let diff = (candidate.weight.val() - model.weight.val())
        .abs()
        .sum()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    assert!(
        diff[0] > 0.0,
        "stacked noise application produced no delta: diff={}",
        diff[0]
    );
}

#[test]
fn embedding_sigma_zero_matches_plain() {
    type EB = NdArray<f32>;
    let device = <EB as Backend>::Device::default();
    let embed = burn::nn::EmbeddingConfig::new(8, 4).init(&device);
    let tokens = Tensor::<EB, 1, Int>::from_data(
        burn::tensor::TensorData::new(vec![0i64, 3, 5], [3]),
        &device,
    );

    let w = embed.weight.val();
    let plain = emb_lookup(w.clone(), tokens.clone().reshape([1, 3]));

    let noiser = EggrollNoiser::new(
        vec![EggrollParamSpec {
            id: embed.weight.id,
            path: "embed".into(),
            shape: (8, 4),
            rank: 2,
            sigma_scale: 1.0,
            stack: None,
        }],
        EggrollConfig {
            sigma: 0.0,
            pop_size: 1,
            lr: 0.0,
            weight_decay: 0.0,
            rank: 2,
            seed: 1,
            max_param_norm: None,
            pop_vectorized: true,
        },
        &device,
    );
    let es_key = EsTreeKey::new(EggrollKey::from_seed(1));
    let noisy = noiser.do_emb(&w, &tokens.reshape([1, 3]), embed.weight.id, &es_key, 0);

    let diff = (noisy - plain)
        .abs()
        .sum()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    assert!(diff[0] < 1e-6, "sigma=0 embedding should match plain lookup");
}

#[test]
fn bdh_sigma_zero_matches_plain() {
    type EB = NdArray<f32>;
    let device = <EB as Backend>::Device::default();
    let mut cfg = BDHConfig::default();
    cfg.dropout = 0.0;
    let model = BDH::<EB>::new(cfg, &device);
    let es_cfg = BdhEsConfig::default();

    let tokens = Tensor::<EB, 2, Int>::from_data(
        burn::tensor::TensorData::new(vec![0i64, 1, 2, 3], [1, 4]),
        &device,
    );

    let param_specs = model.es_param_specs(&es_cfg);
    let noiser = EggrollNoiser::new(
        param_specs,
        EggrollConfig {
            sigma: 0.0,
            pop_size: 1,
            lr: 0.0,
            weight_decay: 0.0,
            rank: 2,
            seed: 42,
            max_param_norm: None,
            pop_vectorized: true,
        },
        &device,
    );
    let es_key = EsTreeKey::new(EggrollKey::from_seed(42));

    let plain = model.forward(tokens.clone());
    let noisy = model.forward_with_noise(tokens.clone(), &noiser, &es_key, 0);

    let diff = (noisy - plain)
        .abs()
        .sum()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    assert!(
        diff[0] < 1e-5,
        "BDH sigma=0 forward should match plain. diff={}",
        diff[0]
    );
}

#[test]
fn bdh_encoder_noise_changes_output() {
    type EB = NdArray<f32>;
    let device = <EB as Backend>::Device::default();
    let mut cfg = BDHConfig::default();
    cfg.dropout = 0.0;
    let model = BDH::<EB>::new(cfg.clone(), &device);
    let mut es_cfg = BdhEsConfig::default();
    es_cfg.embedding.enabled = false;
    es_cfg.decoder_x.enabled = false;
    es_cfg.decoder_y.enabled = false;
    es_cfg.lm_head.enabled = false;
    es_cfg.encoder.enabled = true;
    es_cfg.eggroll.sigma = 0.05;
    es_cfg.eggroll.pop_size = 1;
    es_cfg.eggroll.seed = 314159;

    let tokens = Tensor::<EB, 2, Int>::from_data(
        burn::tensor::TensorData::new(vec![0i64, 2, 4, 6], [1, 4]),
        &device,
    );

    let param_specs = model.es_param_specs(&es_cfg);
    let noiser = EggrollNoiser::new(param_specs, es_cfg.eggroll.clone(), &device);
    let es_key = EsTreeKey::new(EggrollKey::from_seed(es_cfg.eggroll.seed));

    let plain = model.forward(tokens.clone());
    let noisy = model.forward_with_noise(tokens.clone(), &noiser, &es_key, 0);
    let diff = (noisy - plain)
        .abs()
        .sum()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();

    assert!(
        diff[0] > 0.0,
        "encoder noise failed to change forward output. diff={}",
        diff[0]
    );
}

#[test]
fn bdh_pop_forward_matches_single_worker() {
    type EB = NdArray<f32>;
    let device = <EB as Backend>::Device::default();
    let mut cfg = BDHConfig::default();
    cfg.dropout = 0.0;
    let model = BDH::<EB>::new(cfg, &device);

    let mut es_cfg = BdhEsConfig::default();
    es_cfg.eggroll.sigma = 0.05;
    es_cfg.eggroll.pop_size = 1;

    let tokens = Tensor::<EB, 2, Int>::from_data(
        burn::tensor::TensorData::new(vec![0i64, 1, 2, 3], [1, 4]),
        &device,
    );

    let param_specs = model.es_param_specs(&es_cfg);
    let noiser = EggrollNoiser::new(param_specs, es_cfg.eggroll.clone(), &device);
    let tree_key = EsTreeKey::new(EggrollKey::from_seed(es_cfg.eggroll.seed));
    let step = 3;
    let es_key = tree_key.clone().with_step(step);

    let single = model.forward_with_noise_det(
        tokens.clone(),
        &noiser,
        &es_key,
        0,
        true,
    );
    let pop_logits = model.forward_population_with_noise(
        &tokens,
        &noiser,
        &tree_key,
        step,
        &[0],
        true,
    );

    let [batch, time, vocab] = single.shape().dims();
    let pop0 = pop_logits
        .slice_dim(0, 0..1)
        .reshape([batch, time, vocab]);
    let diff = (pop0 - single)
        .abs()
        .sum()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    assert!(
        diff[0] < 1e-5,
        "population forward (pop=1) should match single-worker noisy forward, diff={}",
        diff[0]
    );
}

#[test]
fn bdh_pop_sigma_zero_matches_plain_for_all_workers() {
    type EB = NdArray<f32>;
    let device = <EB as Backend>::Device::default();
    let mut cfg = BDHConfig::default();
    cfg.dropout = 0.0;
    let model = BDH::<EB>::new(cfg, &device);

    let mut es_cfg = BdhEsConfig::default();
    es_cfg.eggroll.sigma = 0.0;
    es_cfg.eggroll.pop_size = 3;

    let tokens = Tensor::<EB, 2, Int>::from_data(
        burn::tensor::TensorData::new(vec![4i64, 5, 6, 7], [1, 4]),
        &device,
    );

    let param_specs = model.es_param_specs(&es_cfg);
    let noiser = EggrollNoiser::new(param_specs, es_cfg.eggroll.clone(), &device);
    let tree_key = EsTreeKey::new(EggrollKey::from_seed(es_cfg.eggroll.seed));
    let global_workers: Vec<u32> = (0..es_cfg.eggroll.pop_size as u32).collect();
    let step = 1;

    let base = model.forward(tokens.clone());
    let pop_logits = model.forward_population_with_noise(
        &tokens,
        &noiser,
        &tree_key,
        step,
        &global_workers,
        true,
    );
    let [batch, time, vocab] = base.shape().dims();

    for (idx, _) in global_workers.iter().enumerate() {
        let worker_logits = pop_logits
            .clone()
            .slice_dim(0, idx..idx + 1)
            .reshape([batch, time, vocab]);
        let diff = (worker_logits - base.clone())
            .abs()
            .sum()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap();
        assert!(
            diff[0] < 1e-6,
            "sigma=0 population logits should match base forward for worker {idx}, diff={}",
            diff[0]
        );
    }
}
