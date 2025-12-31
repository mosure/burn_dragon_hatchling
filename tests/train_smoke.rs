use std::path::PathBuf;

use burn::tensor::{Int, Tensor, TensorData};
use burn::tensor::backend::Backend;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

use burn_dragon_hatchling::{BDH, build_model_config, language_model_loss, load_training_config};

type TrainBackend = Autodiff<NdArray<f32>>;

#[test]
fn training_forward_backward_from_configs() {
    let config_paths = [
        PathBuf::from("config/base.toml"),
        PathBuf::from("config/small.toml"),
    ];
    let config = load_training_config(&config_paths).expect("load training config");
    let mut model_config = build_model_config(&config.model, config.training.block_size);
    model_config.vocab_size = 64;

    let device = <TrainBackend as Backend>::Device::default();
    let model = BDH::<TrainBackend>::new(model_config, &device);

    let batch = 2;
    let time = 4;
    let inputs = Tensor::<TrainBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 2, 3, 4, 5, 6, 7], [batch, time]),
        &device,
    );
    let targets = Tensor::<TrainBackend, 2, Int>::from_data(
        TensorData::new(vec![1, 2, 3, 4, 5, 6, 7, 0], [batch, time]),
        &device,
    );

    let logits = model.forward(inputs.clone());
    let loss = language_model_loss::<TrainBackend>(logits, targets.clone());
    let _ = loss.backward();

    let logits_fast = model.forward_fast(inputs.clone());
    let loss_fast = language_model_loss::<TrainBackend>(logits_fast, targets);
    let _ = loss_fast.backward();
}
