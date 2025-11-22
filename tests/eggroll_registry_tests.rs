use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use burn_dragon_hatchling::{BDH, BDHConfig, BdhEsConfig};

#[test]
fn bdh_es_registry_contains_expected_targets() {
    type EB = NdArray<f32>;
    let device = <EB as Backend>::Device::default();
    let model_cfg = BDHConfig::default();
    let model = BDH::<EB>::new(model_cfg.clone(), &device);
    let es_cfg = BdhEsConfig::default();
    let specs = model.es_param_specs(&es_cfg);

    let mut targets = std::collections::HashMap::new();
    for spec in specs {
        targets.insert(spec.path.clone(), spec);
    }

    assert_eq!(
        targets.len(),
        5,
        "expected embedding, decoder, lm_head, encoder, encoder_v"
    );
    for name in ["embedding", "decoder", "lm_head"] {
        assert!(
            targets
                .get(name)
                .and_then(|spec| spec.stack)
                .is_none(),
            "{name} should be treated as 2D"
        );
    }
    let latent = model_cfg.latent_per_head();
    let enc = targets
        .get("encoder")
        .expect("missing encoder spec");
    assert_eq!(enc.stack, Some(model_cfg.n_head));
    assert_eq!(enc.shape, (model_cfg.n_embd, latent));

    let enc_v = targets
        .get("encoder_v")
        .expect("missing encoder_v spec");
    assert_eq!(enc_v.stack, Some(model_cfg.n_head));
    assert_eq!(enc_v.shape, (model_cfg.n_embd, latent));
}
