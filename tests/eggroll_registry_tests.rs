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

    assert_eq!(targets.len(), 5, "expected embedding, decoder_x, decoder_y, encoder, lm_head");
    for name in ["embedding", "lm_head"] {
        assert!(
            targets
                .get(name)
                .and_then(|spec| spec.stack)
                .is_none(),
            "{name} should be treated as 2D"
        );
    }
    let latent = model_cfg.latent_per_head();
    let dec_x = targets.get("decoder_x").expect("missing decoder_x spec");
    assert_eq!(dec_x.stack, Some(model_cfg.n_head));
    assert_eq!(dec_x.shape, (model_cfg.n_embd, latent));

    let dec_y = targets.get("decoder_y").expect("missing decoder_y spec");
    assert_eq!(dec_y.stack, Some(model_cfg.n_head));
    assert_eq!(dec_y.shape, (model_cfg.n_embd, latent));

    let enc = targets.get("encoder").expect("missing encoder spec");
    assert_eq!(enc.stack, None);
    assert_eq!(enc.shape, (model_cfg.latent_total(), model_cfg.n_embd));
}
