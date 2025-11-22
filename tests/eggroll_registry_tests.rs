use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use burn_dragon_hatchling::{BDH, BDHConfig, BdhEsConfig};

#[test]
fn bdh_es_registry_contains_expected_targets() {
    type EB = NdArray<f32>;
    let device = <EB as Backend>::Device::default();
    let model = BDH::<EB>::new(BDHConfig::default(), &device);
    let cfg = BdhEsConfig::default();
    let specs = model.es_param_specs(&cfg);

    let mut targets = std::collections::HashMap::new();
    for spec in specs {
        targets.insert(spec.path.clone(), (spec.shape, spec.rank));
    }

    assert_eq!(targets.len(), 3, "expected embedding, decoder, lm_head");
    assert!(targets.contains_key("embedding"));
    assert!(targets.contains_key("decoder"));
    assert!(targets.contains_key("lm_head"));
}
