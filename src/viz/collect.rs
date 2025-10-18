use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use super::frame::{AttnEdge, HotNeuron, LayerFrame, SynEdge, TokenFrame};

pub struct HeadRow<B: Backend> {
    pub w: Tensor<B, 1>,
    pub head: u8,
}

pub struct LayerTap<B: Backend> {
    pub layer: u8,
    pub attn_rows: Vec<HeadRow<B>>,
    pub neurons: Tensor<B, 1>,
    pub syn_readout: Box<dyn Fn(&[(u32, u32)]) -> Vec<(u32, u32, f32)> + Send + Sync>,
}

pub struct StepTaps<B: Backend> {
    pub t: u32,
    pub token_id: u32,
    pub token_text: String,
    pub layers: Vec<LayerTap<B>>,
}

pub struct CollectKnobs {
    pub k_attn: usize,
    pub k_neuron: usize,
    pub probes: Vec<(u8, Vec<(u32, u32)>)>,
}

pub fn to_frame<B: Backend>(taps: StepTaps<B>, knobs: &CollectKnobs) -> TokenFrame {
    let mut layers = Vec::with_capacity(taps.layers.len());

    for layer in taps.layers {
        let mut attn_edges = Vec::new();
        for hr in layer.attn_rows {
            let (vals, idxs) = hr.w.clone().topk_with_indices(knobs.k_attn, 0);
            let vals = vals
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .unwrap_or_else(|_| Vec::new());
            let idxs: Vec<i32> = idxs
                .to_data()
                .convert::<i32>()
                .into_vec::<i32>()
                .unwrap_or_else(|_| Vec::new());
            for (weight, target) in vals.into_iter().zip(idxs.into_iter()) {
                attn_edges.push(AttnEdge {
                    from: taps.t,
                    to: target.max(0) as u32,
                    head: hr.head,
                    w: weight,
                });
            }
        }

        let (act_vals, act_idxs) = layer.neurons.clone().topk_with_indices(knobs.k_neuron, 0);
        let act_vals: Vec<f32> = act_vals
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap_or_else(|_| Vec::new());
        let act_idxs: Vec<i32> = act_idxs
            .to_data()
            .convert::<i32>()
            .into_vec::<i32>()
            .unwrap_or_else(|_| Vec::new());
        let hot_neurons = act_vals
            .into_iter()
            .zip(act_idxs.into_iter().map(|idx| idx.max(0) as u32))
            .map(|(act, idx)| HotNeuron {
                id_or_cluster: idx,
                act,
            })
            .collect::<Vec<_>>();

        let syn_pairs = knobs
            .probes
            .iter()
            .find(|(l, _)| *l == layer.layer)
            .map(|(_, pairs)| pairs.as_slice())
            .unwrap_or_default();
        let syn_edges = if syn_pairs.is_empty() {
            Vec::new()
        } else {
            (layer.syn_readout)(syn_pairs)
                .into_iter()
                .map(|(i, j, value)| SynEdge {
                    i,
                    j,
                    delta: 0.0,
                    value,
                })
                .collect()
        };

        layers.push(LayerFrame {
            layer: layer.layer,
            attn_edges,
            hot_neurons,
            syn_edges,
            attn_entropy: 0.0,
        });
    }

    TokenFrame {
        t: taps.t,
        token_id: taps.token_id,
        token_text: taps.token_text,
        layers,
    }
}

#[cfg(all(test, feature = "viz"))]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn to_frame_collects_expected_topk_values() {
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();
        let attn = Tensor::<TestBackend, 1>::from_floats([0.1, 0.7, 0.2], &device);
        let neurons = Tensor::<TestBackend, 1>::from_floats([0.05, 0.9, 0.4], &device);

        let syn_values = vec![(1_u32, 3_u32, 0.33_f32)];
        let syn_values_clone = syn_values.clone();

        let layer = LayerTap {
            layer: 0,
            attn_rows: vec![HeadRow { w: attn, head: 2 }],
            neurons,
            syn_readout: Box::new(move |pairs| {
                assert_eq!(pairs, &[(1, 3)]);
                syn_values_clone.clone()
            }),
        };

        let taps = StepTaps {
            t: 5,
            token_id: 42,
            token_text: "dragon".to_string(),
            layers: vec![layer],
        };

        let knobs = CollectKnobs {
            k_attn: 2,
            k_neuron: 2,
            probes: vec![(0, vec![(1, 3)])],
        };

        let frame = to_frame(taps, &knobs);
        assert_eq!(frame.t, 5);
        assert_eq!(frame.token_id, 42);
        assert_eq!(frame.layers.len(), 1);
        let layer = &frame.layers[0];
        assert_eq!(layer.layer, 0);
        assert_eq!(layer.attn_edges.len(), 2);
        assert_eq!(layer.attn_edges[0].head, 2);
        assert_eq!(layer.attn_edges[0].to, 1);
        assert!(
            (layer.attn_edges[0].w - 0.7).abs() < 1e-6,
            "expected top attention weight, got {}",
            layer.attn_edges[0].w
        );
        assert_eq!(layer.hot_neurons.len(), 2);
        assert_eq!(layer.hot_neurons[0].id_or_cluster, 1);
        assert!(
            (layer.hot_neurons[0].act - 0.9).abs() < 1e-6,
            "expected hottest neuron activation, got {}",
            layer.hot_neurons[0].act
        );
        assert_eq!(layer.syn_edges.len(), 1);
        let syn_edge = &layer.syn_edges[0];
        assert_eq!(syn_edge.i, 1);
        assert_eq!(syn_edge.j, 3);
        assert!(
            (syn_edge.value - 0.33).abs() < 1e-6,
            "expected synapse value, got {}",
            syn_edge.value
        );
    }
}
