use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttnEdge {
    pub from: u32,
    pub to: u32,
    pub head: u8,
    pub w: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HotNeuron {
    pub id_or_cluster: u32,
    pub act: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynEdge {
    pub i: u32,
    pub j: u32,
    pub delta: f32,
    pub value: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerFrame {
    pub layer: u8,
    pub attn_edges: Vec<AttnEdge>,
    pub hot_neurons: Vec<HotNeuron>,
    pub syn_edges: Vec<SynEdge>,
    pub attn_entropy: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenFrame {
    pub t: u32,
    pub token_id: u32,
    pub token_text: String,
    pub layers: Vec<LayerFrame>,
}
