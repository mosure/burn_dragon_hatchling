use std::cmp::{Ordering, max, min};
use std::collections::{HashMap, HashSet};

use super::frame::{LayerFrame, TokenFrame};

const LOCAL_THRESHOLD_DIVISOR: u32 = 24;
const MAX_PLASTIC_POINTS: usize = 128;
const MAX_ACTIVITY_SAMPLES: usize = 48;

#[derive(Debug, Clone)]
pub struct PlasticPoint {
    pub i: u32,
    pub j: u32,
    pub delta: f32,
    pub value: f32,
}

#[derive(Debug, Clone)]
pub struct EdgePoint {
    pub from: u32,
    pub to: u32,
    pub is_attention: bool,
}

#[derive(Debug, Clone)]
pub struct LayerActivity {
    pub layer_id: u8,
    pub hot_count: usize,
    pub node_count: usize,
    pub graph_edge_count: usize,
    pub syn_edge_count: usize,
    pub node_span_min: u32,
    pub node_span_max: u32,
    pub top_hubs: Vec<(u32, usize)>,
    pub degree_spectrum: Vec<(u32, usize)>,
    pub max_degree: usize,
    pub hub_ratio: f32,
    pub local_edges: usize,
    pub local_fraction: f32,
    pub local_threshold: u32,
    pub sum_norm_gap: f32,
    pub mean_norm_gap: f32,
    pub span_i_min: u32,
    pub span_i_max: u32,
    pub span_j_min: u32,
    pub span_j_max: u32,
    pub delta_pos: usize,
    pub delta_neg: usize,
    pub delta_samples: usize,
    pub sum_abs_delta: f32,
    pub mean_abs_delta: f32,
    pub value_pos: usize,
    pub value_neg: usize,
    pub max_delta_abs: f32,
    pub graph_points: Vec<EdgePoint>,
    pub plastic_points: Vec<PlasticPoint>,
    pub activation_density: f32,
    pub activation_energy: f32,
    pub max_hot_activation: f32,
    pub hot_samples: Vec<(u32, f32)>,
    pub synapse_density: f32,
}

#[derive(Debug, Clone, Default)]
pub struct FrameTotals {
    pub nodes: usize,
    pub edges: usize,
    pub hot_neurons: usize,
    pub max_hub_degree: usize,
    pub max_hub_ratio: f32,
    pub local_edges: usize,
    pub sum_norm_gap: f32,
    pub delta_pos: usize,
    pub delta_neg: usize,
    pub value_pos: usize,
    pub value_neg: usize,
    pub sum_abs_delta: f32,
    pub delta_samples: usize,
    pub activation_density_sum: f32,
    pub activation_energy_sum: f32,
    pub synapse_density_sum: f32,
    pub layer_count: usize,
}

impl FrameTotals {
    pub fn from_layers(layers: &[LayerActivity]) -> Self {
        let mut totals = FrameTotals::default();
        for layer in layers {
            totals.nodes += layer.node_count;
            totals.edges += layer.graph_edge_count;
            totals.hot_neurons += layer.hot_count;
            totals.max_hub_degree = max(totals.max_hub_degree, layer.max_degree);
            totals.max_hub_ratio = totals.max_hub_ratio.max(layer.hub_ratio);
            totals.local_edges += layer.local_edges;
            totals.sum_norm_gap += layer.sum_norm_gap;
            totals.delta_pos += layer.delta_pos;
            totals.delta_neg += layer.delta_neg;
            totals.value_pos += layer.value_pos;
            totals.value_neg += layer.value_neg;
            totals.sum_abs_delta += layer.sum_abs_delta;
            totals.delta_samples += layer.delta_samples;
            totals.activation_density_sum += layer.activation_density;
            totals.activation_energy_sum += layer.activation_energy;
            totals.synapse_density_sum += layer.synapse_density;
            totals.layer_count += 1;
        }
        totals
    }

    pub fn local_fraction(&self) -> f32 {
        if self.edges == 0 {
            0.0
        } else {
            (self.local_edges as f32 / self.edges as f32).clamp(0.0, 1.0)
        }
    }

    pub fn mean_norm_gap(&self) -> f32 {
        if self.edges == 0 {
            0.0
        } else {
            self.sum_norm_gap / self.edges as f32
        }
    }

    pub fn active_fraction(&self) -> f32 {
        if self.nodes == 0 {
            0.0
        } else {
            (self.hot_neurons as f32 / self.nodes as f32).clamp(0.0, 1.0)
        }
    }

    pub fn mean_abs_delta(&self) -> f32 {
        if self.delta_samples == 0 {
            0.0
        } else {
            self.sum_abs_delta / self.delta_samples as f32
        }
    }

    pub fn mean_activation_density(&self) -> f32 {
        if self.layer_count == 0 {
            0.0
        } else {
            (self.activation_density_sum / self.layer_count as f32).clamp(0.0, 1.0)
        }
    }

    pub fn mean_activation_energy(&self) -> f32 {
        if self.layer_count == 0 {
            0.0
        } else {
            (self.activation_energy_sum / self.layer_count as f32).max(0.0)
        }
    }

    pub fn mean_synapse_density(&self) -> f32 {
        if self.layer_count == 0 {
            0.0
        } else {
            (self.synapse_density_sum / self.layer_count as f32).clamp(0.0, 1.0)
        }
    }
}

#[derive(Debug, Clone)]
pub struct FrameActivity {
    pub layers: Vec<LayerActivity>,
    pub totals: FrameTotals,
}

impl FrameActivity {
    pub fn new(layers: Vec<LayerActivity>) -> Self {
        let totals = FrameTotals::from_layers(&layers);
        Self { layers, totals }
    }
}

#[derive(Default)]
pub struct ActivityComputer {
    prev_synapse_values: HashMap<(u8, u32, u32), f32>,
}

impl ActivityComputer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn compute(&mut self, frame: &TokenFrame) -> FrameActivity {
        let layers = frame
            .layers
            .iter()
            .map(|layer| self.collect_layer_metrics(layer.layer, layer))
            .collect::<Vec<_>>();
        FrameActivity::new(layers)
    }

    fn collect_layer_metrics(&mut self, layer_id: u8, layer: &LayerFrame) -> LayerActivity {
        let mut degree_map: HashMap<u32, usize> = HashMap::new();
        let mut node_ids: HashSet<u32> = HashSet::new();
        let mut graph_points: Vec<EdgePoint> = Vec::new();
        let mut hot_samples = Vec::with_capacity(layer.hot_neurons.len().min(MAX_ACTIVITY_SAMPLES));
        let mut activation_sum = 0.0_f32;
        let mut max_hot_activation = 0.0_f32;

        let mut node_span_min = u32::MAX;
        let mut node_span_max = 0u32;
        let mut span_i_min = u32::MAX;
        let mut span_i_max = 0u32;
        let mut span_j_min = u32::MAX;
        let mut span_j_max = 0u32;

        for edge in &layer.attn_edges {
            node_ids.insert(edge.from);
            node_ids.insert(edge.to);
            *degree_map.entry(edge.from).or_default() += 1;
            *degree_map.entry(edge.to).or_default() += 1;
            node_span_min = min(node_span_min, min(edge.from, edge.to));
            node_span_max = max(node_span_max, max(edge.from, edge.to));
            graph_points.push(EdgePoint {
                from: edge.from,
                to: edge.to,
                is_attention: true,
            });
        }

        for edge in &layer.syn_edges {
            node_ids.insert(edge.i);
            node_ids.insert(edge.j);
            *degree_map.entry(edge.i).or_default() += 1;
            *degree_map.entry(edge.j).or_default() += 1;
            node_span_min = min(node_span_min, min(edge.i, edge.j));
            node_span_max = max(node_span_max, max(edge.i, edge.j));
            span_i_min = min(span_i_min, edge.i);
            span_i_max = max(span_i_max, edge.i);
            span_j_min = min(span_j_min, edge.j);
            span_j_max = max(span_j_max, edge.j);
            graph_points.push(EdgePoint {
                from: edge.i,
                to: edge.j,
                is_attention: false,
            });
        }

        for neuron in &layer.hot_neurons {
            node_ids.insert(neuron.id_or_cluster);
            node_span_min = min(node_span_min, neuron.id_or_cluster);
            node_span_max = max(node_span_max, neuron.id_or_cluster);
            if hot_samples.len() < MAX_ACTIVITY_SAMPLES {
                hot_samples.push((neuron.id_or_cluster, neuron.act));
            }
            activation_sum += neuron.act;
            max_hot_activation = max_hot_activation.max(neuron.act);
        }

        if node_ids.is_empty() {
            node_span_min = 0;
            node_span_max = 1;
        }

        if span_i_min == u32::MAX {
            span_i_min = node_span_min;
            span_i_max = max(node_span_max, node_span_min + 1);
        }
        if span_j_min == u32::MAX {
            span_j_min = node_span_min;
            span_j_max = max(node_span_max, node_span_min + 1);
        }

        let span_range = node_span_max.saturating_sub(node_span_min).max(1);
        let span_range_f = span_range as f32;
        let local_threshold = (span_range_f / LOCAL_THRESHOLD_DIVISOR as f32)
            .max(1.0)
            .round() as u32;

        let mut local_edges = 0usize;
        let mut sum_norm_gap = 0.0_f32;

        for point in &graph_points {
            let diff = u32::abs_diff(point.from, point.to);
            if diff <= local_threshold {
                local_edges = local_edges.saturating_add(1);
            }
            let norm = (diff as f32 / span_range_f).clamp(0.0, 1.0);
            sum_norm_gap += norm;
        }

        let mut degree_values: Vec<usize> = degree_map.values().copied().collect();
        degree_values.sort_unstable();
        let max_degree = degree_values.last().copied().unwrap_or(0);
        let median_degree = median_from_sorted(&degree_values);
        let hub_ratio = if median_degree > 0.0 {
            max_degree as f32 / median_degree
        } else {
            max_degree as f32
        };

        let mut degree_entries: Vec<(u32, usize)> = degree_map.into_iter().collect();
        let mut top_hubs = degree_entries.clone();
        top_hubs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        top_hubs.truncate(6);
        degree_entries.sort_by(|a, b| a.0.cmp(&b.0));

        let stats_edge_count = graph_points.len();
        let local_fraction = if stats_edge_count == 0 {
            0.0
        } else {
            (local_edges as f32 / stats_edge_count as f32).clamp(0.0, 1.0)
        };

        let mean_norm_gap = if stats_edge_count == 0 {
            0.0
        } else {
            sum_norm_gap / stats_edge_count as f32
        };

        let activation_density = if node_ids.is_empty() {
            0.0
        } else {
            (layer.hot_neurons.len() as f32 / node_ids.len() as f32).clamp(0.0, 1.0)
        };
        let activation_energy = if layer.hot_neurons.is_empty() {
            0.0
        } else {
            activation_sum / layer.hot_neurons.len() as f32
        };
        let synapse_capacity =
            ((span_i_max - span_i_min).max(1) as f32) * ((span_j_max - span_j_min).max(1) as f32);
        let synapse_density = if synapse_capacity <= f32::EPSILON {
            0.0
        } else {
            (layer.syn_edges.len() as f32 / synapse_capacity).clamp(0.0, 1.0)
        };

        let mut delta_pos = 0usize;
        let mut delta_neg = 0usize;
        let mut delta_samples = 0usize;
        let mut sum_abs_delta = 0.0_f32;
        let mut max_delta_abs = 0.0_f32;
        let mut value_pos = 0usize;
        let mut value_neg = 0usize;
        let mut plastic_points = Vec::with_capacity(layer.syn_edges.len());

        let current_pairs: HashSet<(u32, u32)> =
            layer.syn_edges.iter().map(|e| (e.i, e.j)).collect();

        for edge in &layer.syn_edges {
            let key = (layer_id, edge.i, edge.j);
            let previous = self.prev_synapse_values.insert(key, edge.value);
            let delta = previous.map(|p| edge.value - p).unwrap_or(0.0);
            if previous.is_some() {
                if delta.is_sign_positive() {
                    delta_pos = delta_pos.saturating_add(1);
                } else if delta.is_sign_negative() {
                    delta_neg = delta_neg.saturating_add(1);
                }
                sum_abs_delta += delta.abs();
                delta_samples = delta_samples.saturating_add(1);
                max_delta_abs = max_delta_abs.max(delta.abs());
            }
            if edge.value.is_sign_positive() {
                value_pos = value_pos.saturating_add(1);
            } else if edge.value.is_sign_negative() {
                value_neg = value_neg.saturating_add(1);
            }
            plastic_points.push(PlasticPoint {
                i: edge.i,
                j: edge.j,
                delta,
                value: edge.value,
            });
        }

        self.prev_synapse_values
            .retain(|(lay, i, j), _| *lay != layer_id || current_pairs.contains(&(*i, *j)));

        plastic_points.sort_by(|a, b| {
            b.delta
                .abs()
                .partial_cmp(&a.delta.abs())
                .unwrap_or(Ordering::Equal)
        });
        if plastic_points.len() > MAX_PLASTIC_POINTS {
            plastic_points.truncate(MAX_PLASTIC_POINTS);
        }

        let mean_abs_delta = if delta_samples == 0 {
            0.0
        } else {
            sum_abs_delta / delta_samples as f32
        };

        LayerActivity {
            layer_id,
            hot_count: layer.hot_neurons.len(),
            node_count: node_ids.len(),
            graph_edge_count: graph_points.len(),
            syn_edge_count: layer.syn_edges.len(),
            node_span_min,
            node_span_max,
            top_hubs,
            degree_spectrum: degree_entries,
            max_degree,
            hub_ratio,
            local_edges,
            local_fraction,
            local_threshold,
            sum_norm_gap,
            mean_norm_gap,
            span_i_min,
            span_i_max,
            span_j_min,
            span_j_max,
            delta_pos,
            delta_neg,
            delta_samples,
            sum_abs_delta,
            mean_abs_delta,
            value_pos,
            value_neg,
            max_delta_abs,
            graph_points,
            plastic_points,
            activation_density,
            activation_energy,
            max_hot_activation,
            hot_samples,
            synapse_density,
        }
    }
}

fn median_from_sorted(values: &[usize]) -> f32 {
    let len = values.len();
    if len == 0 {
        0.0
    } else if len % 2 == 1 {
        values[len / 2] as f32
    } else {
        (values[len / 2 - 1] + values[len / 2]) as f32 / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viz::frame::{AttnEdge, HotNeuron, LayerFrame, SynEdge, TokenFrame};

    fn sample_layer() -> LayerFrame {
        LayerFrame {
            layer: 0,
            attn_edges: vec![
                AttnEdge {
                    from: 1,
                    to: 2,
                    head: 0,
                    w: 0.9,
                },
                AttnEdge {
                    from: 2,
                    to: 3,
                    head: 1,
                    w: 0.4,
                },
            ],
            hot_neurons: vec![
                HotNeuron {
                    id_or_cluster: 10,
                    act: 0.8,
                },
                HotNeuron {
                    id_or_cluster: 12,
                    act: 0.5,
                },
            ],
            syn_edges: vec![
                SynEdge {
                    i: 1,
                    j: 4,
                    delta: 0.0,
                    value: 0.2,
                },
                SynEdge {
                    i: 5,
                    j: 9,
                    delta: 0.0,
                    value: -0.3,
                },
            ],
            attn_entropy: 0.0,
        }
    }

    #[test]
    fn collects_activity_and_updates_totals() {
        let mut computer = ActivityComputer::new();
        let frame = TokenFrame {
            t: 1,
            token_id: 42,
            token_text: "dragon".into(),
            layers: vec![sample_layer()],
        };

        let snapshot = computer.compute(&frame);

        assert_eq!(snapshot.layers.len(), 1);
        let layer = &snapshot.layers[0];
        assert_eq!(layer.layer_id, 0);
        assert_eq!(layer.hot_count, 2);
        assert_eq!(layer.syn_edge_count, 2);
        assert_eq!(layer.graph_edge_count, 4);
        assert!(layer.hub_ratio >= 1.0);
        assert!(layer.activation_density > 0.0);
        assert!(!layer.hot_samples.is_empty());
        assert!(layer.max_hot_activation > 0.0);

        assert_eq!(snapshot.totals.nodes, layer.node_count);
        assert_eq!(snapshot.totals.edges, layer.graph_edge_count);
        assert!(snapshot.totals.max_hub_degree >= layer.max_degree);
    }

    #[test]
    fn tracks_plasticity_deltas_over_time() {
        let mut computer = ActivityComputer::new();
        let mut base_layer = sample_layer();
        let mut frame = TokenFrame {
            t: 1,
            token_id: 1,
            token_text: "a".into(),
            layers: vec![base_layer.clone()],
        };

        computer.compute(&frame);

        base_layer.syn_edges[0].value = 0.8;
        base_layer.syn_edges[1].value = -0.6;
        frame.layers[0] = base_layer;
        let snapshot = computer.compute(&frame);
        let layer = &snapshot.layers[0];

        assert_eq!(layer.delta_samples, 2);
        assert_eq!(layer.delta_pos, 1);
        assert_eq!(layer.delta_neg, 1);
        assert!(layer.mean_abs_delta > 0.0);
        assert!(layer.synapse_density > 0.0);
        assert!(layer.activation_energy > 0.0);
        assert_eq!(snapshot.totals.delta_samples, 2);
    }

    #[test]
    fn locality_metrics_reflect_edge_distance() {
        let layer = LayerFrame {
            layer: 0,
            attn_edges: vec![
                AttnEdge {
                    from: 0,
                    to: 1,
                    head: 0,
                    w: 0.2,
                },
                AttnEdge {
                    from: 0,
                    to: 6,
                    head: 1,
                    w: 0.2,
                },
            ],
            hot_neurons: Vec::new(),
            syn_edges: Vec::new(),
            attn_entropy: 0.0,
        };
        let frame = TokenFrame {
            t: 0,
            token_id: 0,
            token_text: "edge".into(),
            layers: vec![layer],
        };
        let mut computer = ActivityComputer::new();
        let snapshot = computer.compute(&frame);
        let metrics = &snapshot.layers[0];
        assert_eq!(metrics.graph_edge_count, 2);
        assert_eq!(
            metrics.local_edges, 1,
            "only the nearest edge should count as local"
        );
        assert!(metrics.local_fraction > 0.0 && metrics.local_fraction < 1.0);
        assert!(snapshot.totals.local_fraction() > 0.0);
    }
}
