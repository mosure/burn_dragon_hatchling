use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::model::LayerVizState;

use super::frame::{VizConfig, VizFrame};
use super::palette::{COLOR_ACTIVITY, COLOR_SEPARATOR, COLOR_WRITES};

pub struct VizEncoder<B: Backend> {
    config: VizConfig,
    layers: usize,
    heads: usize,
    latent_per_head: usize,
    latent_total: usize,
    overview_activity: Tensor<B, 3>,
    overview_writes: Tensor<B, 3>,
    units_activity: Tensor<B, 3>,
    units_writes: Tensor<B, 3>,
    color_activity: Tensor<B, 3>,
    color_writes: Tensor<B, 3>,
    separator_rgba: Option<Tensor<B, 3>>,
    zero_row: Tensor<B, 3>,
    zero_units: Tensor<B, 3>,
}

impl<B: Backend> VizEncoder<B> {
    pub fn new(
        config: VizConfig,
        layers: usize,
        heads: usize,
        latent_per_head: usize,
        device: &B::Device,
    ) -> Self {
        let history = config.history.max(1);
        let latent_total = heads.saturating_mul(latent_per_head).max(1);

        let overview_activity = Tensor::<B, 3>::zeros([layers.max(1), history, 4], device);
        let overview_writes = Tensor::<B, 3>::zeros([layers.max(1), history, 4], device);
        let units_activity = Tensor::<B, 3>::zeros([latent_total, history, 4], device);
        let units_writes = Tensor::<B, 3>::zeros([latent_total, history, 4], device);

        let color_activity = color_tensor::<B>(COLOR_ACTIVITY, device);
        let color_writes = color_tensor::<B>(COLOR_WRITES, device);

        let separator_rgba = build_separator::<B>(latent_total, latent_per_head, device);
        let zero_row = Tensor::<B, 3>::zeros([1, 1, 4], device);
        let zero_units = Tensor::<B, 3>::zeros([latent_total, 1, 4], device);

        Self {
            config,
            layers,
            heads,
            latent_per_head,
            latent_total,
            overview_activity,
            overview_writes,
            units_activity,
            units_writes,
            color_activity,
            color_writes,
            separator_rgba,
            zero_row,
            zero_units,
        }
    }

    pub fn should_capture(&self, token_index: usize) -> bool {
        let stride = self.config.stride_tokens.max(1);
        token_index % stride == 0
    }

    pub fn step(
        &mut self,
        layers: &[Option<LayerVizState<B>>],
        token_index: usize,
    ) -> VizFrame<B> {
        let device = self.zero_units.device();
        let history = self.config.history.max(1);
        let cursor = token_index % history;
        let layer_count = self.layers.min(layers.len()).max(1);

        let mut activity_rows: Vec<Tensor<B, 3>> = Vec::with_capacity(layer_count);
        let mut write_rows: Vec<Tensor<B, 3>> = Vec::with_capacity(layer_count);

        for layer_idx in 0..layer_count {
            if let Some(state) = layers[layer_idx].as_ref() {
                let write_frac = state
                    .x_last
                    .clone()
                    .greater_elem(0.0)
                    .float()
                    .mean();
                let activity_frac = state
                    .xy_last
                    .clone()
                    .greater_elem(0.0)
                    .float()
                    .mean();

                write_rows.push(self.encode_scalar(write_frac, self.config.gain_x, &self.color_writes));
                activity_rows.push(self.encode_scalar(
                    activity_frac,
                    self.config.gain_xy,
                    &self.color_activity,
                ));
            } else {
                write_rows.push(self.zero_row.clone());
                activity_rows.push(self.zero_row.clone());
            }
        }

        let activity_column = Tensor::cat(activity_rows, 0);
        let writes_column = Tensor::cat(write_rows, 0);

        self.overview_activity = self.overview_activity.clone().slice_assign(
            [0..layer_count, cursor..cursor + 1, 0..4],
            activity_column,
        );
        self.overview_writes = self.overview_writes.clone().slice_assign(
            [0..layer_count, cursor..cursor + 1, 0..4],
            writes_column,
        );

        let focus = self.config.layer_focus.min(layer_count.saturating_sub(1));
        let (x_last, xy_last) = layers
            .get(focus)
            .and_then(|layer| layer.as_ref())
            .map(|layer| (layer.x_last.clone(), layer.xy_last.clone()))
            .unwrap_or_else(|| {
                (
                    Tensor::<B, 2>::zeros([self.heads, self.latent_per_head], &device),
                    Tensor::<B, 2>::zeros([self.heads, self.latent_per_head], &device),
                )
            });

        let x_flat = x_last.reshape([self.latent_total]);
        let xy_flat = xy_last.reshape([self.latent_total]);

        let units_writes_col = self.encode_units(x_flat, self.config.gain_x, &self.color_writes);
        let units_activity_col =
            self.encode_units(xy_flat, self.config.gain_xy, &self.color_activity);

        self.units_writes = self.units_writes.clone().slice_assign(
            [0..self.latent_total, cursor..cursor + 1, 0..4],
            units_writes_col,
        );
        self.units_activity = self.units_activity.clone().slice_assign(
            [0..self.latent_total, cursor..cursor + 1, 0..4],
            units_activity_col,
        );

        VizFrame {
            overview_activity: self.overview_activity.clone(),
            overview_writes: self.overview_writes.clone(),
            units_activity: self.units_activity.clone(),
            units_writes: self.units_writes.clone(),
            cursor,
            token_index,
        }
    }

    fn encode_scalar(&self, value: Tensor<B, 1>, gain: f32, color: &Tensor<B, 3>) -> Tensor<B, 3> {
        let mag = value.clone().mul_scalar(gain).clamp(0.0, 1.0);
        let mask = value.greater_elem(0.0).float();
        let intensity = mask * (mag.mul_scalar(0.75).add_scalar(0.25));
        let intensity = intensity.reshape([1, 1, 1]);
        intensity * color.clone()
    }

    fn encode_units(&self, values: Tensor<B, 1>, gain: f32, color: &Tensor<B, 3>) -> Tensor<B, 3> {
        if self.latent_total == 0 {
            return self.zero_units.clone();
        }
        let values = values.reshape([self.latent_total, 1]);
        let mag = values.clone().mul_scalar(gain).clamp(0.0, 1.0);
        let mask = values.greater_elem(0.0).float();
        let intensity = mask * (mag.mul_scalar(0.75).add_scalar(0.25));
        let intensity = intensity.reshape([self.latent_total, 1, 1]);
        let mut column = intensity * color.clone();
        if let Some(sep) = &self.separator_rgba {
            column = column.max_pair(sep.clone());
        }
        column
    }
}

fn color_tensor<B: Backend>(rgba: [f32; 4], device: &B::Device) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(TensorData::new(rgba.to_vec(), [1, 1, 4]), device)
}

fn build_separator<B: Backend>(
    latent_total: usize,
    latent_per_head: usize,
    device: &B::Device,
) -> Option<Tensor<B, 3>> {
    if latent_total == 0 || latent_per_head == 0 {
        return None;
    }
    let mut mask = vec![0.0f32; latent_total];
    for idx in 0..latent_total {
        if idx > 0 && idx % latent_per_head == 0 {
            mask[idx] = 1.0;
        }
    }
    let mask = Tensor::<B, 1>::from_data(TensorData::new(mask, [latent_total]), device)
        .reshape([latent_total, 1, 1]);
    let color = color_tensor::<B>(COLOR_SEPARATOR, device);
    Some(mask * color)
}

#[cfg(all(test, feature = "viz", feature = "cli"))]
mod tests {
    use super::{VizConfig, VizEncoder};
    use burn::tensor::{Int, Tensor, TensorData};
    use burn_ndarray::{NdArray, NdArrayDevice};

    use super::super::super::model::{BDH, BDHConfig, LayerVizState, ModelState};

    type Backend = NdArray<f32>;

    fn device() -> NdArrayDevice {
        NdArrayDevice::default()
    }

    #[test]
    fn viz_state_collects_last_token() {
        let device = device();

        let mut config = BDHConfig::default();
        config.n_layer = 2;
        config.n_embd = 8;
        config.n_head = 2;
        config.mlp_internal_dim_multiplier = 2;
        config.vocab_size = 16;
        config.dropout = 0.0;
        config.fused_kernels.enabled = false;

        let model = BDH::<Backend>::new(config.clone(), &device);
        let tokens = Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3], [1, 3]),
            &device,
        );

        let mut state = ModelState::<Backend>::new(config.n_layer);
        let _ = model.forward_with_state(tokens, &mut state);

        let layers: Vec<Option<LayerVizState<Backend>>> = state.take_viz();
        assert_eq!(layers.len(), config.n_layer);

        for layer in layers {
            let layer = layer.expect("expected viz state per layer");
            let dims = layer.x_last.shape().dims::<2>();
            assert_eq!(dims[0], config.n_head);
            assert_eq!(dims[1], config.latent_per_head());
            assert_eq!(layer.xy_last.shape().dims::<2>(), dims);
        }
    }

    #[test]
    fn viz_encoder_emits_expected_shapes() {
        let device = device();
        let config = VizConfig {
            history: 4,
            layer_focus: 0,
            stride_tokens: 1,
            gain_x: 1.0,
            gain_xy: 1.0,
        };

        let mut encoder = VizEncoder::<Backend>::new(config, 2, 2, 2, &device);

        let layer0 = LayerVizState {
            x_last: Tensor::<Backend, 2>::from_data(
                TensorData::new(vec![1.0, 0.0, 0.0, 0.0], [2, 2]),
                &device,
            ),
            xy_last: Tensor::<Backend, 2>::from_data(
                TensorData::new(vec![0.5, 0.0, 0.0, 0.0], [2, 2]),
                &device,
            ),
        };
        let layer1 = LayerVizState {
            x_last: Tensor::<Backend, 2>::zeros([2, 2], &device),
            xy_last: Tensor::<Backend, 2>::zeros([2, 2], &device),
        };

        let frame = encoder.step(&[Some(layer0), Some(layer1)], 0);

        assert_eq!(frame.overview_activity.shape().dims::<3>(), [2, 4, 4]);
        assert_eq!(frame.overview_writes.shape().dims::<3>(), [2, 4, 4]);
        assert_eq!(frame.units_activity.shape().dims::<3>(), [4, 4, 4]);
        assert_eq!(frame.units_writes.shape().dims::<3>(), [4, 4, 4]);
        assert_eq!(frame.cursor, 0);

        let overview = frame
            .overview_activity
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("overview vec");
        let layer_stride = 4 * 4;
        let layer0_sum: f32 = overview[0..4].iter().sum();
        let layer1_sum: f32 = overview[layer_stride..layer_stride + 4].iter().sum();
        assert!(layer0_sum > 0.0);
        assert_eq!(layer1_sum, 0.0);

        let units = frame
            .units_activity
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("units vec");
        let unit_stride = 4 * 4;
        let unit0_sum: f32 = units[0..4].iter().sum();
        let unit1_sum: f32 = units[unit_stride..unit_stride + 4].iter().sum();
        assert!(unit0_sum > 0.0);
        assert_eq!(unit1_sum, 0.0);
    }
}
