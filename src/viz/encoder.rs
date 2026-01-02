use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::model::LayerVizState;

use super::frame::{LAYER_GAP, VizConfig, VizFrame};
use super::palette::{
    COLOR_ACTIVITY, COLOR_INTERACTION, COLOR_MEMORY, COLOR_SEPARATOR, COLOR_WRITES,
};

const ACTIVE_EPS: f32 = 1e-3;

pub struct VizEncoder<B: Backend> {
    config: VizConfig,
    layers: usize,
    heads: usize,
    latent_per_head: usize,
    latent_total: usize,
    layer_gap: usize,
    units_x: Tensor<B, 3>,
    units_y: Tensor<B, 3>,
    units_xy: Tensor<B, 3>,
    units_rho: Tensor<B, 3>,
    color_x: Tensor<B, 3>,
    color_y: Tensor<B, 3>,
    color_xy: Tensor<B, 3>,
    color_rho: Tensor<B, 3>,
    separator_rgba: Option<Tensor<B, 3>>,
    zero_units: Tensor<B, 3>,
    zero_gap: Option<Tensor<B, 3>>,
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
        let layers = layers.max(1);
        let layer_gap = LAYER_GAP;
        let units_height = latent_total
            .saturating_mul(layers)
            .saturating_add(layer_gap.saturating_mul(layers.saturating_sub(1)))
            .max(1);

        let units_x = Tensor::<B, 3>::zeros([units_height, history, 4], device);
        let units_y = Tensor::<B, 3>::zeros([units_height, history, 4], device);
        let units_xy = Tensor::<B, 3>::zeros([units_height, history, 4], device);
        let units_rho = Tensor::<B, 3>::zeros([units_height, history, 4], device);

        let color_x = color_tensor::<B>(COLOR_WRITES, device);
        let color_y = color_tensor::<B>(COLOR_ACTIVITY, device);
        let color_xy = color_tensor::<B>(COLOR_INTERACTION, device);
        let color_rho = color_tensor::<B>(COLOR_MEMORY, device);

        let separator_rgba = build_separator::<B>(latent_total, latent_per_head, device);
        let zero_units = Tensor::<B, 3>::zeros([latent_total, 1, 4], device);
        let zero_gap = if layer_gap > 0 {
            Some(Tensor::<B, 3>::zeros([layer_gap, 1, 4], device))
        } else {
            None
        };

        Self {
            config,
            layers,
            heads,
            latent_per_head,
            latent_total,
            layer_gap,
            units_x,
            units_y,
            units_xy,
            units_rho,
            color_x,
            color_y,
            color_xy,
            color_rho,
            separator_rgba,
            zero_units,
            zero_gap,
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

        for layer_idx in 0..layer_count {
            let offset = layer_idx
                .saturating_mul(self.latent_total.saturating_add(self.layer_gap));
            if let Some(gap) = &self.zero_gap {
                if layer_idx > 0 {
                    let gap_start = offset.saturating_sub(self.layer_gap);
                    let gap_end = offset;
                    self.units_x = self.units_x.clone().slice_assign(
                        [gap_start..gap_end, cursor..cursor + 1, 0..4],
                        gap.clone(),
                    );
                    self.units_y = self.units_y.clone().slice_assign(
                        [gap_start..gap_end, cursor..cursor + 1, 0..4],
                        gap.clone(),
                    );
                    self.units_xy = self.units_xy.clone().slice_assign(
                        [gap_start..gap_end, cursor..cursor + 1, 0..4],
                        gap.clone(),
                    );
                    self.units_rho = self.units_rho.clone().slice_assign(
                        [gap_start..gap_end, cursor..cursor + 1, 0..4],
                        gap.clone(),
                    );
                }
            }

            let (x_last, y_last, xy_last, rho_last) = layers
                .get(layer_idx)
                .and_then(|layer| layer.as_ref())
                .map(|layer| {
                    (
                        layer.x_last.clone(),
                        layer.y_last.clone(),
                        layer.xy_last.clone(),
                        layer.rho_last.clone(),
                    )
                })
                .unwrap_or_else(|| {
                    (
                        Tensor::<B, 2>::zeros([self.heads, self.latent_per_head], &device),
                        Tensor::<B, 2>::zeros([self.heads, self.latent_per_head], &device),
                        Tensor::<B, 2>::zeros([self.heads, self.latent_per_head], &device),
                        Tensor::<B, 2>::zeros([self.heads, self.latent_per_head], &device),
                    )
                });

            let x_flat = x_last.reshape([self.latent_total]);
            let y_flat = y_last.reshape([self.latent_total]);
            let xy_flat = xy_last.reshape([self.latent_total]);
            let rho_flat = rho_last.reshape([self.latent_total]);

            let units_x_col = self.encode_units(x_flat, self.config.gain_x, &self.color_x);
            let units_y_col = self.encode_units(y_flat, self.config.gain_xy, &self.color_y);
            let units_xy_col = self.encode_units(xy_flat, self.config.gain_xy, &self.color_xy);
            let units_rho_col = self.encode_units(rho_flat, self.config.gain_xy, &self.color_rho);

            let range = offset..offset + self.latent_total;
            self.units_x = self.units_x.clone().slice_assign(
                [range.clone(), cursor..cursor + 1, 0..4],
                units_x_col,
            );
            self.units_y = self.units_y.clone().slice_assign(
                [range.clone(), cursor..cursor + 1, 0..4],
                units_y_col,
            );
            self.units_xy = self.units_xy.clone().slice_assign(
                [range.clone(), cursor..cursor + 1, 0..4],
                units_xy_col,
            );
            self.units_rho = self.units_rho.clone().slice_assign(
                [range, cursor..cursor + 1, 0..4],
                units_rho_col,
            );
        }

        VizFrame {
            units_x: self.units_x.clone(),
            units_y: self.units_y.clone(),
            units_xy: self.units_xy.clone(),
            units_rho: self.units_rho.clone(),
            cursor,
            token_index,
        }
    }

    fn encode_units(&self, values: Tensor<B, 1>, gain: f32, color: &Tensor<B, 3>) -> Tensor<B, 3> {
        if self.latent_total == 0 {
            return self.zero_units.clone();
        }
        let values = values.clamp_min(0.0).reshape([self.latent_total, 1]);
        let mag = values
            .clone()
            .mul_scalar(gain)
            .div(values.clone().mul_scalar(gain).add_scalar(1.0))
            .powf_scalar(0.5);
        let mask = values.clone().div(values.clone().add_scalar(ACTIVE_EPS * 4.0));
        let intensity = mag * mask;
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
            assert_eq!(layer.y_last.shape().dims::<2>(), dims);
            assert_eq!(layer.xy_last.shape().dims::<2>(), dims);
            assert_eq!(layer.rho_last.shape().dims::<2>(), dims);
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
            y_last: Tensor::<Backend, 2>::from_data(
                TensorData::new(vec![0.0, 1.0, 0.0, 0.0], [2, 2]),
                &device,
            ),
            xy_last: Tensor::<Backend, 2>::from_data(
                TensorData::new(vec![0.5, 0.0, 0.0, 0.0], [2, 2]),
                &device,
            ),
            rho_last: Tensor::<Backend, 2>::from_data(
                TensorData::new(vec![0.2, 0.0, 0.0, 0.0], [2, 2]),
                &device,
            ),
        };
        let layer1 = LayerVizState {
            x_last: Tensor::<Backend, 2>::zeros([2, 2], &device),
            y_last: Tensor::<Backend, 2>::zeros([2, 2], &device),
            xy_last: Tensor::<Backend, 2>::zeros([2, 2], &device),
            rho_last: Tensor::<Backend, 2>::zeros([2, 2], &device),
        };

        let frame = encoder.step(&[Some(layer0), Some(layer1)], 0);

        assert_eq!(frame.units_x.shape().dims::<3>(), [11, 4, 4]);
        assert_eq!(frame.units_y.shape().dims::<3>(), [11, 4, 4]);
        assert_eq!(frame.units_xy.shape().dims::<3>(), [11, 4, 4]);
        assert_eq!(frame.units_rho.shape().dims::<3>(), [11, 4, 4]);
        assert_eq!(frame.cursor, 0);

        let units_x = frame
            .units_x
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("units x vec");
        let units_y = frame
            .units_y
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("units y vec");
        let units_xy = frame
            .units_xy
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("units xy vec");
        let units_rho = frame
            .units_rho
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("units rho vec");
        let unit_stride = 4 * 4;
        let unit0_x_sum: f32 = units_x[0..4].iter().sum();
        let unit1_y_sum: f32 = units_y[unit_stride..unit_stride + 4].iter().sum();
        let unit0_xy_sum: f32 = units_xy[0..4].iter().sum();
        let unit0_rho_sum: f32 = units_rho[0..4].iter().sum();
        assert!(unit0_x_sum > 0.0);
        assert!(unit1_y_sum > 0.0);
        assert!(unit0_xy_sum > 0.0);
        assert!(unit0_rho_sum > 0.0);
    }
}
