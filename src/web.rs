use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use burn::module::Module;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, HalfPrecisionSettings, Recorder};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_wgpu::{RuntimeOptions, graphics};
use serde::Deserialize;
use wasm_bindgen::prelude::*;
#[cfg(feature = "viz")]
use wasm_bindgen::JsCast;
#[cfg(feature = "viz")]
use wasm_bindgen_futures::JsFuture;
#[cfg(feature = "viz")]
use js_sys::Promise;
#[cfg(feature = "viz")]
use wasm_bindgen::closure::Closure;

use crate::generation::{
    ContextStrategy, prefill_state, resolve_context_strategy, sample_next_token_async,
};
use crate::tokenizer::SharedTokenizer;
use crate::tokenizer::char_vocab::CharVocab;
use crate::{BDH, ContextStrategyConfig, ModelOverrides, ModelState, build_model_config};

#[cfg(feature = "viz")]
use crate::viz::{self, VizConfig, VizDimensions, VizEncoder};
type WebBackend = burn_wgpu::WebGpu<f32>;
type WebDevice = <WebBackend as Backend>::Device;

#[derive(Debug, Deserialize, Default)]
struct WebModelConfig {
    #[serde(default)]
    block_size: Option<usize>,
    #[serde(default)]
    max_tokens: Option<i32>,
    #[serde(default)]
    overrides: ModelOverrides,
}

struct StreamState {
    state: ModelState<WebBackend>,
    last_logits: Option<Tensor<WebBackend, 1>>,
    generated: usize,
    max_tokens: Option<usize>,
    temperature: f32,
    top_k: Option<usize>,
    strategy: ContextStrategy,
}

#[cfg(feature = "viz")]
struct WebVizRuntime {
    encoder: VizEncoder<WebBackend>,
    sender: viz::VizSender<WebBackend>,
}

#[cfg(feature = "viz")]
async fn yield_now() {
    let promise = Promise::new(&mut |resolve, _reject| {
        let Some(window) = web_sys::window() else {
            let _ = resolve.call0(&JsValue::NULL);
            return;
        };
        let resolve = resolve.clone();
        let closure = Closure::once_into_js(move |_ts: f64| {
            let _ = resolve.call0(&JsValue::NULL);
        });
        let _ = window.request_animation_frame(closure.unchecked_ref());
    });
    let _ = JsFuture::from(promise).await;
}

#[wasm_bindgen]
pub struct WasmInference {
    model: BDH<WebBackend>,
    tokenizer: SharedTokenizer,
    device: WebDevice,
    block_size: usize,
    default_max_tokens: Option<usize>,
    stream: Option<StreamState>,
    #[cfg(feature = "viz")]
    viz: Option<WebVizRuntime>,
}

#[wasm_bindgen(js_name = "loadModel")]
pub async fn load_model(
    model_bytes: Vec<u8>,
    vocab_json: String,
    config_json: Option<String>,
    start_viz: Option<bool>,
) -> Result<WasmInference, JsValue> {
    async fn load_inner(
        model_bytes: Vec<u8>,
        vocab_json: String,
        config_json: Option<String>,
        start_viz: Option<bool>,
    ) -> Result<WasmInference> {
        console_error_panic_hook::set_once();

        let config = match config_json {
            Some(json) if !json.trim().is_empty() => {
                serde_json::from_str::<WebModelConfig>(&json)
                    .context("failed to parse web model config json")?
            }
            _ => WebModelConfig::default(),
        };

        let overrides = config.overrides;
        let block_size = config
            .block_size
            .or(overrides.block_size)
            .unwrap_or(256)
            .max(1);
        let default_max_tokens = normalize_max_tokens(config.max_tokens);
        let tokenizer = Arc::new(CharVocab::from_json_str(&vocab_json)?) as SharedTokenizer;

        let mut model_config = build_model_config(&overrides, block_size);
        model_config.vocab_size = tokenizer.len();
        #[cfg(feature = "viz")]
        let dims = VizDimensions {
            layers: model_config.n_layer,
            heads: model_config.n_head,
            latent_per_head: model_config.latent_per_head(),
        };

        #[cfg(feature = "viz")]
        let (device, viz_runtime) = if start_viz.unwrap_or(false) {
            let viz_config = VizConfig::default();
            let overlay = viz::start_overlay_wasm::<WebBackend>(viz_config.clone(), dims);
            let (handle, app) = overlay.split();
            let sender = handle.sender();
            viz::bevy_app::run_app_wasm(app);

            let device = loop {
                if let Some(device) = handle.device_ready() {
                    break device;
                }
                yield_now().await;
            };

            WebBackend::seed(&device, 1337);
            (
                device.clone(),
                Some(WebVizRuntime {
                    encoder: VizEncoder::new(
                        viz_config,
                        dims.layers,
                        dims.heads,
                        dims.latent_per_head,
                        &device,
                    ),
                    sender,
                }),
            )
        } else {
            let device = WebDevice::default();
            burn_wgpu::init_setup_async::<graphics::WebGpu>(&device, RuntimeOptions::default())
                .await;
            WebBackend::seed(&device, 1337);
            (device, None)
        };

        #[cfg(not(feature = "viz"))]
        let device = {
            let device = WebDevice::default();
            burn_wgpu::init_setup_async::<graphics::WebGpu>(&device, RuntimeOptions::default())
                .await;
            WebBackend::seed(&device, 1337);
            device
        };

        let bytes = model_bytes.as_slice();
        let record: <BDH<WebBackend> as Module<WebBackend>>::Record =
            match BinBytesRecorder::<FullPrecisionSettings, &[u8]>::default()
                .load(bytes, &device)
            {
                Ok(record) => record,
                Err(full_err) => BinBytesRecorder::<HalfPrecisionSettings, &[u8]>::default()
                    .load(bytes, &device)
                    .map_err(|half_err| {
                        anyhow!(
                            "failed to load model weights as f32 ({full_err}) or f16 ({half_err})"
                        )
                    })?,
            };
        let model = BDH::<WebBackend>::new(model_config, &device).load_record(record);

        #[cfg(not(feature = "viz"))]
        let _ = start_viz;

        Ok(WasmInference {
            model,
            tokenizer,
            device,
            block_size,
            default_max_tokens,
            stream: None,
            #[cfg(feature = "viz")]
            viz: viz_runtime,
        })
    }

    load_inner(model_bytes, vocab_json, config_json, start_viz)
        .await
        .map_err(|err| JsValue::from_str(&err.to_string()))
}

#[wasm_bindgen]
impl WasmInference {
    #[wasm_bindgen(js_name = "startStream")]
    pub fn start_stream(
        &mut self,
        prompt: String,
        max_tokens: Option<i32>,
        temperature: f32,
        top_k: Option<u32>,
        context_window: Option<u32>,
    ) -> Result<String, JsValue> {
        let max_tokens = match max_tokens {
            Some(value) => normalize_max_tokens(Some(value)),
            None => self.default_max_tokens,
        };
        let strategy_cfg = match context_window {
            Some(window) => ContextStrategyConfig::Sliding {
                window: window as usize,
            },
            None => ContextStrategyConfig::Infinite,
        };
        let strategy = resolve_context_strategy(&strategy_cfg, self.block_size);

        let mut prompt_ids = self.tokenizer.encode(&prompt, false, false);
        if let ContextStrategy::Sliding { window } = strategy
            && prompt_ids.len() > window
        {
            prompt_ids = prompt_ids[prompt_ids.len() - window..].to_vec();
        }

        let prompt_tokens: Vec<i64> = prompt_ids.iter().map(|&id| id as i64).collect();
        let prompt_text = self.tokenizer.decode(&prompt_ids);

        let (mut state, last_logits) =
            prefill_state::<WebBackend>(&self.model, &prompt_tokens, &self.device)
                .map_err(|err| JsValue::from_str(&err.to_string()))?;

        if let ContextStrategy::Sliding { window } = strategy
            && window > 0
            && state.position > window
        {
            state.trim(window);
        }

        self.stream = Some(StreamState {
            state,
            last_logits: Some(last_logits),
            generated: 0,
            max_tokens,
            temperature,
            top_k: top_k.map(|value| value as usize),
            strategy,
        });

        Ok(prompt_text)
    }

    #[wasm_bindgen(js_name = "nextChunk")]
    pub async fn next_chunk(&mut self) -> Result<Option<String>, JsValue> {
        let mut stream = match self.stream.take() {
            Some(stream) => stream,
            None => return Err(JsValue::from_str("stream not initialized")),
        };

        while stream
            .max_tokens
            .map_or(true, |max_tokens| stream.generated < max_tokens)
        {
            let last_logits = stream
                .last_logits
                .take()
                .ok_or_else(|| JsValue::from_str("stream missing logits"))?;
            let (next, logits) = sample_next_token_async(
                &self.model,
                &mut stream.state,
                last_logits,
                stream.temperature,
                stream.top_k,
                &self.device,
            )
            .await
            .map_err(|err| JsValue::from_str(&err.to_string()))?;
            stream.last_logits = Some(logits);
            stream.generated = stream.generated.saturating_add(1);

            #[cfg(feature = "viz")]
            if let Some(viz) = self.viz.as_mut() {
                let token_index = stream.state.position.saturating_sub(1);
                if viz.encoder.should_capture(token_index) {
                    let layers = stream.state.take_viz();
                    let frame = viz.encoder.step(&layers, token_index);
                    viz.sender.try_send(frame);
                }
            }

            if let Ok(token_u32) = u32::try_from(next) {
                if self.tokenizer.eos_id() == Some(token_u32) {
                    self.stream = None;
                    return Ok(None);
                }

                let new_text = self.tokenizer.decode(&[token_u32]);
                if let ContextStrategy::Sliding { window } = stream.strategy
                    && window > 0
                    && stream.state.position > window
                {
                    stream.state.trim(window);
                }
                if !new_text.is_empty() {
                    self.stream = Some(stream);
                    return Ok(Some(new_text));
                }
            }

            if let ContextStrategy::Sliding { window } = stream.strategy
                && window > 0
                && stream.state.position > window
            {
                stream.state.trim(window);
            }
        }

        self.stream = None;
        Ok(None)
    }

    #[wasm_bindgen(js_name = "clearStream")]
    pub fn clear_stream(&mut self) {
        self.stream = None;
    }
}

fn normalize_max_tokens(max_tokens: Option<i32>) -> Option<usize> {
    match max_tokens {
        Some(value) if value >= 0 => Some(value as usize),
        _ => None,
    }
}
