use std::cmp::{Ordering, max, min};
use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result, anyhow};
use crossbeam_channel::{Receiver, Sender, unbounded};
use ndarray::Array3;
use parking_lot::Mutex;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::Color as PlottersColor;
use plotters_bitmap::BitMapBackendError;
use tempfile::NamedTempFile;

use super::{FrameSnapshot, VideoConfig, VideoLayout, VizTarget};
use crate::viz::activity::FrameActivity;
use crate::viz::frame::TokenFrame;

#[derive(Clone, Copy)]
struct Rect {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

impl Rect {
    fn width(&self) -> usize {
        self.x1.saturating_sub(self.x0)
    }

    fn height(&self) -> usize {
        self.y1.saturating_sub(self.y0)
    }
}

pub struct VideoHandle {
    tx: Mutex<Option<Sender<FrameSnapshot>>>,
    join: Mutex<Option<JoinHandle<()>>>,
    errors: Arc<Mutex<Option<anyhow::Error>>>,
}

const EEG_HISTORY_LEN: usize = 240;
const TRANSCRIPT_HISTORY_CHARS: usize = 4096;

#[derive(Clone, Copy, Default)]
struct EegSample {
    activation: f32,
    synapse: f32,
    energy: f32,
    plasticity: f32,
}

struct EegHistory {
    per_layer: Vec<VecDeque<EegSample>>,
    capacity: usize,
}

#[derive(Clone, Copy)]
enum EegMode {
    Activation,
    Plasticity,
}

struct TranscriptHistory {
    chars: VecDeque<char>,
    capacity: usize,
}

const MAX_EEG_BANDS: usize = 24;
const EEG_TITLE_GAP: usize = 20;

#[derive(Clone, Copy)]
enum PanelSize {
    Fixed(u32),
    Flex(f32),
}

#[derive(Clone, Copy)]
enum PanelKind {
    TopBar,
    Eeg(EegMode),
    Transcript,
    ScaleFree,
}

#[derive(Clone, Copy)]
struct LayoutModule {
    size: PanelSize,
    kind: PanelKind,
}

impl LayoutModule {
    fn fixed(kind: PanelKind, px: u32) -> Self {
        Self {
            size: PanelSize::Fixed(px),
            kind,
        }
    }

    fn flex(kind: PanelKind, weight: f32) -> Self {
        Self {
            size: PanelSize::Flex(weight),
            kind,
        }
    }
}

struct VizRenderContext<'a> {
    frame: &'a TokenFrame,
    activity: &'a FrameActivity,
    history: &'a EegHistory,
    transcript: &'a TranscriptHistory,
}

impl<'a> VizRenderContext<'a> {
    fn new(
        snapshot: &'a FrameSnapshot,
        history: &'a EegHistory,
        transcript: &'a TranscriptHistory,
    ) -> Self {
        Self {
            frame: snapshot.frame(),
            activity: snapshot.activity(),
            history,
            transcript,
        }
    }

    fn layer_count(&self) -> usize {
        self.frame.layers.len()
    }
}

#[derive(Clone)]
struct EegBand {
    start_layer: usize,
    end_layer: usize,
    samples: Vec<EegSample>,
}

struct LayerDegreeSeries {
    layer: u8,
    points: Vec<(f32, f32)>,
}

#[derive(Clone, Copy)]
struct Color {
    r: u8,
    g: u8,
    b: u8,
}

impl Color {
    const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    fn to_array(self) -> [u8; 3] {
        [self.r, self.g, self.b]
    }

    fn to_rgb(self) -> RGBColor {
        RGBColor(self.r, self.g, self.b)
    }

    fn lerp(self, other: Color, t: f32) -> Self {
        let clamped = t.clamp(0.0, 1.0);
        let r = self.r as f32 + (other.r as f32 - self.r as f32) * clamped;
        let g = self.g as f32 + (other.g as f32 - self.g as f32) * clamped;
        let b = self.b as f32 + (other.b as f32 - self.b as f32) * clamped;
        Self {
            r: r.round() as u8,
            g: g.round() as u8,
            b: b.round() as u8,
        }
    }
}

#[derive(Clone, Copy)]
struct ColorStop {
    at: f32,
    color: Color,
}

struct EegColorTheme {
    title: &'static str,
    title_color: Color,
    panel_bg: Color,
    gradient: &'static [ColorStop],
    trace: Color,
    overlay: Color,
    grid: Color,
}

const ACTIVATION_STOPS: [ColorStop; 4] = [
    ColorStop {
        at: 0.0,
        color: Color::new(4, 8, 20),
    },
    ColorStop {
        at: 0.35,
        color: Color::new(24, 78, 106),
    },
    ColorStop {
        at: 0.7,
        color: Color::new(90, 184, 140),
    },
    ColorStop {
        at: 1.0,
        color: Color::new(236, 244, 214),
    },
];

const PLASTICITY_STOPS: [ColorStop; 4] = [
    ColorStop {
        at: 0.0,
        color: Color::new(20, 4, 32),
    },
    ColorStop {
        at: 0.4,
        color: Color::new(88, 36, 140),
    },
    ColorStop {
        at: 0.75,
        color: Color::new(190, 96, 64),
    },
    ColorStop {
        at: 1.0,
        color: Color::new(250, 220, 160),
    },
];

const ACTIVATION_THEME: EegColorTheme = EegColorTheme {
    title: "EEG • Activation / Sparsity",
    title_color: Color::new(210, 232, 224),
    panel_bg: Color::new(12, 14, 24),
    gradient: &ACTIVATION_STOPS,
    trace: Color::new(112, 226, 196),
    overlay: Color::new(248, 196, 92),
    grid: Color::new(52, 58, 74),
};

const PLASTICITY_THEME: EegColorTheme = EegColorTheme {
    title: "EEG • Plasticity / Synapse density",
    title_color: Color::new(230, 210, 240),
    panel_bg: Color::new(14, 10, 28),
    gradient: &PLASTICITY_STOPS,
    trace: Color::new(242, 156, 212),
    overlay: Color::new(160, 212, 255),
    grid: Color::new(66, 50, 92),
};

impl EegColorTheme {
    fn heat_color(&self, value: f32) -> Color {
        sample_gradient(self.gradient, value)
    }
}

impl EegMode {
    fn theme(self) -> &'static EegColorTheme {
        match self {
            EegMode::Activation => &ACTIVATION_THEME,
            EegMode::Plasticity => &PLASTICITY_THEME,
        }
    }

    fn metric_value(self, sample: &EegSample) -> f32 {
        match self {
            EegMode::Activation => sample.activation,
            EegMode::Plasticity => {
                let plastic = sample.plasticity * 8.0;
                let syn = sample.synapse * 0.4;
                let energy = sample.energy * 0.2;
                (plastic + syn + energy).clamp(0.0, 1.0)
            }
        }
    }
}

fn sample_gradient(stops: &[ColorStop], value: f32) -> Color {
    if stops.is_empty() {
        return Color::new(255, 255, 255);
    }
    if stops.len() == 1 {
        return stops[0].color;
    }
    let clamped = value.clamp(0.0, 1.0);
    if clamped <= stops[0].at {
        return stops[0].color;
    }
    for window in stops.windows(2) {
        if let [start, end] = window {
            if clamped <= end.at {
                let span = (end.at - start.at).max(f32::EPSILON);
                let t = ((clamped - start.at) / span).clamp(0.0, 1.0);
                return start.color.lerp(end.color, t);
            }
        }
    }
    stops.last().copied().unwrap().color
}

impl EegHistory {
    fn new(initial_layers: usize, capacity: usize) -> Self {
        Self {
            per_layer: vec![VecDeque::with_capacity(capacity); initial_layers],
            capacity,
        }
    }

    fn ensure_layers(&mut self, count: usize) {
        if count > self.per_layer.len() {
            self.per_layer.extend(
                (self.per_layer.len()..count).map(|_| VecDeque::with_capacity(self.capacity)),
            );
        }
    }

    fn record_all(&mut self, activity: &FrameActivity) {
        self.ensure_layers(activity.layers.len());
        for (idx, layer) in activity.layers.iter().enumerate() {
            self.record(
                idx,
                EegSample {
                    activation: layer.activation_density,
                    synapse: layer.synapse_density,
                    energy: layer.activation_energy,
                    plasticity: layer.mean_abs_delta,
                },
            );
        }
    }

    fn record(&mut self, layer_index: usize, value: EegSample) {
        if layer_index >= self.per_layer.len() {
            self.ensure_layers(layer_index + 1);
        }
        let deque = &mut self.per_layer[layer_index];
        deque.push_back(value);
        if deque.len() > self.capacity {
            deque.pop_front();
        }
    }

    fn layers(&self) -> usize {
        self.per_layer.len()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn series(&self, layer_index: usize) -> Vec<EegSample> {
        if layer_index >= self.per_layer.len() {
            return vec![EegSample::default(); self.capacity];
        }
        let deque = &self.per_layer[layer_index];
        let missing = self.capacity.saturating_sub(deque.len());
        let mut data = vec![EegSample::default(); missing];
        data.extend(deque.iter().copied());
        data
    }
}

impl TranscriptHistory {
    fn new(capacity: usize) -> Self {
        Self {
            chars: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn record(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        for ch in text.chars() {
            let normalized = match ch {
                '\r' => ' ',
                '\n' => '\n',
                c if c.is_control() => ' ',
                _ => ch,
            };
            self.chars.push_back(normalized);
        }
        let trailing_whitespace = text
            .chars()
            .last()
            .map(|c| c.is_whitespace())
            .unwrap_or(false);
        if !trailing_whitespace {
            self.chars.push_back(' ');
        }
        self.trim();
    }

    fn trim(&mut self) {
        while self.chars.len() > self.capacity {
            self.chars.pop_front();
        }
    }

    fn lines(&self, chars_per_line: usize, max_lines: usize) -> Vec<String> {
        if chars_per_line == 0 || max_lines == 0 {
            return Vec::new();
        }
        let mut lines = Vec::new();
        let mut current = String::new();
        let mut count = 0usize;
        for &ch in &self.chars {
            if ch == '\n' {
                if !current.is_empty() {
                    lines.push(current.clone());
                    current.clear();
                }
                count = 0;
                continue;
            }
            current.push(ch);
            count += 1;
            if count >= chars_per_line {
                lines.push(current.clone());
                current.clear();
                count = 0;
            }
        }
        if !current.is_empty() {
            lines.push(current);
        }
        if lines.is_empty() {
            return Vec::new();
        }
        let keep = lines.len().min(max_lines);
        lines
            .into_iter()
            .rev()
            .take(keep)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }
}

fn image_dims(image: &Array3<u8>) -> (usize, usize) {
    let shape = image.shape();
    (shape[1], shape[0])
}

fn put_pixel(image: &mut Array3<u8>, x: usize, y: usize, color: [u8; 3]) {
    let (width, height) = image_dims(image);
    if x < width && y < height {
        image[[y, x, 0]] = color[0];
        image[[y, x, 1]] = color[1];
        image[[y, x, 2]] = color[2];
    }
}

fn fill_rect(image: &mut Array3<u8>, rect: Rect, color: [u8; 3]) {
    let (width, height) = image_dims(image);
    let x0 = rect.x0.min(width);
    let x1 = rect.x1.min(width);
    let y0 = rect.y0.min(height);
    let y1 = rect.y1.min(height);
    for y in y0..y1 {
        for x in x0..x1 {
            image[[y, x, 0]] = color[0];
            image[[y, x, 1]] = color[1];
            image[[y, x, 2]] = color[2];
        }
    }
}

fn draw_border(image: &mut Array3<u8>, rect: Rect, color: [u8; 3]) {
    if rect.width() == 0 || rect.height() == 0 {
        return;
    }
    let (width, height) = image_dims(image);
    let x0 = rect.x0.min(width.saturating_sub(1));
    let x1 = rect.x1.min(width).saturating_sub(1);
    let y0 = rect.y0.min(height.saturating_sub(1));
    let y1 = rect.y1.min(height).saturating_sub(1);

    for x in x0..=x1 {
        put_pixel(image, x, y0, color);
        put_pixel(image, x, y1, color);
    }
    for y in y0..=y1 {
        put_pixel(image, x0, y, color);
        put_pixel(image, x1, y, color);
    }
}

type PlotResult = Result<(), DrawingAreaErrorKind<BitMapBackendError>>;

fn paint_plot<F>(image: &mut Array3<u8>, rect: Rect, mut draw: F)
where
    F: FnMut(DrawingArea<BitMapBackend<'_>, Shift>) -> PlotResult,
{
    let (img_width, img_height) = image_dims(image);
    let bounded = Rect {
        x0: rect.x0.min(img_width),
        y0: rect.y0.min(img_height),
        x1: rect.x1.min(img_width),
        y1: rect.y1.min(img_height),
    };
    let width = bounded.width();
    let height = bounded.height();
    if width == 0 || height == 0 {
        return;
    }
    let mut buffer = vec![0u8; width * height * 3];
    {
        let backend = BitMapBackend::with_buffer(&mut buffer, (width as u32, height as u32));
        let area = backend.into_drawing_area();
        if let Err(err) = area.fill(&RGBColor(16, 18, 28)) {
            eprintln!("viz: plot fill failed: {err:?}");
        }
        if let Err(err) = draw(area) {
            eprintln!("viz: plot render failed: {err:?}");
        }
    }
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let dest_x = bounded.x0 + x;
            let dest_y = bounded.y0 + y;
            if dest_x < img_width && dest_y < img_height {
                image[[dest_y, dest_x, 0]] = buffer[idx];
                image[[dest_y, dest_x, 1]] = buffer[idx + 1];
                image[[dest_y, dest_x, 2]] = buffer[idx + 2];
            }
        }
    }
}

fn sanitized_token_text(text: &str, max_chars: usize) -> String {
    let mut rendered = String::new();
    let mut count = 0usize;
    for ch in text.chars() {
        let printable = if ch.is_ascii_graphic() || ch == ' ' {
            ch
        } else {
            '?'
        };
        rendered.push(printable);
        count += 1;
        if count >= max_chars {
            break;
        }
    }
    if text.chars().count() > count {
        rendered.push_str("...");
    }
    rendered
}

const fn glyph_bits(ch: char) -> [u8; 5] {
    match ch {
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        'A' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'B' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'C' => [0b111, 0b100, 0b100, 0b100, 0b111],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b110, 0b100, 0b111],
        'F' => [0b111, 0b100, 0b110, 0b100, 0b100],
        'G' => [0b111, 0b100, 0b111, 0b101, 0b111],
        'H' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'J' => [0b111, 0b001, 0b001, 0b101, 0b111],
        'K' => [0b101, 0b101, 0b110, 0b101, 0b101],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'M' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'N' => [0b101, 0b111, 0b111, 0b111, 0b101],
        'O' => [0b111, 0b101, 0b101, 0b101, 0b111],
        'P' => [0b111, 0b101, 0b111, 0b100, 0b100],
        'Q' => [0b111, 0b101, 0b101, 0b111, 0b011],
        'R' => [0b111, 0b101, 0b111, 0b110, 0b101],
        'S' => [0b111, 0b100, 0b111, 0b001, 0b111],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'U' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'V' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'W' => [0b101, 0b101, 0b111, 0b111, 0b101],
        'X' => [0b101, 0b101, 0b010, 0b101, 0b101],
        'Y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        'Z' => [0b111, 0b001, 0b010, 0b100, 0b111],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        ':' => [0b000, 0b010, 0b000, 0b010, 0b000],
        '.' => [0b000, 0b000, 0b000, 0b010, 0b000],
        ',' => [0b000, 0b000, 0b000, 0b010, 0b100],
        '!' => [0b010, 0b010, 0b010, 0b000, 0b010],
        '?' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '_' => [0b000, 0b000, 0b000, 0b000, 0b111],
        '/' => [0b001, 0b001, 0b010, 0b100, 0b100],
        ' ' => [0b000, 0b000, 0b000, 0b000, 0b000],
        _ => [0b111, 0b101, 0b010, 0b010, 0b111],
    }
}

fn draw_char(image: &mut Array3<u8>, x: usize, y: usize, ch: char, color: [u8; 3]) -> usize {
    let glyph_char = if ch.is_ascii_lowercase() {
        ch.to_ascii_uppercase()
    } else {
        ch
    };
    let glyph = glyph_bits(glyph_char);
    for (row, pattern) in glyph.iter().enumerate() {
        for col in 0..3 {
            if (pattern >> (2 - col)) & 1 == 1 {
                put_pixel(image, x + col, y + row, color);
            }
        }
    }
    4
}

fn draw_text_line(image: &mut Array3<u8>, x: usize, y: usize, text: &str, color: [u8; 3]) {
    let (width, height) = image_dims(image);
    if y + 5 >= height || x >= width {
        return;
    }
    let mut cursor = x;
    for ch in text.chars() {
        if cursor + 3 >= width {
            break;
        }
        cursor += draw_char(image, cursor, y, ch, color);
    }
}

impl VideoHandle {
    pub fn start(config: VideoConfig) -> Result<Arc<Self>> {
        let (tx_raw, rx) = unbounded();
        let errors = Arc::new(Mutex::new(None));
        let thread_errors = Arc::clone(&errors);

        eprintln!(
            "viz: starting video encoder thread -> {} ({}x{} @ {}fps)",
            config.path.display(),
            config.width,
            config.height,
            config.fps
        );

        let handle = Arc::new(Self {
            tx: Mutex::new(Some(tx_raw)),
            join: Mutex::new(None),
            errors,
        });

        let thread_config = config.clone();
        let thread_handle = thread::Builder::new()
            .name("bdh-viz-encoder".into())
            .spawn(move || {
                if let Err(err) = run_encoder(rx, thread_config) {
                    *thread_errors.lock() = Some(err);
                }
            })
            .context("failed to spawn viz encoder thread")?;

        *handle.join.lock() = Some(thread_handle);
        Ok(handle)
    }
}

impl VizTarget for VideoHandle {
    fn push(&self, snapshot: FrameSnapshot) {
        let mut guard = self.tx.lock();
        if let Some(sender) = guard.as_ref() {
            if sender.send(snapshot).is_err() {
                eprintln!("viz: encoder channel closed; disabling visualization output");
                *guard = None;
            }
        }
    }

    fn finalize(&self) {
        self.tx.lock().take();
        if let Some(join) = self.join.lock().take() {
            if let Err(err) = join.join() {
                eprintln!("viz: encoder thread join failed: {err:?}");
            }
        }
        if let Some(err) = self.errors.lock().take() {
            eprintln!("viz encoder error: {err:#}");
        }
    }
}

impl Drop for VideoHandle {
    fn drop(&mut self) {
        self.finalize();
    }
}

fn sanitize_dimension(dim: u32) -> usize {
    // YUV420 encoders require even dimensions; fall back to the nearest even value >= 2.
    let sanitized = dim.max(2);
    (sanitized - sanitized % 2) as usize
}

struct EncoderPlan {
    description: &'static str,
    args: Vec<String>,
    extension: Option<&'static str>,
    requires_faststart: bool,
}

fn encoder_plans(config: &VideoConfig) -> Vec<EncoderPlan> {
    let mut plans = Vec::new();
    let crf = config.crf.unwrap_or(18).clamp(0, 51).to_string();
    let preset = if config.realtime {
        "medium".to_string()
    } else {
        "slow".to_string()
    };

    let mut x264_args = vec![
        "-c:v".into(),
        "libx264".into(),
        "-preset".into(),
        preset,
        "-crf".into(),
        crf.clone(),
        "-pix_fmt".into(),
        "yuv420p".into(),
    ];
    if let Some(bitrate) = config.bitrate_kbps {
        let target = format!("{}k", bitrate);
        x264_args.push("-b:v".into());
        x264_args.push(target.clone());
        x264_args.push("-maxrate".into());
        x264_args.push(target.clone());
        x264_args.push("-bufsize".into());
        x264_args.push(format!("{}k", bitrate.saturating_mul(2)));
    }
    if let Some(interval) = config.keyframe_interval {
        x264_args.push("-g".into());
        x264_args.push(interval.to_string());
    }
    plans.push(EncoderPlan {
        description: "H.264 (libx264)",
        args: x264_args,
        extension: None,
        requires_faststart: true,
    });

    let mut prores_args = vec![
        "-c:v".into(),
        "prores_ks".into(),
        "-profile:v".into(),
        "3".into(),
        "-pix_fmt".into(),
        "yuv422p10le".into(),
    ];
    if let Some(interval) = config.keyframe_interval {
        prores_args.push("-g".into());
        prores_args.push(interval.to_string());
    }
    plans.push(EncoderPlan {
        description: "Apple ProRes 422 HQ (prores_ks)",
        args: prores_args,
        extension: Some("mov"),
        requires_faststart: false,
    });

    let mut ffv1_args = vec![
        "-c:v".into(),
        "ffv1".into(),
        "-level".into(),
        "3".into(),
        "-coder".into(),
        "1".into(),
        "-context".into(),
        "1".into(),
        "-pix_fmt".into(),
        "rgb24".into(),
    ];
    if let Some(interval) = config.keyframe_interval {
        ffv1_args.push("-g".into());
        ffv1_args.push(interval.to_string());
    }
    plans.push(EncoderPlan {
        description: "FFV1 lossless",
        args: ffv1_args,
        extension: Some("mkv"),
        requires_faststart: false,
    });

    plans
}

fn encode_with_ffmpeg(
    raw_path: &Path,
    frame_count: usize,
    width: usize,
    height: usize,
    fps: usize,
    config: &VideoConfig,
) -> Result<PathBuf> {
    if frame_count == 0 {
        return Err(anyhow!("no frames available for encoding"));
    }

    let ffmpeg_bin = env::var("BDH_FFMPEG").unwrap_or_else(|_| "ffmpeg".into());
    let dims = format!("{}x{}", width, height);
    let fps_str = fps.to_string();
    let frame_str = frame_count.to_string();
    let input_path = raw_path
        .to_str()
        .ok_or_else(|| anyhow!("temporary raw frame path is not valid UTF-8"))?
        .to_owned();

    let mut failures: Vec<(String, String)> = Vec::new();

    for plan in encoder_plans(config) {
        let mut output_path = config.path.clone();
        if let Some(ext) = plan.extension {
            output_path.set_extension(ext);
        }

        let mut cmd = Command::new(&ffmpeg_bin);
        cmd.arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-y")
            .arg("-f")
            .arg("rawvideo")
            .arg("-pix_fmt")
            .arg("rgb24")
            .arg("-s")
            .arg(&dims)
            .arg("-r")
            .arg(&fps_str)
            .arg("-i")
            .arg(&input_path)
            .arg("-frames:v")
            .arg(&frame_str);

        for arg in &plan.args {
            cmd.arg(arg);
        }

        if plan.requires_faststart {
            cmd.arg("-movflags").arg("+faststart");
        }

        cmd.arg(output_path.as_os_str());

        eprintln!(
            "viz: encoding {} frames with {}...",
            frame_count, plan.description
        );

        match cmd.output() {
            Ok(output) if output.status.success() => {
                eprintln!(
                    "viz: wrote {} using {}.",
                    output_path.display(),
                    plan.description
                );
                if output_path != config.path {
                    eprintln!(
                        "viz: requested output {} but using {}; adjust `viz_output` if you require a specific container.",
                        config.path.display(),
                        output_path.display()
                    );
                }
                return Ok(output_path);
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
                failures.push((plan.description.to_string(), stderr));
            }
            Err(err) => {
                failures.push((plan.description.to_string(), err.to_string()));
            }
        }
    }

    if failures
        .iter()
        .any(|(desc, _)| desc.contains("H.264 (libx264)"))
    {
        eprintln!(
            "viz: none of the preferred ffmpeg encoders are available; install an ffmpeg build with libx264 for best quality."
        );
    }

    let mut report = String::new();
    for (desc, err) in failures {
        report.push_str(&format!("- {desc}: {err}\n"));
    }

    Err(anyhow!(
        "failed to encode visualization frames with ffmpeg:\n{}",
        report.trim_end()
    ))
}

fn run_encoder(rx: Receiver<FrameSnapshot>, config: VideoConfig) -> Result<()> {
    if let Some(parent) = config.path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create viz output directory {parent:?}"))?;
        }
    }

    let requested_width = config.width;
    let requested_height = config.height;
    let width = sanitize_dimension(requested_width);
    let height = sanitize_dimension(requested_height);
    let fps = config.fps.max(1) as usize;

    if width as u32 != requested_width || height as u32 != requested_height {
        eprintln!(
            "viz: adjusting frame size from {}x{} to {}x{} for YUV420 encoder compatibility",
            requested_width, requested_height, width, height
        );
    }

    let mut raw_file =
        NamedTempFile::new().context("failed to create temporary file for viz frames")?;
    let mut writer = BufWriter::new(raw_file.as_file_mut());
    let mut count = 0usize;
    let layout = config.layout;

    let mut eeg_history = EegHistory::new(0, EEG_HISTORY_LEN);
    let mut transcript_history = TranscriptHistory::new(TRANSCRIPT_HISTORY_CHARS);

    for snapshot in rx.iter() {
        eeg_history.record_all(snapshot.activity());
        transcript_history.record(snapshot.frame().token_text.as_str());
        let image = render_frame(
            &snapshot,
            width,
            height,
            layout,
            &eeg_history,
            &transcript_history,
        );
        let slice = image
            .as_slice()
            .ok_or_else(|| anyhow!("video frame buffer is not contiguous"))?;
        writer.write_all(slice)?;
        count += 1;
    }

    writer.flush()?;
    drop(writer);

    if count == 0 {
        eprintln!("viz: no frames collected; skipping video encoding.");
        return Ok(());
    }

    let raw_path = raw_file.into_temp_path();
    let final_path = encode_with_ffmpeg(raw_path.as_ref(), count, width, height, fps, &config)
        .with_context(|| "ffmpeg encoding pipeline failed")?;

    eprintln!(
        "viz: encoded {count} frames to {} ({}x{} @ {}fps).",
        final_path.display(),
        width,
        height,
        fps
    );

    raw_path.close().ok();

    Ok(())
}

fn render_frame(
    snapshot: &FrameSnapshot,
    width: usize,
    height: usize,
    layout: VideoLayout,
    eeg_history: &EegHistory,
    transcript_history: &TranscriptHistory,
) -> Array3<u8> {
    let modules = modules_for_layout(layout);
    render_layout(
        snapshot,
        width,
        height,
        eeg_history,
        transcript_history,
        &modules,
    )
}

fn modules_for_layout(layout: VideoLayout) -> Vec<LayoutModule> {
    match layout {
        VideoLayout::Timeline => vec![
            LayoutModule::fixed(PanelKind::TopBar, 112),
            LayoutModule::fixed(PanelKind::Transcript, 120),
            LayoutModule::flex(PanelKind::Eeg(EegMode::Activation), 1.0),
        ],
        VideoLayout::Plasticity => vec![
            LayoutModule::fixed(PanelKind::TopBar, 112),
            LayoutModule::fixed(PanelKind::Transcript, 120),
            LayoutModule::flex(PanelKind::Eeg(EegMode::Plasticity), 1.0),
        ],
        VideoLayout::ScaleFree => vec![
            LayoutModule::fixed(PanelKind::TopBar, 112),
            LayoutModule::fixed(PanelKind::Transcript, 120),
            LayoutModule::flex(PanelKind::ScaleFree, 1.0),
        ],
    }
}

fn render_layout(
    snapshot: &FrameSnapshot,
    width: usize,
    height: usize,
    eeg_history: &EegHistory,
    transcript_history: &TranscriptHistory,
    modules: &[LayoutModule],
) -> Array3<u8> {
    let safe_width = width.max(1);
    let safe_height = height.max(1);
    let mut image = Array3::<u8>::from_elem((safe_height, safe_width, 3), 10);

    if modules.is_empty() || width == 0 || height == 0 {
        return image;
    }

    let context = VizRenderContext::new(snapshot, eeg_history, transcript_history);
    let heights = compute_panel_heights(modules, safe_height);
    let mut cursor_y = 0usize;
    for (module, module_height) in modules.iter().zip(heights.iter()) {
        if *module_height == 0 {
            continue;
        }

        let rect = Rect {
            x0: 0,
            y0: cursor_y,
            x1: safe_width,
            y1: (cursor_y + module_height).min(safe_height),
        };

        match module.kind {
            PanelKind::TopBar => render_top_bar(&context, rect, &mut image),
            PanelKind::Eeg(mode) => render_eeg_panel(&context, rect, mode, &mut image),
            PanelKind::Transcript => render_transcript_panel(&context, rect, &mut image),
            PanelKind::ScaleFree => render_scalefree_panel(&context, rect, &mut image),
        }

        cursor_y = rect.y1;
    }

    image
}

fn render_top_bar(context: &VizRenderContext, rect: Rect, image: &mut Array3<u8>) {
    let width = rect.width();
    let height = rect.height();
    if width == 0 || height == 0 {
        return;
    }

    let top_rect = Rect {
        x0: rect.x0,
        y0: rect.y0,
        x1: rect.x1,
        y1: rect.y1,
    };
    fill_rect(image, top_rect, [18, 24, 32]);
    draw_border(image, top_rect, [60, 68, 82]);

    let frame = context.frame;
    let info = format!("STEP {:>6}   TOKEN {:>7}", frame.t, frame.token_id);
    draw_text_line(
        image,
        rect.x0 + 8,
        rect.y0 + min(8, height.saturating_sub(6)),
        &info,
        [232, 232, 224],
    );

    if height >= 24 {
        let text = sanitized_token_text(&frame.token_text, (width / 6).max(12));
        draw_text_line(
            image,
            rect.x0 + 8,
            rect.y0 + min(24, height.saturating_sub(6)),
            &format!("TEXT {}", text),
            [210, 210, 180],
        );
    }

    if !context.activity.layers.is_empty() {
        let totals = &context.activity.totals;
        let total_nodes = totals.nodes;
        let total_edges = totals.edges;
        let max_hub_degree = totals.max_hub_degree;
        let max_hub_ratio = totals.max_hub_ratio;
        let local_percent = totals.local_fraction() * 100.0;
        let mean_norm_gap = totals.mean_norm_gap();
        let active_percent = totals.active_fraction() * 100.0;
        let mean_abs_delta = totals.mean_abs_delta();
        let mean_activation_density = totals.mean_activation_density() * 100.0;
        let mean_synapse_density = totals.mean_synapse_density() * 100.0;
        let mean_activation_energy = totals.mean_activation_energy();

        if height >= 40 {
            let summary = format!(
                "GRAPH nodes {:>4} edges {:>4} hub {:>3} ratio {:>5.1}",
                total_nodes, total_edges, max_hub_degree, max_hub_ratio
            );
            draw_text_line(
                image,
                rect.x0 + 8,
                rect.y0 + min(40, height.saturating_sub(30)),
                &summary,
                [188, 204, 220],
            );
        }

        if height >= 56 {
            let summary = format!(
                "SPARSITY active {:>5.1}% syn {:>5.2}% energy {:>4.2}",
                mean_activation_density, mean_synapse_density, mean_activation_energy
            );
            draw_text_line(
                image,
                rect.x0 + 8,
                rect.y0 + min(56, height.saturating_sub(20)),
                &summary,
                [210, 214, 184],
            );
        }

        if height >= 72 {
            let summary = format!(
                "LOCAL edges {:>5.1}% meand {:>4.2} active {:>5.1}%",
                local_percent, mean_norm_gap, active_percent
            );
            draw_text_line(
                image,
                rect.x0 + 8,
                rect.y0 + min(72, height.saturating_sub(10)),
                &summary,
                [196, 220, 196],
            );
        }

        if height >= 88 {
            let summary = format!(
                "PLASTIC d+ {:>3} d- {:>3} mean|d| {:>5.3} val+ {:>3} val- {:>3}",
                totals.delta_pos,
                totals.delta_neg,
                mean_abs_delta,
                totals.value_pos,
                totals.value_neg
            );
            draw_text_line(
                image,
                rect.x0 + 8,
                rect.y0 + min(88, height.saturating_sub(2)),
                &summary,
                [200, 190, 230],
            );
        }
    }

    let (sw_r, sw_g, sw_b) = token_color(frame.token_id);
    let token_swatch = [sw_r, sw_g, sw_b];
    let sparsity_color_sample = sparsity_color(0.75);
    let local_color = local_edge_color(0.02);
    let long_color = local_edge_color(0.65);
    let pos_delta_color = plasticity_color(1.0, 1.0, 1.0, 1.0);
    let neg_delta_color = plasticity_color(-1.0, -1.0, 1.0, 1.0);

    let legend_entries = [
        ("Token", token_swatch),
        ("Sparsity", sparsity_color_sample),
        ("Local Edge", local_color),
        ("Long Edge", long_color),
        ("d+", pos_delta_color),
        ("d-", neg_delta_color),
    ];

    let legend_reserved = if width > 260 && height > 32 {
        let legend_width = 148.min(width.saturating_sub(16));
        let legend_x1 = rect.x1.saturating_sub(8);
        let legend_x0 = legend_x1.saturating_sub(legend_width);
        let legend_height = 6 + legend_entries.len() * 10 + 8;
        let legend_y1 = rect.y0 + min(height.saturating_sub(8), 8 + legend_height);
        let legend_rect = Rect {
            x0: legend_x0,
            y0: rect.y0 + 8,
            x1: legend_x1,
            y1: legend_y1,
        };
        fill_rect(image, legend_rect, [24, 30, 42]);
        draw_border(image, legend_rect, [70, 80, 96]);
        let mut legend_y = legend_rect.y0 + 4;
        draw_text_line(
            image,
            legend_rect.x0 + 6,
            legend_y,
            "Legend",
            [220, 224, 232],
        );
        legend_y += 10;
        for (label, color) in legend_entries {
            if legend_y + 8 >= legend_rect.y1 {
                break;
            }
            let swatch = Rect {
                x0: legend_rect.x0 + 6,
                y0: legend_y,
                x1: legend_rect.x0 + 18,
                y1: min(legend_rect.y1, legend_y + 8),
            };
            fill_rect(image, swatch, color);
            draw_border(image, swatch, [80, 88, 96]);
            draw_text_line(image, legend_rect.x0 + 24, legend_y, label, [220, 220, 220]);
            legend_y += 10;
        }
        rect.x1.saturating_sub(legend_rect.x0)
    } else {
        let swatch_rect = Rect {
            x0: rect.x1.saturating_sub(48),
            y0: rect.y0 + 8,
            x1: rect.x1.saturating_sub(24),
            y1: rect.y0 + min(height, 28),
        };
        fill_rect(image, swatch_rect, token_swatch);
        draw_border(image, swatch_rect, [220, 220, 220]);
        rect.x1.saturating_sub(swatch_rect.x0)
    };

    if height > 56 && !frame.layers.is_empty() {
        let spark_rect = Rect {
            x0: rect.x0 + 8,
            y0: rect.y1.saturating_sub(18),
            x1: rect.x1.saturating_sub(legend_reserved),
            y1: rect.y1.saturating_sub(10),
        };
        fill_rect(image, spark_rect, [28, 34, 46]);
        draw_border(image, spark_rect, [60, 68, 82]);
        let span = spark_rect.width().max(1);
        for (idx, layer) in frame.layers.iter().enumerate() {
            let x = spark_rect.x0 + min(span - 1, idx * span / frame.layers.len().max(1));
            let norm = (layer.attn_entropy.max(0.0).min(4.0) / 4.0) * spark_rect.height() as f32;
            let bar_height = max(1, norm as usize);
            let y_start = spark_rect.y1.saturating_sub(bar_height);
            for y in y_start..spark_rect.y1 {
                put_pixel(image, x, y, [140, 200, 220]);
            }
        }
    }
}

fn render_transcript_panel(context: &VizRenderContext, rect: Rect, image: &mut Array3<u8>) {
    let width = rect.width();
    let height = rect.height();
    fill_rect(image, rect, [14, 18, 28]);
    draw_border(image, rect, [60, 68, 82]);

    if width < 24 || height < 20 {
        return;
    }

    draw_text_line(
        image,
        rect.x0 + 6,
        rect.y0 + 4,
        "Transcript history",
        [196, 208, 228],
    );

    let usable_height = height.saturating_sub(16);
    let chars_per_line = ((width.saturating_sub(12)) / 4).max(4);
    let max_lines = (usable_height / 8).max(1);
    let lines = context.transcript.lines(chars_per_line, max_lines);

    if lines.is_empty() {
        draw_text_line(
            image,
            rect.x0 + 6,
            rect.y0 + 14,
            "Waiting for decoded text...",
            [150, 160, 176],
        );
        return;
    }

    let mut cursor_y = rect.y0 + 14;
    for line in lines {
        if cursor_y + 6 >= rect.y1 {
            break;
        }
        draw_text_line(image, rect.x0 + 6, cursor_y, &line, [216, 220, 210]);
        cursor_y += 8;
    }
}

fn render_scalefree_panel(context: &VizRenderContext, rect: Rect, image: &mut Array3<u8>) {
    let width = rect.width();
    let height = rect.height();
    fill_rect(image, rect, [10, 12, 20]);
    draw_border(image, rect, [58, 64, 90]);

    if width < 48 || height < 48 {
        draw_text_line(
            image,
            rect.x0 + 6,
            rect.y0 + 4,
            "Scale-free panel too small",
            [190, 200, 220],
        );
        return;
    }

    draw_text_line(
        image,
        rect.x0 + 6,
        rect.y0 + 4,
        "Scale-free degree profile",
        [208, 220, 244],
    );

    let totals = &context.activity.totals;
    let layer_series = degree_series_by_layer(context.activity);
    let degree_counts = aggregate_degree_counts(context.activity);
    let aggregated_points = degree_counts_to_log_points(&degree_counts);
    let slope = fit_power_law(&aggregated_points);
    let gamma = slope.map(|(m, _)| (-m).max(0.0));
    let summary = if let Some(gamma) = gamma {
        format!(
            "gamma~{gamma:.2}  max-degree {:>4}  hub-ratio {:>4.2}",
            totals.max_hub_degree, totals.max_hub_ratio
        )
    } else {
        format!(
            "gamma~n/a  max-degree {:>4}  hub-ratio {:>4.2}",
            totals.max_hub_degree, totals.max_hub_ratio
        )
    };
    let mut text_y = rect.y0 + 16;
    draw_text_line(image, rect.x0 + 6, text_y, &summary, [200, 212, 224]);
    text_y += 10;
    draw_text_line(
        image,
        rect.x0 + 6,
        text_y,
        "axes: ln(degree) vs ln(count)",
        [150, 160, 184],
    );
    text_y += 10;

    let hubs = collect_top_hubs(context.activity, 3);
    if !hubs.is_empty() {
        for (layer, node, degree) in hubs {
            let line = format!("Hub L{layer}#{} deg={degree}", node);
            draw_text_line(image, rect.x0 + 6, text_y, &line, [190, 210, 190]);
            text_y += 10;
        }
    }

    let plastic_layers = dominant_plastic_layers(context.activity, 2);
    if !plastic_layers.is_empty() {
        for (layer, mean_delta, samples) in plastic_layers {
            let line = format!("Delta L{layer} avg={mean_delta:.3} ({samples} edges)");
            draw_text_line(image, rect.x0 + 6, text_y, &line, [220, 200, 180]);
            text_y += 10;
        }
    }

    let plot_margin = 4;
    let plot_rect = Rect {
        x0: rect.x0 + plot_margin,
        y0: (text_y + 4).min(rect.y1),
        x1: rect.x1.saturating_sub(plot_margin),
        y1: rect.y1.saturating_sub(plot_margin),
    };

    if plot_rect.height() < 24 || plot_rect.width() < 24 {
        draw_text_line(
            image,
            rect.x0 + 6,
            plot_rect.y0,
            "Not enough room for log-log plot",
            [180, 190, 210],
        );
        return;
    }

    if aggregated_points.len() < 2 {
        draw_text_line(
            image,
            plot_rect.x0 + 6,
            plot_rect.y0 + 4,
            "Insufficient degree diversity",
            [200, 190, 200],
        );
        return;
    }

    let all_points = if layer_series.iter().any(|s| !s.points.is_empty()) {
        layer_series
            .iter()
            .flat_map(|series| series.points.iter().copied())
            .collect::<Vec<_>>()
    } else {
        aggregated_points.clone()
    };
    let (x_min, x_max) = bounds(&all_points, |p| p.0);
    let (y_min, y_max) = bounds(&all_points, |p| p.1);
    let mut x0 = x_min;
    let mut x1 = x_max;
    let mut y0 = y_min;
    let mut y1 = y_max;
    if (x1 - x0).abs() < 0.25 {
        x0 -= 0.5;
        x1 += 0.5;
    }
    if (y1 - y0).abs() < 0.25 {
        y0 -= 0.5;
        y1 += 0.5;
    }

    let mut legend_entries: Vec<(u8, RGBColor)> = Vec::new();
    paint_plot(image, plot_rect, |area| {
        let mut chart = ChartBuilder::on(&area)
            .margin_left(32)
            .margin_right(10)
            .margin_top(6)
            .margin_bottom(28)
            .x_label_area_size(24)
            .y_label_area_size(32)
            .build_cartesian_2d(x0..x1, y0..y1)?;

        chart
            .configure_mesh()
            .x_labels(0)
            .y_labels(0)
            .axis_style(&RGBColor(90, 96, 120))
            .light_line_style(&RGBColor(40, 44, 60))
            .draw()?;

        if layer_series.iter().any(|series| !series.points.is_empty()) {
            for (idx, series) in layer_series.iter().enumerate() {
                if series.points.is_empty() {
                    continue;
                }
                let color = Palette99::pick(idx);
                let tone = color.mix(0.95);
                let rgb = RGBColor(tone.0, tone.1, tone.2);
                legend_entries.push((series.layer, rgb));
                chart.draw_series(
                    series
                        .points
                        .iter()
                        .map(|(x, y)| Circle::new((*x, *y), 3, tone.filled())),
                )?;
            }
        } else {
            chart.draw_series(
                aggregated_points
                    .iter()
                    .map(|(x, y)| Circle::new((*x, *y), 3, RGBColor(236, 196, 128).filled())),
            )?;
        }

        if let Some((slope, intercept)) = slope {
            let line = vec![(x0, slope * x0 + intercept), (x1, slope * x1 + intercept)];
            chart.draw_series(std::iter::once(PathElement::new(
                line,
                ShapeStyle::from(&RGBColor(120, 210, 255)).stroke_width(2),
            )))?;
        }

        Ok(())
    });

    if plot_rect.y0 > rect.y0 + 18 {
        draw_text_line(
            image,
            rect.x0 + 6,
            plot_rect.y0.saturating_sub(10),
            "ln count ^",
            [150, 160, 184],
        );
    }
    draw_text_line(
        image,
        plot_rect.x1.saturating_sub(70),
        plot_rect.y1 + 2,
        "ln degree ->",
        [150, 160, 184],
    );

    if !legend_entries.is_empty() {
        let legend_x = plot_rect.x0 + 6;
        let mut legend_y = plot_rect.y0 + 6;
        let max_entries = legend_entries.len().min(6);
        for (idx, (layer, color)) in legend_entries.iter().enumerate().take(max_entries) {
            let swatch = Rect {
                x0: legend_x,
                y0: legend_y,
                x1: legend_x + 8,
                y1: legend_y + 6,
            };
            fill_rect(image, swatch, [color.0, color.1, color.2]);
            draw_border(image, swatch, [40, 44, 60]);
            draw_text_line(
                image,
                legend_x + 10,
                legend_y,
                &format!("Layer {}", layer),
                [200, 210, 226],
            );
            legend_y += 8;
            if idx + 1 == max_entries && legend_entries.len() > max_entries {
                draw_text_line(image, legend_x + 10, legend_y, "+ more...", [170, 180, 196]);
            }
        }
    }
}
fn render_eeg_panel(context: &VizRenderContext, rect: Rect, mode: EegMode, image: &mut Array3<u8>) {
    let width = rect.width();
    let height = rect.height();
    let theme = mode.theme();

    fill_rect(image, rect, theme.panel_bg.to_array());
    draw_border(image, rect, theme.grid.to_array());

    if width < 32 || height < EEG_TITLE_GAP + 8 || context.history.layers() == 0 {
        draw_text_line(
            image,
            rect.x0 + 6,
            rect.y0 + 4,
            "EEG view unavailable",
            theme.title_color.to_array(),
        );
        return;
    }

    draw_text_line(
        image,
        rect.x0 + 6,
        rect.y0 + 4,
        theme.title,
        theme.title_color.to_array(),
    );

    let plot_rect = Rect {
        x0: rect.x0 + 6,
        y0: rect.y0 + EEG_TITLE_GAP,
        x1: rect.x1.saturating_sub(4),
        y1: rect.y1.saturating_sub(6),
    };

    if plot_rect.width() < 16 || plot_rect.height() < 12 {
        draw_text_line(
            image,
            rect.x0 + 6,
            plot_rect.y0,
            "EEG panel too small",
            theme.title_color.to_array(),
        );
        return;
    }

    let bands = build_eeg_bands(context.history, context.layer_count(), MAX_EEG_BANDS);
    if bands.is_empty() {
        draw_text_line(
            image,
            rect.x0 + 6,
            plot_rect.y0,
            "Waiting for neuron history...",
            theme.title_color.to_array(),
        );
        return;
    }
    let energy_series = aggregate_energy(&bands);
    let x_max = context.history.capacity().max(1) as f32;
    let energy_available = !energy_series.is_empty();

    paint_plot(image, plot_rect, |area| {
        let y_max = bands.len().max(1) as f32;
        let mut chart = ChartBuilder::on(&area)
            .margin_left(18)
            .margin_right(8)
            .margin_top(6)
            .margin_bottom(18)
            .x_label_area_size(16)
            .y_label_area_size(0)
            .build_cartesian_2d(0f32..x_max, 0f32..y_max)?;

        chart
            .configure_mesh()
            .disable_mesh()
            .x_labels(0)
            .y_labels(0)
            .axis_style(&theme.grid.to_rgb())
            .light_line_style(&theme.grid.to_rgb())
            .draw()?;

        for idx in 1..bands.len() {
            let y = idx as f32;
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(0f32, y), (x_max, y)],
                ShapeStyle::from(&theme.grid.to_rgb()).stroke_width(1),
            )))?;
        }

        for (band_idx, band) in bands.iter().enumerate() {
            let base = band_idx as f32;
            for (sample_idx, sample) in band.samples.iter().enumerate() {
                let metric = mode.metric_value(sample).clamp(0.0, 1.0);
                let color = theme.heat_color(metric).to_rgb();
                let patch = Rectangle::new(
                    [
                        (sample_idx as f32, base),
                        ((sample_idx + 1) as f32, base + 0.95),
                    ],
                    color.filled(),
                );
                chart.draw_series(std::iter::once(patch))?;
            }

            let trace = band
                .samples
                .iter()
                .enumerate()
                .map(|(idx, sample)| {
                    let amplitude = mode.metric_value(sample).clamp(0.0, 1.0);
                    (idx as f32, base + 0.5 + (amplitude - 0.5) * 0.8)
                })
                .collect::<Vec<_>>();
            chart.draw_series(std::iter::once(PathElement::new(
                trace,
                ShapeStyle::from(&theme.trace.to_rgb()).stroke_width(1),
            )))?;
        }

        if energy_available {
            let energy_max = energy_series
                .iter()
                .copied()
                .fold(0f32, f32::max)
                .max(f32::EPSILON);
            let overlay = energy_series
                .iter()
                .enumerate()
                .map(|(idx, energy)| (idx as f32, (energy / energy_max).clamp(0.0, 1.0) * y_max))
                .collect::<Vec<_>>();
            chart.draw_series(std::iter::once(PathElement::new(
                overlay,
                ShapeStyle::from(&theme.overlay.to_rgb()).stroke_width(2),
            )))?;
        }

        Ok(())
    });

    if energy_available {
        let label_x = rect.x1.saturating_sub(160);
        draw_text_line(
            image,
            label_x,
            rect.y0 + 4,
            "Energy envelope",
            theme.overlay.to_array(),
        );
    }

    if bands.len() <= 32 {
        let rows = bands.len().max(1);
        let row_height = plot_rect.height() as f32 / rows as f32;
        for (idx, band) in bands.iter().enumerate() {
            let label = if band.start_layer == band.end_layer {
                format!("L{}", band.start_layer)
            } else {
                format!("L{}-{}", band.start_layer, band.end_layer)
            };
            let y = (plot_rect.y0 as f32 + row_height * idx as f32 + row_height * 0.2) as usize;
            draw_text_line(
                image,
                rect.x0 + 8,
                min(plot_rect.y1.saturating_sub(10), y),
                &label,
                theme.title_color.to_array(),
            );
        }
    }
}

fn build_eeg_bands(history: &EegHistory, visible_layers: usize, max_bands: usize) -> Vec<EegBand> {
    if history.capacity() == 0 || history.layers() == 0 || visible_layers == 0 || max_bands == 0 {
        return Vec::new();
    }

    let bounded_layers = visible_layers.max(1).min(history.layers().max(1));
    let target_bands = max_bands.max(1).min(bounded_layers);
    let chunk = (bounded_layers + target_bands - 1) / target_bands;
    let mut bands = Vec::with_capacity(target_bands);

    for band_idx in 0..target_bands {
        let start = band_idx * chunk;
        if start >= bounded_layers {
            break;
        }
        let end = min(bounded_layers, start + chunk);
        let mut samples = vec![EegSample::default(); history.capacity()];
        for layer in start..end {
            let series = history.series(layer);
            for (idx, sample) in series.iter().enumerate() {
                samples[idx].activation += sample.activation;
                samples[idx].synapse += sample.synapse;
                samples[idx].energy += sample.energy;
                samples[idx].plasticity += sample.plasticity;
            }
        }
        let denom = (end - start) as f32;
        if denom > 0.0 {
            for sample in &mut samples {
                sample.activation = (sample.activation / denom).clamp(0.0, 1.0);
                sample.synapse = (sample.synapse / denom).clamp(0.0, 1.0);
                sample.energy = (sample.energy / denom).max(0.0);
                sample.plasticity = (sample.plasticity / denom).max(0.0);
            }
        }
        bands.push(EegBand {
            start_layer: start,
            end_layer: end.saturating_sub(1),
            samples,
        });
    }

    bands
}

fn aggregate_energy(bands: &[EegBand]) -> Vec<f32> {
    if bands.is_empty() {
        return Vec::new();
    }
    let len = bands.first().map(|band| band.samples.len()).unwrap_or(0);
    if len == 0 {
        return Vec::new();
    }
    let mut totals = vec![0f32; len];
    for band in bands {
        for (idx, sample) in band.samples.iter().enumerate() {
            totals[idx] += sample.energy.max(0.0);
        }
    }
    let denom = bands.len().max(1) as f32;
    for value in &mut totals {
        *value = (*value / denom).max(0.0);
    }
    totals
}

fn compute_panel_heights(modules: &[LayoutModule], total_height: usize) -> Vec<usize> {
    if modules.is_empty() {
        return Vec::new();
    }
    if total_height == 0 {
        return vec![0; modules.len()];
    }

    let mut heights = vec![0usize; modules.len()];
    let mut remaining = total_height;
    for (idx, module) in modules.iter().enumerate() {
        if let PanelSize::Fixed(px) = module.size {
            let assigned = px.min(remaining as u32) as usize;
            heights[idx] = assigned;
            remaining = remaining.saturating_sub(assigned);
        }
    }

    let flex_modules: Vec<(usize, f32)> = modules
        .iter()
        .enumerate()
        .filter_map(|(idx, module)| match module.size {
            PanelSize::Flex(weight) => Some((idx, if weight <= 0.0 { 1.0 } else { weight })),
            _ => None,
        })
        .collect();

    let flex_space = remaining;
    if flex_space > 0 && !flex_modules.is_empty() {
        let total_weight: f32 = flex_modules
            .iter()
            .map(|(_, w)| *w)
            .sum::<f32>()
            .max(f32::EPSILON);
        let mut undistributed = flex_space;
        for (pos, (idx, weight)) in flex_modules.iter().enumerate() {
            if undistributed == 0 {
                break;
            }
            let modules_left = flex_modules.len() - pos;
            let mut portion = ((flex_space as f32) * (weight / total_weight)).round() as usize;
            if portion == 0 {
                portion = (undistributed / modules_left.max(1)).max(1);
            }
            portion = portion.min(undistributed);
            heights[*idx] = portion;
            undistributed = undistributed.saturating_sub(portion);
        }
        if undistributed > 0 {
            if let Some((idx, _)) = flex_modules.last() {
                heights[*idx] += undistributed;
            }
        }
    }

    let assigned: usize = heights.iter().sum();
    if assigned < total_height {
        if let Some(last) = heights.last_mut() {
            *last += total_height - assigned;
        }
    }

    heights
}

fn aggregate_degree_counts(activity: &FrameActivity) -> Vec<(u32, usize)> {
    let mut map: BTreeMap<u32, usize> = BTreeMap::new();
    for layer in &activity.layers {
        for (degree, count) in &layer.degree_spectrum {
            if *degree == 0 || *count == 0 {
                continue;
            }
            *map.entry(*degree).or_default() += *count;
        }
    }
    map.into_iter().collect()
}

fn degree_counts_to_log_points(counts: &[(u32, usize)]) -> Vec<(f32, f32)> {
    counts
        .iter()
        .filter_map(|(degree, count)| {
            if *degree == 0 || *count == 0 {
                None
            } else {
                let x = (*degree as f32).max(1.0).ln();
                let y = (*count as f32).max(1.0).ln();
                if !x.is_finite() || !y.is_finite() {
                    None
                } else {
                    Some((x, y))
                }
            }
        })
        .collect()
}

fn degree_series_by_layer(activity: &FrameActivity) -> Vec<LayerDegreeSeries> {
    let mut series = Vec::new();
    for layer in &activity.layers {
        let points = layer
            .degree_spectrum
            .iter()
            .filter_map(|(degree, count)| {
                if *degree == 0 || *count == 0 {
                    None
                } else {
                    let x = (*degree as f32).max(1.0).ln();
                    let y = (*count as f32).max(1.0).ln();
                    if x.is_finite() && y.is_finite() {
                        Some((x, y))
                    } else {
                        None
                    }
                }
            })
            .collect::<Vec<_>>();
        if points.is_empty() {
            continue;
        }
        series.push(LayerDegreeSeries {
            layer: layer.layer_id,
            points,
        });
    }
    series
}

fn fit_power_law(points: &[(f32, f32)]) -> Option<(f32, f32)> {
    if points.len() < 2 {
        return None;
    }
    let n = points.len() as f32;
    let sum_x: f32 = points.iter().map(|(x, _)| *x).sum();
    let sum_y: f32 = points.iter().map(|(_, y)| *y).sum();
    let sum_xx: f32 = points.iter().map(|(x, _)| x * x).sum();
    let sum_xy: f32 = points.iter().map(|(x, y)| x * y).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() <= f32::EPSILON {
        return None;
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    if slope.is_finite() && intercept.is_finite() {
        Some((slope, intercept))
    } else {
        None
    }
}

fn bounds(points: &[(f32, f32)], getter: impl Fn(&(f32, f32)) -> f32) -> (f32, f32) {
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for p in points {
        let v = getter(p);
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }
    if !min_val.is_finite() || !max_val.is_finite() {
        (0.0, 1.0)
    } else {
        (min_val, max_val)
    }
}

fn collect_top_hubs(activity: &FrameActivity, limit: usize) -> Vec<(u8, u32, usize)> {
    let mut hubs = Vec::new();
    for layer in &activity.layers {
        for (node, degree) in &layer.top_hubs {
            hubs.push((layer.layer_id, *node, *degree));
        }
    }
    hubs.sort_by(|a, b| b.2.cmp(&a.2));
    hubs.truncate(limit);
    hubs
}

fn dominant_plastic_layers(activity: &FrameActivity, limit: usize) -> Vec<(u8, f32, usize)> {
    let mut layers = activity
        .layers
        .iter()
        .map(|layer| (layer.layer_id, layer.mean_abs_delta, layer.delta_samples))
        .collect::<Vec<_>>();
    layers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    layers
        .into_iter()
        .filter(|(_, mean_delta, _)| *mean_delta > 0.0)
        .take(limit)
        .collect()
}

fn sparsity_color(norm: f32) -> [u8; 3] {
    let n = norm.clamp(0.0, 1.0);
    [
        (80.0 + 140.0 * n) as u8,
        (130.0 + 80.0 * n) as u8,
        (150.0 - 60.0 * n) as u8,
    ]
}

fn local_edge_color(norm: f32) -> [u8; 3] {
    let n = norm.clamp(0.0, 1.0);
    let inv = 1.0 - n;
    [
        (100.0 + 120.0 * n) as u8,
        (200.0 * inv + 60.0 * n) as u8,
        (150.0 * inv + 180.0 * n) as u8,
    ]
}

fn plasticity_color(delta: f32, value: f32, max_value: f32, max_delta: f32) -> [u8; 3] {
    if max_delta <= f32::EPSILON {
        return [170, 170, 180];
    }
    let delta_norm = (delta / max_delta).clamp(-1.0, 1.0);
    let value_norm = if max_value <= f32::EPSILON {
        0.0
    } else {
        (value.abs() / max_value).clamp(0.0, 1.0)
    };
    let magnitude = delta_norm.abs();
    if delta_norm >= 0.0 {
        [
            (190.0 + 50.0 * magnitude) as u8,
            (120.0 + 40.0 * (1.0 - value_norm)) as u8,
            (70.0 + 20.0 * (1.0 - magnitude)) as u8,
        ]
    } else {
        [
            (80.0 + 30.0 * (1.0 - magnitude)) as u8,
            (120.0 + 30.0 * (1.0 - value_norm)) as u8,
            (180.0 + 60.0 * magnitude) as u8,
        ]
    }
}

fn token_color(token_id: u32) -> (u8, u8, u8) {
    let hash = (token_id as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let r = ((hash >> 40) & 0xFF) as u8;
    let g = ((hash >> 16) & 0xFF) as u8;
    let b = ((hash >> 8) & 0xFF) as u8;
    (
        r.saturating_div(2).saturating_add(96),
        g.saturating_div(2).saturating_add(64),
        b.saturating_div(2).saturating_add(48),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viz::activity::ActivityComputer;
    use crate::viz::frame::{AttnEdge, LayerFrame, TokenFrame};

    #[test]
    fn eeg_bands_group_layers_and_average_metrics() {
        let mut history = EegHistory::new(0, 2);
        let samples = [
            (
                EegSample {
                    activation: 0.2,
                    synapse: 0.5,
                    energy: 0.1,
                    plasticity: 0.05,
                },
                EegSample {
                    activation: 0.8,
                    synapse: 0.3,
                    energy: 0.2,
                    plasticity: 0.1,
                },
            ),
            (
                EegSample {
                    activation: 0.4,
                    synapse: 0.5,
                    energy: 0.3,
                    plasticity: 0.2,
                },
                EegSample {
                    activation: 0.6,
                    synapse: 0.3,
                    energy: 0.4,
                    plasticity: 0.12,
                },
            ),
        ];
        for (left, right) in samples {
            history.record(0, left);
            history.record(1, right);
        }

        let bands = build_eeg_bands(&history, 2, 1);
        assert_eq!(bands.len(), 1);
        let band = &bands[0];
        assert_eq!((band.start_layer, band.end_layer), (0, 1));
        let last_sample = band.samples.last().unwrap();
        assert!((last_sample.activation - 0.5).abs() < 1e-6);
        assert!((last_sample.synapse - 0.4).abs() < 1e-6);
        assert!((last_sample.energy - 0.35).abs() < 1e-6);
        assert!((last_sample.plasticity - 0.16).abs() < 1e-6);
    }

    #[test]
    fn energy_envelope_is_average_over_bands() {
        let bands = vec![
            EegBand {
                start_layer: 0,
                end_layer: 0,
                samples: vec![
                    EegSample {
                        energy: 0.2,
                        ..EegSample::default()
                    },
                    EegSample {
                        energy: 0.4,
                        ..EegSample::default()
                    },
                ],
            },
            EegBand {
                start_layer: 1,
                end_layer: 1,
                samples: vec![
                    EegSample {
                        energy: 0.3,
                        ..EegSample::default()
                    },
                    EegSample {
                        energy: 0.5,
                        ..EegSample::default()
                    },
                ],
            },
        ];

        let envelope = aggregate_energy(&bands);
        assert_eq!(envelope.len(), 2);
        assert!((envelope[0] - 0.25).abs() < 1e-6);
        assert!((envelope[1] - 0.45).abs() < 1e-6);
    }

    #[test]
    fn panel_layout_respects_fixed_sections() {
        let modules = vec![
            LayoutModule::fixed(PanelKind::TopBar, 40),
            LayoutModule::flex(PanelKind::Eeg(EegMode::Activation), 1.0),
            LayoutModule::flex(PanelKind::Eeg(EegMode::Plasticity), 2.0),
        ];
        let heights = compute_panel_heights(&modules, 160);
        assert_eq!(heights.len(), 3);
        assert_eq!(heights[0], 40);
        assert_eq!(heights.iter().sum::<usize>(), 160);
        assert!(heights[1] < heights[2]);
        assert!(heights[1] > 0 && heights[2] > 0);
    }

    #[test]
    fn transcript_history_wraps_and_limits_lines() {
        let mut transcript = TranscriptHistory::new(64);
        transcript.record("dragon hatchlings dream in sparse pulses");
        let lines = transcript.lines(8, 3);
        assert!(!lines.is_empty());
        assert!(lines.len() <= 3);
        let stitched: String = lines.concat();
        assert!(stitched.contains("pulses"));
    }

    fn build_layer(layer_id: u8, edges: &[(u32, u32)]) -> LayerFrame {
        LayerFrame {
            layer: layer_id,
            attn_edges: edges
                .iter()
                .enumerate()
                .map(|(idx, (from, to))| AttnEdge {
                    from: *from,
                    to: *to,
                    head: (idx % 2) as u8,
                    w: 0.5,
                })
                .collect(),
            hot_neurons: Vec::new(),
            syn_edges: Vec::new(),
            attn_entropy: 0.0,
        }
    }

    #[test]
    fn degree_series_tracks_each_layer() {
        let mut computer = ActivityComputer::new();
        let frame = TokenFrame {
            t: 0,
            token_id: 0,
            token_text: "test".into(),
            layers: vec![
                build_layer(0, &[(0, 1), (1, 2)]),
                build_layer(1, &[(0, 3), (3, 4), (4, 0)]),
            ],
        };
        let activity = computer.compute(&frame);
        let series = degree_series_by_layer(&activity);
        assert!(series.iter().any(|s| s.layer == 0 && !s.points.is_empty()));
        assert!(series.iter().any(|s| s.layer == 1 && !s.points.is_empty()));
    }
}
