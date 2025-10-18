use std::cmp::{Ordering, max, min};
use std::collections::HashSet;
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
use tempfile::NamedTempFile;

use super::VideoConfig;
use super::frame::{LayerFrame, TokenFrame};

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
    tx: Arc<Mutex<Option<Sender<TokenFrame>>>>,
    join: Option<JoinHandle<()>>,
    errors: Arc<Mutex<Option<anyhow::Error>>>,
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
    pub fn start(config: VideoConfig) -> Result<Self> {
        let (tx_raw, rx) = unbounded();
        let errors = Arc::new(Mutex::new(None));
        let thread_errors = errors.clone();
        let tx = Arc::new(Mutex::new(Some(tx_raw)));

        eprintln!(
            "viz: starting video encoder thread -> {} ({}x{} @ {}fps)",
            config.path.display(),
            config.width,
            config.height,
            config.fps
        );

        let thread_config = config.clone();
        let join = thread::Builder::new()
            .name("bdh-viz-encoder".into())
            .spawn(move || {
                if let Err(err) = run_encoder(rx, thread_config) {
                    *thread_errors.lock() = Some(err);
                }
            })
            .context("failed to spawn viz encoder thread")?;

        Ok(Self {
            tx,
            join: Some(join),
            errors,
        })
    }

    pub fn push(&self, frame: TokenFrame) {
        let mut guard = self.tx.lock();
        if let Some(sender) = guard.as_ref() {
            if sender.send(frame).is_err() {
                eprintln!("viz: encoder channel closed; disabling visualization output");
                *guard = None;
            }
        }
    }
}

impl Drop for VideoHandle {
    fn drop(&mut self) {
        self.tx.lock().take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
        if let Some(err) = self.errors.lock().take() {
            eprintln!("viz encoder error: {err:#}");
        }
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

fn run_encoder(rx: Receiver<TokenFrame>, config: VideoConfig) -> Result<()> {
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
    for frame in rx.iter() {
        let image = render_frame(&frame, width, height);
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

fn render_frame(frame: &TokenFrame, width: usize, height: usize) -> Array3<u8> {
    let safe_width = width.max(1);
    let safe_height = height.max(1);
    let mut image = Array3::<u8>::from_elem((safe_height, safe_width, 3), 10);

    if width == 0 || height == 0 {
        return image;
    }

    let top_height = min(height, max(48, height / 6));
    render_top_bar(&mut image, frame, top_height);

    let mut y = top_height;
    let layer_count = frame.layers.len().max(1);
    for (index, layer) in frame.layers.iter().enumerate() {
        let remaining_layers = layer_count - index;
        let remaining_height = height.saturating_sub(y);
        if remaining_height == 0 {
            break;
        }

        let mut block_height = remaining_height / remaining_layers;
        block_height = block_height.max(72).min(remaining_height);
        let y_end = if index == layer_count - 1 {
            height
        } else {
            min(height, y + block_height)
        };

        render_layer_block(
            &mut image,
            Rect {
                x0: 0,
                y0: y,
                x1: width,
                y1: y_end,
            },
            layer,
            index,
            frame.t,
        );
        y = y_end;
    }

    image
}

fn render_top_bar(image: &mut Array3<u8>, frame: &TokenFrame, height: usize) {
    let (width, _) = image_dims(image);
    let top_rect = Rect {
        x0: 0,
        y0: 0,
        x1: width,
        y1: height,
    };
    fill_rect(image, top_rect, [18, 24, 32]);
    draw_border(image, top_rect, [60, 68, 82]);

    let info = format!("STEP {:>6}   TOKEN {:>7}", frame.t, frame.token_id);
    draw_text_line(
        image,
        8,
        min(8, height.saturating_sub(6)),
        &info,
        [232, 232, 224],
    );

    if height >= 24 {
        let text = sanitized_token_text(&frame.token_text, (width / 6).max(12));
        draw_text_line(
            image,
            8,
            min(24, height.saturating_sub(6)),
            &format!("TEXT {}", text),
            [210, 210, 180],
        );
    }

    if !frame.layers.is_empty() {
        let layer_count = frame.layers.len();
        let attn_edges_total: usize = frame
            .layers
            .iter()
            .map(|layer| layer.attn_edges.len())
            .sum();
        let neuron_total: usize = frame
            .layers
            .iter()
            .map(|layer| layer.hot_neurons.len())
            .sum();
        let syn_edge_total: usize = frame.layers.iter().map(|layer| layer.syn_edges.len()).sum();
        let positive_syn = frame
            .layers
            .iter()
            .flat_map(|layer| layer.syn_edges.iter())
            .filter(|edge| edge.value.is_sign_positive())
            .count();
        let negative_syn = syn_edge_total.saturating_sub(positive_syn);

        let unique_targets: HashSet<u32> = frame
            .layers
            .iter()
            .flat_map(|layer| layer.attn_edges.iter().map(|edge| edge.to))
            .collect();
        let unique_synapses: HashSet<(u32, u32)> = frame
            .layers
            .iter()
            .flat_map(|layer| layer.syn_edges.iter().map(|edge| (edge.i, edge.j)))
            .collect();

        let avg_entropy = frame
            .layers
            .iter()
            .map(|layer| layer.attn_entropy)
            .sum::<f32>()
            / layer_count as f32;

        if height >= 40 {
            let summary = format!(
                "ATT {:>4} uniq {:>4}  ⌀H {:.2}",
                attn_edges_total,
                unique_targets.len(),
                avg_entropy
            );
            draw_text_line(
                image,
                8,
                min(40, height.saturating_sub(18)),
                &summary,
                [188, 204, 220],
            );
        }

        if height >= 56 {
            let summary = format!(
                "NEU {:>4}  SYN {:>4} (+{:>3}/-{:<3})",
                neuron_total, syn_edge_total, positive_syn, negative_syn
            );
            draw_text_line(
                image,
                8,
                min(56, height.saturating_sub(10)),
                &summary,
                [196, 220, 196],
            );
        }

        if height >= 70 {
            let summary = format!("UNIQUE SYN {:>4}", unique_synapses.len());
            draw_text_line(
                image,
                8,
                min(70, height.saturating_sub(2)),
                &summary,
                [200, 190, 230],
            );
        }
    }

    let swatch_rect = Rect {
        x0: width.saturating_sub(48),
        y0: 8,
        x1: width.saturating_sub(24),
        y1: min(height, 28),
    };
    let (sw_r, sw_g, sw_b) = token_color(frame.token_id);
    fill_rect(image, swatch_rect, [sw_r, sw_g, sw_b]);
    draw_border(image, swatch_rect, [220, 220, 220]);

    let attn_color = attention_color(0.85);
    let neuron_color_tone = neuron_color(0.9);
    let pos_color = synapse_color(1.0, 1.0);
    let neg_color = synapse_color(-1.0, 1.0);

    if width > 220 {
        let legend_x = width.saturating_sub(120);
        let mut legend_y = 8usize;
        for (label, color) in [
            ("ATT", attn_color),
            ("NEU", neuron_color_tone),
            ("+syn", pos_color),
            ("-syn", neg_color),
        ] {
            let swatch = Rect {
                x0: legend_x,
                y0: legend_y,
                x1: legend_x + 12,
                y1: legend_y + 6,
            };
            fill_rect(image, swatch, color);
            draw_border(image, swatch, [80, 88, 96]);
            draw_text_line(image, legend_x + 16, legend_y, label, [220, 220, 220]);
            legend_y += 10;
        }
    }

    if height > 56 && !frame.layers.is_empty() {
        let spark_rect = Rect {
            x0: 8,
            y0: height.saturating_sub(18),
            x1: width.saturating_sub(48),
            y1: height.saturating_sub(10),
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

fn render_layer_block(
    image: &mut Array3<u8>,
    rect: Rect,
    layer: &LayerFrame,
    index: usize,
    step_t: u32,
) {
    if rect.width() < 24 || rect.height() < 16 {
        return;
    }

    fill_rect(image, rect, [14, 18, 26]);
    draw_border(image, rect, [48, 54, 66]);

    let label_width = min(72, rect.width().min(96));
    let label_rect = Rect {
        x0: rect.x0,
        y0: rect.y0,
        x1: rect.x0 + label_width,
        y1: rect.y1,
    };
    fill_rect(image, label_rect, [24, 30, 42]);
    draw_border(image, label_rect, [60, 70, 84]);

    draw_text_line(
        image,
        label_rect.x0 + 6,
        rect.y0 + 6,
        &format!("L{}", index),
        [230, 210, 170],
    );
    draw_text_line(
        image,
        label_rect.x0 + 6,
        rect.y0 + 22,
        &format!("H{}", layer.attn_edges.len()),
        [180, 200, 220],
    );
    draw_text_line(
        image,
        label_rect.x0 + 6,
        rect.y0 + 34,
        &format!("N{}", layer.hot_neurons.len()),
        [200, 220, 180],
    );
    draw_text_line(
        image,
        label_rect.x0 + 6,
        rect.y0 + 46,
        &format!("S{}", layer.syn_edges.len()),
        [216, 190, 220],
    );

    let unique_targets = layer
        .attn_edges
        .iter()
        .map(|edge| edge.to)
        .collect::<HashSet<_>>()
        .len();
    let syn_positive = layer
        .syn_edges
        .iter()
        .filter(|edge| edge.value.is_sign_positive())
        .count();
    let syn_negative = layer.syn_edges.len().saturating_sub(syn_positive);
    draw_text_line(
        image,
        label_rect.x0 + 6,
        rect.y0 + 58,
        &format!("T{} +{} -{}", unique_targets, syn_positive, syn_negative),
        [204, 210, 236],
    );

    let remaining_width = rect.width().saturating_sub(label_width);
    if remaining_width < 16 {
        return;
    }

    let attn_width = max(48, (remaining_width as f32 * 0.45).round() as usize);
    let neuron_width = max(40, (remaining_width as f32 * 0.30).round() as usize);
    let mut x_cursor = label_rect.x1;

    let attn_rect = Rect {
        x0: x_cursor,
        y0: rect.y0,
        x1: min(rect.x1, x_cursor + attn_width),
        y1: rect.y1,
    };
    render_attention_panel(image, attn_rect, layer, step_t);
    x_cursor = attn_rect.x1;

    let neuron_rect = Rect {
        x0: x_cursor,
        y0: rect.y0,
        x1: min(rect.x1, x_cursor + neuron_width),
        y1: rect.y1,
    };
    render_neuron_panel(image, neuron_rect, layer);
    x_cursor = neuron_rect.x1;

    let syn_rect = Rect {
        x0: x_cursor,
        y0: rect.y0,
        x1: rect.x1,
        y1: rect.y1,
    };
    render_synapse_panel(image, syn_rect, layer);
}

fn render_attention_panel(image: &mut Array3<u8>, rect: Rect, layer: &LayerFrame, step_t: u32) {
    fill_rect(image, rect, [18, 20, 32]);
    draw_border(image, rect, [60, 68, 82]);
    if rect.width() == 0 || rect.height() == 0 {
        return;
    }

    let max_weight = layer
        .attn_edges
        .iter()
        .map(|edge| edge.w.abs())
        .fold(0.0_f32, f32::max);
    if max_weight <= f32::EPSILON {
        return;
    }

    let head_count = layer
        .attn_edges
        .iter()
        .map(|edge| edge.head as usize)
        .max()
        .map(|v| v + 1)
        .unwrap_or(1)
        .max(1);
    let row_height = max(1, rect.height() / head_count);

    for head in 0..head_count {
        let y0 = rect.y0 + head * row_height;
        let y1 = if head == head_count - 1 {
            rect.y1
        } else {
            min(rect.y1, y0 + row_height)
        };
        for x in rect.x0..rect.x1 {
            put_pixel(image, x, y0, [40, 44, 60]);
            if y1 > rect.y0 {
                put_pixel(image, x, y1.saturating_sub(1), [32, 36, 52]);
            }
        }
        draw_text_line(
            image,
            rect.x0 + 4,
            min(y0 + 2, rect.y1.saturating_sub(6)),
            &format!("H{}", head),
            [150, 160, 200],
        );
    }

    let denominator = step_t.max(1) as f32;

    for edge in &layer.attn_edges {
        let head_idx = min(head_count - 1, edge.head as usize);
        let normalized_target = (edge.to as f32) / denominator;
        let x = rect.x0
            + min(
                rect.width().saturating_sub(1),
                (normalized_target.clamp(0.0, 1.0) * rect.width() as f32) as usize,
            );
        let y0 = rect.y0 + head_idx * row_height;
        let y1 = if head_idx == head_count - 1 {
            rect.y1
        } else {
            min(rect.y1, y0 + row_height)
        };
        let color = attention_color((edge.w.abs() / max_weight).clamp(0.0, 1.0));
        for y in y0..y1 {
            put_pixel(image, x, y, color);
            if x + 1 < rect.x1 {
                put_pixel(image, x + 1, y, [color[0] / 2, color[1] / 2, color[2] / 2]);
            }
        }
    }

    let attention_targets: HashSet<u32> = layer.attn_edges.iter().map(|edge| edge.to).collect();
    if rect.height() > 16 {
        let density = if step_t == 0 {
            0.0
        } else {
            attention_targets.len() as f32 / step_t.max(1) as f32
        };
        let summary = format!("uniq {:>3} dens {:.2}", attention_targets.len(), density);
        draw_text_line(
            image,
            rect.x0 + 4,
            rect.y1.saturating_sub(10),
            &summary,
            [170, 188, 220],
        );
    }
}

fn render_neuron_panel(image: &mut Array3<u8>, rect: Rect, layer: &LayerFrame) {
    fill_rect(image, rect, [18, 26, 30]);
    draw_border(image, rect, [60, 74, 70]);
    if rect.width() == 0 || rect.height() == 0 || layer.hot_neurons.is_empty() {
        return;
    }

    let mut neurons = layer.hot_neurons.clone();
    neurons.sort_by(|a, b| b.act.partial_cmp(&a.act).unwrap_or(Ordering::Equal));
    let max_act = neurons
        .iter()
        .map(|n| n.act)
        .fold(0.0_f32, f32::max)
        .max(f32::EPSILON);

    let max_columns = rect.width().saturating_div(4).max(1);
    let count = min(neurons.len(), max_columns);
    if count == 0 {
        return;
    }
    let bar_width = max(3, rect.width() / count.max(1));

    if let Some(top) = neurons.first() {
        draw_text_line(
            image,
            rect.x0 + 4,
            rect.y0 + 4,
            &format!("peak {}:{:.2}", top.id_or_cluster, top.act),
            [210, 236, 200],
        );
    }

    for (idx, neuron) in neurons.iter().take(count).enumerate() {
        let x0 = rect.x0 + idx * bar_width;
        let x1 = min(rect.x1, x0 + bar_width - 1);
        let norm = (neuron.act / max_act).clamp(0.0, 1.0);
        let bar_height = max(1, (norm * rect.height() as f32) as usize);
        let y0 = rect.y1.saturating_sub(bar_height);
        let color = neuron_color(norm);
        for y in y0..rect.y1 {
            for x in x0..=x1 {
                put_pixel(image, x, y, color);
            }
        }
    }
}

fn render_synapse_panel(image: &mut Array3<u8>, rect: Rect, layer: &LayerFrame) {
    fill_rect(image, rect, [20, 24, 36]);
    draw_border(image, rect, [72, 76, 96]);
    if rect.width() == 0 || rect.height() == 0 || layer.syn_edges.is_empty() {
        return;
    }

    let max_value = layer
        .syn_edges
        .iter()
        .map(|edge| edge.value.abs())
        .fold(0.0_f32, f32::max)
        .max(f32::EPSILON);

    let min_i = layer.syn_edges.iter().map(|e| e.i).min().unwrap_or(0);
    let max_i = layer
        .syn_edges
        .iter()
        .map(|e| e.i)
        .max()
        .unwrap_or(min_i + 1);
    let min_j = layer.syn_edges.iter().map(|e| e.j).min().unwrap_or(0);
    let max_j = layer
        .syn_edges
        .iter()
        .map(|e| e.j)
        .max()
        .unwrap_or(min_j + 1);
    let span_i = max(1, (max_i - min_i) as usize);
    let span_j = max(1, (max_j - min_j) as usize);

    let strongest = layer.syn_edges.iter().max_by(|a, b| {
        a.value
            .abs()
            .partial_cmp(&b.value.abs())
            .unwrap_or(Ordering::Equal)
    });

    for edge in &layer.syn_edges {
        let norm_i = (edge.i.saturating_sub(min_i)) as f32 / span_i as f32;
        let norm_j = (edge.j.saturating_sub(min_j)) as f32 / span_j as f32;
        let x = rect.x0
            + min(
                rect.width().saturating_sub(1),
                (norm_j.clamp(0.0, 1.0) * rect.width() as f32) as usize,
            );
        let y = rect.y1.saturating_sub(1).saturating_sub(min(
            rect.height().saturating_sub(1),
            (norm_i.clamp(0.0, 1.0) * rect.height() as f32) as usize,
        ));
        let color = synapse_color(edge.value, max_value);
        put_pixel(image, x, y, color);
        if x + 1 < rect.x1 {
            put_pixel(image, x + 1, y, [color[0] / 2, color[1] / 2, color[2] / 2]);
        }
        if y + 1 < rect.y1 {
            put_pixel(image, x, y + 1, [color[0] / 2, color[1] / 2, color[2] / 2]);
        }
    }

    if let Some(edge) = strongest {
        draw_text_line(
            image,
            rect.x0 + 4,
            rect.y0 + 4,
            &format!("peak {}→{} {:.2}", edge.i, edge.j, edge.value),
            [220, 210, 220],
        );
    }
}

fn attention_color(norm: f32) -> [u8; 3] {
    let n = norm.clamp(0.0, 1.0);
    [
        (n * 255.0) as u8,
        (n * 160.0 + 20.0) as u8,
        (n * 90.0 + 40.0) as u8,
    ]
}

fn neuron_color(norm: f32) -> [u8; 3] {
    let n = norm.clamp(0.0, 1.0);
    [
        (n * 120.0 + 40.0) as u8,
        (n * 220.0 + 50.0) as u8,
        (n * 190.0 + 60.0) as u8,
    ]
}

fn synapse_color(value: f32, max_value: f32) -> [u8; 3] {
    let norm = (value / max_value).clamp(-1.0, 1.0);
    if norm >= 0.0 {
        [(norm * 255.0) as u8, (norm * 120.0 + 30.0) as u8, 60]
    } else {
        let pos = (-norm).clamp(0.0, 1.0);
        [70, (pos * 120.0 + 40.0) as u8, (pos * 255.0) as u8]
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
