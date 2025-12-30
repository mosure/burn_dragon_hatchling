use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, HalfPrecisionSettings, Recorder};
use burn::tensor::backend::Backend as BackendTrait;
use burn_ndarray::NdArray;
use clap::Parser;

use burn_dragon_hatchling::BDH;

type QuantBackend = NdArray<f32>;

#[derive(Parser, Debug)]
#[command(author, version, about = "Quantize BDH checkpoints for web deployment")]
struct Args {
    /// Path to a checkpoint directory or model file.
    #[arg(long, value_name = "PATH")]
    checkpoint: PathBuf,
    /// Specific checkpoint epoch to use when the path is a directory.
    #[arg(long, value_name = "N")]
    epoch: Option<usize>,
    /// Output path for the quantized checkpoint (defaults alongside the input).
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    let (checkpoint_base, epoch) = resolve_checkpoint_base(&args.checkpoint, args.epoch)?;
    let output_base = resolve_output_base(args.output.as_ref(), &checkpoint_base)?;

    let device = <QuantBackend as BackendTrait>::Device::default();
    QuantBackend::seed(&device, 1337);

    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<<BDH<QuantBackend> as Module<QuantBackend>>::Record>(
            checkpoint_base.clone(),
            &device,
        )
        .with_context(|| {
            format!(
                "failed to load checkpoint {}",
                format_checkpoint(&checkpoint_base)
            )
        })?;

    BinFileRecorder::<HalfPrecisionSettings>::new()
        .record(record, output_base.clone())
        .with_context(|| {
            format!(
                "failed to write quantized checkpoint {}",
                format_checkpoint(&output_base)
            )
        })?;

    eprintln!(
        "Quantized epoch {epoch} -> {}",
        format_checkpoint(&output_base)
    );

    Ok(())
}

fn resolve_output_base(output: Option<&PathBuf>, checkpoint_base: &Path) -> Result<PathBuf> {
    if let Some(path) = output {
        let base = base_without_extension(path);
        return Ok(base);
    }

    let stem = checkpoint_base
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("unable to determine checkpoint file name"))?;
    let mut output_base = checkpoint_base.to_path_buf();
    output_base.set_file_name(format!("{stem}-f16"));
    Ok(output_base)
}

fn resolve_checkpoint_base(path: &Path, epoch: Option<usize>) -> Result<(PathBuf, usize)> {
    if path.is_dir() {
        let target_epoch = epoch.unwrap_or(find_latest_epoch(path)?);
        let base = path.join(format!("model-{target_epoch}"));
        ensure_checkpoint_exists(&base)?;
        return Ok((base, target_epoch));
    }

    let mut base = base_without_extension(path);
    let detected_epoch = parse_epoch_from_stem(&base);
    let target_epoch = match (epoch, detected_epoch) {
        (Some(explicit), Some(detected)) if explicit != detected => {
            let parent = base.parent().map(Path::to_path_buf).unwrap_or_default();
            base = parent.join(format!("model-{explicit}"));
            explicit
        }
        (Some(explicit), _) => {
            if detected_epoch.is_none() {
                let parent = base
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("runs"));
                base = parent.join(format!("model-{explicit}"));
            }
            explicit
        }
        (None, Some(detected)) => detected,
        (None, None) => {
            return Err(anyhow!(
                "unable to infer checkpoint epoch from {}; provide --epoch",
                path.display()
            ));
        }
    };

    ensure_checkpoint_exists(&base)?;
    Ok((base, target_epoch))
}

fn base_without_extension(path: &Path) -> PathBuf {
    let mut base = path.to_path_buf();
    if base.extension().is_some() {
        base.set_extension("");
    }
    base
}

fn ensure_checkpoint_exists(base: &Path) -> Result<()> {
    let mut candidate = base.to_path_buf();
    candidate.set_extension("bin");
    if candidate.is_file() {
        return Ok(());
    }

    Err(anyhow!("checkpoint file {}.bin not found", base.display()))
}

fn find_latest_epoch(dir: &Path) -> Result<usize> {
    let mut max_epoch = None;
    for entry in fs::read_dir(dir)
        .with_context(|| format!("failed to read checkpoint directory {}", dir.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let mut base = entry.path();
        base.set_extension("");
        if let Some(epoch) = parse_epoch_from_stem(&base) {
            let updated = max_epoch
                .map(|current: usize| current.max(epoch))
                .unwrap_or(epoch);
            max_epoch = Some(updated);
        }
    }

    max_epoch.ok_or_else(|| anyhow!("no model checkpoints found in {}", dir.display()))
}

fn parse_epoch_from_stem(path: &Path) -> Option<usize> {
    let stem = path.file_name()?.to_string_lossy();
    let stem = stem.strip_suffix(".bin").unwrap_or(&stem);
    let epoch_part = stem.strip_prefix("model-")?;
    epoch_part.parse().ok()
}

fn format_checkpoint(base: &Path) -> String {
    let mut path = base.to_path_buf();
    path.set_extension("bin");
    path.display().to_string()
}
