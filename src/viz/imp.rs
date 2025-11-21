use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;

use super::activity::{ActivityComputer, FrameActivity};
use super::frame::TokenFrame;

mod video;

pub enum VizMode {
    Off,
    Video(VideoConfig),
    Targets(Vec<VizTargetConfig>),
}

pub enum VizTargetConfig {
    Video(VideoConfig),
}

pub struct Viz {
    activity: Mutex<ActivityComputer>,
    targets: Vec<Arc<dyn VizTarget>>,
}

impl Viz {
    pub fn new(mode: VizMode) -> Self {
        let mut viz = Self {
            activity: Mutex::new(ActivityComputer::new()),
            targets: Vec::new(),
        };

        match mode {
            VizMode::Off => {}
            VizMode::Video(config) => viz.attach_target(VizTargetConfig::Video(config)),
            VizMode::Targets(configs) => {
                for config in configs {
                    viz.attach_target(config);
                }
            }
        }

        viz
    }

    pub fn enabled(&self) -> bool {
        !self.targets.is_empty()
    }

    pub fn push(&self, frame: TokenFrame) {
        if self.targets.is_empty() {
            return;
        }

        let activity = {
            let mut guard = self.activity.lock();
            guard.compute(&frame)
        };
        let snapshot = FrameSnapshot::from_parts(frame, activity);

        for target in &self.targets {
            target.push(snapshot.clone());
        }
    }

    fn attach_target(&mut self, config: VizTargetConfig) {
        match config {
            VizTargetConfig::Video(config) => match video::VideoHandle::start(config) {
                Ok(handle) => self.targets.push(handle),
                Err(err) => eprintln!("viz: failed to initialize video encoder: {err:#}"),
            },
        }
    }
}

impl Drop for Viz {
    fn drop(&mut self) {
        for target in &self.targets {
            target.finalize();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_mode_is_disabled() {
        let viz = Viz::new(VizMode::Off);
        assert!(!viz.enabled());
    }

    #[test]
    fn pipeline_mode_enables_when_target_created() {
        let viz = Viz::new(VizMode::Targets(Vec::new()));
        assert!(!viz.enabled());
    }
}

#[derive(Clone, Debug)]
pub struct VideoConfig {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub realtime: bool,
    pub crf: Option<u8>,
    pub bitrate_kbps: Option<u32>,
    pub keyframe_interval: Option<u32>,
    pub layout: VideoLayout,
}

impl VideoConfig {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            ..Default::default()
        }
    }
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("viz.mp4"),
            width: 1920,
            height: 1080,
            fps: 6,
            realtime: false,
            crf: Some(18),
            bitrate_kbps: None,
            keyframe_interval: Some(48),
            layout: VideoLayout::Timeline,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub enum VideoLayout {
    #[default]
    Timeline,
    Plasticity,
    ScaleFree,
}

#[derive(Clone)]
pub(super) struct FrameSnapshot {
    frame: Arc<TokenFrame>,
    activity: Arc<FrameActivity>,
}

impl FrameSnapshot {
    fn from_parts(frame: TokenFrame, activity: FrameActivity) -> Self {
        Self {
            frame: Arc::new(frame),
            activity: Arc::new(activity),
        }
    }

    pub(super) fn frame(&self) -> &TokenFrame {
        &self.frame
    }

    pub(super) fn activity(&self) -> &FrameActivity {
        &self.activity
    }
}

pub(super) trait VizTarget: Send + Sync {
    fn push(&self, snapshot: FrameSnapshot);
    fn finalize(&self);
}
