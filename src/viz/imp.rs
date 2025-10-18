use std::path::PathBuf;

mod video;

use super::frame;
use frame::TokenFrame;

pub enum VizMode {
    Off,
    Video(VideoConfig),
}

pub struct Viz {
    inner: Option<video::VideoHandle>,
}

impl Viz {
    pub fn new(mode: VizMode) -> Self {
        let inner = match mode {
            VizMode::Off => None,
            VizMode::Video(config) => match video::VideoHandle::start(config) {
                Ok(handle) => Some(handle),
                Err(err) => {
                    eprintln!("viz: failed to initialize video encoder: {err:#}");
                    None
                }
            },
        };

        Self { inner }
    }

    pub fn enabled(&self) -> bool {
        self.inner.is_some()
    }

    pub fn push(&self, frame: TokenFrame) {
        if let Some(handle) = &self.inner {
            handle.push(frame);
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
            width: 1280,
            height: 720,
            fps: 24,
            realtime: false,
            crf: Some(18),
            bitrate_kbps: None,
            keyframe_interval: Some(48),
        }
    }
}
