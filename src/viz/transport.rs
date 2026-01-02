use bevy::prelude::Resource;
use burn::tensor::backend::Backend;

use super::frame::VizFrame;

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use std::rc::Rc;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct VizSender<B: Backend> {
    #[cfg(not(target_arch = "wasm32"))]
    inner: Arc<Mutex<Option<VizFrame<B>>>>,
    #[cfg(target_arch = "wasm32")]
    inner: Rc<RefCell<Option<VizFrame<B>>>>,
}

#[derive(Resource)]
pub struct VizReceiver<B: Backend> {
    #[cfg(not(target_arch = "wasm32"))]
    inner: Arc<Mutex<Option<VizFrame<B>>>>,
    #[cfg(target_arch = "wasm32")]
    inner: Rc<RefCell<Option<VizFrame<B>>>>,
}

impl<B: Backend> VizSender<B> {
    pub fn try_send(&self, frame: VizFrame<B>) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Ok(mut slot) = self.inner.try_lock() {
                *slot = Some(frame);
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            *self.inner.borrow_mut() = Some(frame);
        }
    }
}

impl<B: Backend> VizReceiver<B> {
    pub fn drain_latest(&self) -> Option<VizFrame<B>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.lock().ok().and_then(|mut slot| slot.take())
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.inner.borrow_mut().take()
        }
    }
}

pub fn channel<B: Backend>() -> (VizSender<B>, VizReceiver<B>) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let slot = Arc::new(Mutex::new(None));
        (
            VizSender { inner: slot.clone() },
            VizReceiver { inner: slot },
        )
    }

    #[cfg(target_arch = "wasm32")]
    {
        let slot = Rc::new(RefCell::new(None));
        (
            VizSender { inner: slot.clone() },
            VizReceiver { inner: slot },
        )
    }
}

#[cfg(all(test, feature = "viz", feature = "cli"))]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};

    use super::super::{VizConfig, VizEncoder};

    type Backend = NdArray<f32>;

    fn device() -> NdArrayDevice {
        NdArrayDevice::default()
    }

    #[test]
    fn viz_channel_returns_latest_frame() {
        let device = device();
        let (sender, receiver) = channel::<Backend>();
        let config = VizConfig {
            history: 2,
            layer_focus: 0,
            stride_tokens: 1,
            gain_x: 1.0,
            gain_xy: 1.0,
        };
        let mut encoder = VizEncoder::<Backend>::new(config, 1, 1, 1, &device);

        sender.try_send(encoder.step(&[None], 0));
        sender.try_send(encoder.step(&[None], 1));

        let latest = receiver.drain_latest().expect("latest frame");
        assert_eq!(latest.token_index, 1);
    }
}
