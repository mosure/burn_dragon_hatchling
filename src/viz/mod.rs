pub mod bevy_app;
pub mod encoder;
pub mod frame;
pub mod palette;
pub mod transport;

use bevy::prelude::App;
use burn::tensor::backend::Backend;

pub use bevy_app::VizDimensions;
pub use encoder::VizEncoder;
pub use frame::{VizConfig, VizFrame};
pub use transport::{VizReceiver, VizSender};

#[derive(Clone)]
pub struct VizHandle<B: Backend> {
    sender: VizSender<B>,
    device: B::Device,
}

impl<B: Backend> VizHandle<B> {
    pub fn submit(&self, frame: VizFrame<B>) {
        self.sender.try_send(frame);
    }

    pub fn sender(&self) -> VizSender<B> {
        self.sender.clone()
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }
}

pub struct VizOverlay<B: Backend> {
    app: App,
    handle: VizHandle<B>,
}

impl<B: Backend> VizOverlay<B> {
    pub fn run(mut self) {
        self.app.run();
    }

    pub fn handle(&self) -> &VizHandle<B> {
        &self.handle
    }

    pub fn split(self) -> (VizHandle<B>, App) {
        (self.handle, self.app)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn start_overlay_native<B: Backend<Device = burn_wgpu::WgpuDevice>>(
    config: VizConfig,
    dims: VizDimensions,
    exit_rx: Option<std::sync::mpsc::Receiver<()>>,
) -> VizOverlay<B>
where
    B: Backend + 'static,
    B::Device: Default + Clone,
    (): bevy_burn::gpu_burn_to_bevy::BurnBevyPrepare<B>,
{
    let (sender, receiver) = transport::channel();
    let (app, device) = bevy_app::build_app::<B>(config, dims, receiver, exit_rx);
    VizOverlay {
        app,
        handle: VizHandle { sender, device },
    }
}

#[cfg(target_arch = "wasm32")]
pub fn start_overlay_wasm<B: Backend<Device = burn_wgpu::WgpuDevice>>(
    config: VizConfig,
    dims: VizDimensions,
) -> VizOverlay<B>
where
    B: Backend + 'static,
    B::Device: Default + Clone,
    (): bevy_burn::gpu_burn_to_bevy::BurnBevyPrepare<B>,
{
    let (sender, receiver) = transport::channel();
    let (app, device) = bevy_app::build_app::<B>(config, dims, receiver, None);
    VizOverlay {
        app,
        handle: VizHandle { sender, device },
    }
}
