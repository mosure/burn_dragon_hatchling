use std::sync::Mutex;

use bevy::asset::RenderAssetUsages;
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy_pancam::{PanCam, PanCamPlugin};
use bevy_burn::{
    BevyBurnBridgePlugin, BevyBurnHandle, BindingDirection, BurnDevice, TransferKind,
};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_wgpu::WgpuDevice;

use super::frame::VizConfig;
use super::transport::VizReceiver;

#[derive(Clone, Copy, Debug)]
pub struct VizDimensions {
    pub layers: usize,
    pub heads: usize,
    pub latent_per_head: usize,
}

#[derive(Resource, Clone, Copy, Debug)]
struct VizLayout {
    layers: usize,
    history: usize,
    latent_total: usize,
}

#[derive(Resource)]
struct ExitReceiver {
    inner: Mutex<std::sync::mpsc::Receiver<()>>,
}

#[derive(Component, Clone, Copy)]
enum PanelKind {
    OverviewActivity,
    OverviewWrites,
    UnitsActivity,
    UnitsWrites,
}

pub fn build_app<B: Backend<Device = WgpuDevice>>(
    config: VizConfig,
    dims: VizDimensions,
    receiver: VizReceiver<B>,
    exit_rx: Option<std::sync::mpsc::Receiver<()>>,
) -> (App, B::Device)
where
    B: Backend + 'static,
    B::Device: Default + Clone,
    (): bevy_burn::gpu_burn_to_bevy::BurnBevyPrepare<B>,
{
    let mut app = App::new();

    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "burn_dragon_hatchling".to_string(),
            canvas: Some("#bevy".to_string()),
            prevent_default_event_handling: true,
            ..default()
        }),
        ..default()
    }));
    app.add_plugins(PanCamPlugin);
    app.insert_resource(ClearColor(Color::BLACK));
    app.add_plugins(BevyBurnBridgePlugin::<B>::default());

    let history = config.history.max(1);
    let layers = dims.layers.max(1);
    let latent_total = dims
        .heads
        .saturating_mul(dims.latent_per_head)
        .max(1);
    app.insert_resource(VizLayout {
        layers,
        history,
        latent_total,
    });

    insert_receiver::<B>(&mut app, receiver);

    if let Some(exit_rx) = exit_rx {
        app.insert_resource(ExitReceiver {
            inner: Mutex::new(exit_rx),
        });
        app.add_systems(Update, poll_exit);
    }

    app.add_systems(Startup, setup::<B>);
    app.add_systems(Update, apply_latest_frame::<B>);

    app.finish();
    app.cleanup();

    let device = app
        .world()
        .resource::<BurnDevice>()
        .device()
        .cloned()
        .expect("viz: burn device not ready; bevy_burn must initialize the shared GPU device");

    (app, device)
}

#[cfg(not(target_arch = "wasm32"))]
fn insert_receiver<B: Backend>(app: &mut App, receiver: VizReceiver<B>) {
    app.insert_resource(receiver);
}

#[cfg(target_arch = "wasm32")]
fn insert_receiver<B: Backend>(app: &mut App, receiver: VizReceiver<B>) {
    app.insert_non_send_resource(receiver);
}

fn poll_exit(receiver: Res<ExitReceiver>, mut exit: MessageWriter<AppExit>) {
    if let Ok(rx) = receiver.inner.lock() {
        if rx.try_recv().is_ok() {
            exit.write(AppExit::Success);
        }
    }
}

fn setup<B: Backend<Device = WgpuDevice>>(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    layout: Res<VizLayout>,
    burn: Res<BurnDevice>,
) {
    let Some(device) = burn.device() else {
        eprintln!("viz: burn device not ready during setup");
        return;
    };

    commands.spawn((Camera2d, PanCam::default()));

    commands
        .spawn((
            Node {
                display: Display::Grid,
                grid_template_columns: vec![GridTrack::flex(1.0), GridTrack::flex(1.0)],
                grid_template_rows: vec![GridTrack::flex(1.0), GridTrack::flex(1.0)],
                width: percent(100.0),
                height: percent(100.0),
                row_gap: px(12.0),
                column_gap: px(12.0),
                padding: UiRect::all(px(12.0)),
                ..default()
            },
            BackgroundColor(Color::BLACK),
        ))
        .with_children(|root| {
            spawn_panel::<B>(
                root,
                "overview_activity",
                PanelKind::OverviewActivity,
                layout.history,
                layout.layers,
                device,
                &mut images,
            );
            spawn_panel::<B>(
                root,
                "overview_writes",
                PanelKind::OverviewWrites,
                layout.history,
                layout.layers,
                device,
                &mut images,
            );
            spawn_panel::<B>(
                root,
                "units_activity",
                PanelKind::UnitsActivity,
                layout.history,
                layout.latent_total,
                device,
                &mut images,
            );
            spawn_panel::<B>(
                root,
                "units_writes",
                PanelKind::UnitsWrites,
                layout.history,
                layout.latent_total,
                device,
                &mut images,
            );
        });
}

fn spawn_panel<B: Backend<Device = WgpuDevice>>(
    parent: &mut ChildSpawnerCommands,
    label: &str,
    kind: PanelKind,
    width: usize,
    height: usize,
    device: &B::Device,
    images: &mut Assets<Image>,
) {
    let (handle, tensor) = build_image::<B>(width, height, device, images);

    parent
        .spawn((
            Node {
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                width: percent(100.0),
                height: percent(100.0),
                padding: UiRect::all(px(8.0)),
                row_gap: px(6.0),
                ..default()
            },
            BackgroundColor(Color::srgb(0.02, 0.02, 0.02)),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new(label),
                TextFont {
                    font: Handle::default(),
                    font_size: 12.0,
                    ..default()
                },
                TextColor(Color::srgb(0.8, 0.8, 0.8)),
            ));
            panel.spawn((
                Node {
                    width: percent(100.0),
                    height: percent(100.0),
                    flex_grow: 1.0,
                    ..default()
                },
                ImageNode::new(handle.clone()),
                BevyBurnHandle::<B> {
                    bevy_image: handle,
                    tensor,
                    upload: true,
                    direction: BindingDirection::BurnToBevy,
                    xfer: TransferKind::Gpu,
                },
                kind,
            ));
        });
}

fn build_image<B: Backend>(
    width: usize,
    height: usize,
    device: &B::Device,
    images: &mut Assets<Image>,
) -> (Handle<Image>, Tensor<B, 3>) {
    let width = width.max(1) as u32;
    let height = height.max(1) as u32;
    let size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let mut img = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0; 16],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.texture_descriptor.usage |= TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::STORAGE_BINDING;
    img.sampler = ImageSampler::nearest();

    let handle = images.add(img);
    let tensor = Tensor::<B, 3>::zeros([height as usize, width as usize, 4], device);
    (handle, tensor)
}

#[cfg(not(target_arch = "wasm32"))]
fn apply_latest_frame<B: Backend>(
    receiver: Res<VizReceiver<B>>,
    mut q: Query<(&PanelKind, &mut BevyBurnHandle<B>)>,
) {
    let Some(frame) = receiver.drain_latest() else {
        return;
    };
    for (kind, mut handle) in &mut q {
        match kind {
            PanelKind::OverviewActivity => handle.tensor = frame.overview_activity.clone(),
            PanelKind::OverviewWrites => handle.tensor = frame.overview_writes.clone(),
            PanelKind::UnitsActivity => handle.tensor = frame.units_activity.clone(),
            PanelKind::UnitsWrites => handle.tensor = frame.units_writes.clone(),
        }
        handle.upload = true;
    }
}

#[cfg(target_arch = "wasm32")]
fn apply_latest_frame<B: Backend>(
    receiver: NonSend<VizReceiver<B>>,
    mut q: Query<(&PanelKind, &mut BevyBurnHandle<B>)>,
) {
    let Some(frame) = receiver.drain_latest() else {
        return;
    };
    for (kind, mut handle) in &mut q {
        match kind {
            PanelKind::OverviewActivity => handle.tensor = frame.overview_activity.clone(),
            PanelKind::OverviewWrites => handle.tensor = frame.overview_writes.clone(),
            PanelKind::UnitsActivity => handle.tensor = frame.units_activity.clone(),
            PanelKind::UnitsWrites => handle.tensor = frame.units_writes.clone(),
        }
        handle.upload = true;
    }
}

#[cfg(all(test, feature = "viz", feature = "cli"))]
mod tests {
    use super::*;
    use bevy::asset::Assets;
    use burn::tensor::Tensor;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use super::super::frame::VizFrame;

    type Backend = NdArray<f32>;

    fn device() -> NdArrayDevice {
        NdArrayDevice::default()
    }

    #[test]
    fn build_image_uses_rgba32_float() {
        let device = device();
        let mut images = Assets::<Image>::default();
        let (handle, tensor) = build_image::<Backend>(4, 3, &device, &mut images);
        let image = images.get(&handle).expect("image exists");

        assert_eq!(image.texture_descriptor.format, TextureFormat::Rgba32Float);
        let usage = image.texture_descriptor.usage;
        assert!(usage.contains(TextureUsages::COPY_SRC));
        assert!(usage.contains(TextureUsages::COPY_DST));
        assert!(usage.contains(TextureUsages::TEXTURE_BINDING));
        assert!(usage.contains(TextureUsages::STORAGE_BINDING));

        assert_eq!(tensor.shape().dims::<3>(), [3, 4, 4]);
    }

    #[test]
    fn apply_latest_frame_updates_handles() {
        let device = device();
        let (sender, receiver) = super::super::transport::channel::<Backend>();

        let mut app = App::new();
        app.insert_resource(receiver);
        app.add_systems(Update, apply_latest_frame::<Backend>);

        let make_handle = |height, width| BevyBurnHandle::<Backend> {
            bevy_image: Handle::default(),
            tensor: Tensor::<Backend, 3>::zeros([height, width, 4], &device),
            upload: false,
            direction: BindingDirection::BurnToBevy,
            xfer: TransferKind::Cpu,
        };

        app.world_mut().spawn((
            PanelKind::OverviewActivity,
            make_handle(2, 3),
        ));
        app.world_mut().spawn((PanelKind::OverviewWrites, make_handle(2, 3)));
        app.world_mut().spawn((PanelKind::UnitsActivity, make_handle(4, 3)));
        app.world_mut().spawn((PanelKind::UnitsWrites, make_handle(4, 3)));

        let frame = VizFrame::<Backend> {
            overview_activity: Tensor::<Backend, 3>::zeros([2, 3, 4], &device),
            overview_writes: Tensor::<Backend, 3>::zeros([2, 3, 4], &device),
            units_activity: Tensor::<Backend, 3>::zeros([4, 3, 4], &device),
            units_writes: Tensor::<Backend, 3>::zeros([4, 3, 4], &device),
            cursor: 0,
            token_index: 0,
        };
        sender.try_send(frame);

        app.update();

        let mut query = app.world_mut().query::<(&PanelKind, &BevyBurnHandle<Backend>)>();
        let mut count = 0;
        for (kind, handle) in query.iter(app.world()) {
            count += 1;
            assert!(handle.upload);
            let dims = handle.tensor.shape().dims::<3>();
            match kind {
                PanelKind::OverviewActivity | PanelKind::OverviewWrites => {
                    assert_eq!(dims, [2, 3, 4]);
                }
                PanelKind::UnitsActivity | PanelKind::UnitsWrites => {
                    assert_eq!(dims, [4, 3, 4]);
                }
            }
        }
        assert_eq!(count, 4);
    }
}
