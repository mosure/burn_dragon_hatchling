use std::sync::Mutex;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{
    Arc,
    atomic::AtomicBool,
};
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
use std::sync::atomic::Ordering;

use bevy::asset::RenderAssetUsages;
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::image::{ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::input::ButtonInput;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::ui::{ComputedNode, UiGlobalTransform};
use bevy::window::{PrimaryWindow, Window};
#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
use bevy::window::WindowCloseRequested;
use bevy_burn::{
    BevyBurnBridgePlugin, BevyBurnHandle, BindingDirection, BurnDevice, TransferKind,
};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_wgpu::WgpuDevice;

use super::frame::{LAYER_GAP, VizConfig, VizFrame};
use super::transport::VizReceiver;

#[derive(Clone, Copy, Debug)]
pub struct VizDimensions {
    pub layers: usize,
    pub heads: usize,
    pub latent_per_head: usize,
}

#[derive(Resource, Clone, Copy, Debug)]
struct VizLayout {
    history: usize,
    units_height: usize,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
#[derive(Resource, Clone)]
struct StopSignal {
    flag: Arc<AtomicBool>,
}

#[derive(Resource)]
struct ExitReceiver {
    inner: Mutex<std::sync::mpsc::Receiver<()>>,
}

#[derive(Resource, Debug)]
struct PanZoomState {
    scale: f32,
    min_scale: f32,
    max_scale: f32,
    offset: Vec2,
    viewport_size: Vec2,
    inverse_scale_factor: f32,
    initialized: bool,
    dragging: bool,
    last_cursor: Option<Vec2>,
}

impl Default for PanZoomState {
    fn default() -> Self {
        Self {
            scale: 1.0,
            min_scale: 1.0,
            max_scale: 1.0,
            offset: Vec2::ZERO,
            viewport_size: Vec2::ZERO,
            inverse_scale_factor: 1.0,
            initialized: false,
            dragging: false,
            last_cursor: None,
        }
    }
}

#[derive(Resource, Clone, Copy, Debug)]
struct PanZoomTexture {
    size: Vec2,
}

#[derive(Component)]
struct PanZoomViewport;

#[derive(Component)]
struct PanZoomImage;

#[derive(Component, Clone, Copy)]
enum PanelKind {
    UnitsWrites,
    UnitsY,
    UnitsXY,
    UnitsRho,
}

impl PanelKind {
    fn label(&self) -> &'static str {
        match self {
            PanelKind::UnitsWrites => "writes",
            PanelKind::UnitsY => "y",
            PanelKind::UnitsXY => "xy",
            PanelKind::UnitsRho => "rho",
        }
    }

    fn select_tensor<'a, B: Backend>(&self, frame: &'a VizFrame<B>) -> &'a Tensor<B, 3> {
        match self {
            PanelKind::UnitsWrites => &frame.units_x,
            PanelKind::UnitsY => &frame.units_y,
            PanelKind::UnitsXY => &frame.units_xy,
            PanelKind::UnitsRho => &frame.units_rho,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn build_app<B: Backend<Device = WgpuDevice>>(
    config: VizConfig,
    dims: VizDimensions,
    receiver: VizReceiver<B>,
    exit_rx: Option<std::sync::mpsc::Receiver<()>>,
    stop_flag: Arc<AtomicBool>,
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
    app.insert_resource(ClearColor(Color::BLACK));
    app.add_plugins(BevyBurnBridgePlugin::<B>::default());

    let history = config.history.max(1);
    let latent_total = dims
        .heads
        .saturating_mul(dims.latent_per_head)
        .max(1);
    let units_height = latent_total
        .saturating_mul(dims.layers.max(1))
        .saturating_add(LAYER_GAP.saturating_mul(dims.layers.saturating_sub(1)))
        .max(1);
    app.insert_resource(VizLayout {
        history,
        units_height,
    });
    #[cfg(feature = "cli")]
    app.insert_resource(StopSignal { flag: stop_flag.clone() });
    #[cfg(not(feature = "cli"))]
    let _ = stop_flag;
    app.insert_resource(PanZoomState::default());
    app.insert_resource(PanZoomTexture {
        size: Vec2::new(history as f32, units_height as f32),
    });

    insert_receiver::<B>(&mut app, receiver);

    if let Some(exit_rx) = exit_rx {
        app.insert_resource(ExitReceiver {
            inner: Mutex::new(exit_rx),
        });
        app.add_systems(Update, poll_exit);
    }

    app.add_systems(Startup, setup::<B>);
    #[cfg(feature = "cli")]
    app.add_systems(Update, handle_window_close);
    app.add_systems(
        Update,
        (update_pan_zoom_bounds, pan_zoom_input, apply_pan_zoom).chain(),
    );
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

#[cfg(target_arch = "wasm32")]
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
    app.insert_resource(ClearColor(Color::BLACK));
    app.add_plugins(BevyBurnBridgePlugin::<B>::default());

    let history = config.history.max(1);
    let latent_total = dims
        .heads
        .saturating_mul(dims.latent_per_head)
        .max(1);
    let units_height = latent_total
        .saturating_mul(dims.layers.max(1))
        .saturating_add(LAYER_GAP.saturating_mul(dims.layers.saturating_sub(1)))
        .max(1);
    app.insert_resource(VizLayout {
        history,
        units_height,
    });
    app.insert_resource(PanZoomState::default());
    app.insert_resource(PanZoomTexture {
        size: Vec2::new(history as f32, units_height as f32),
    });

    insert_receiver::<B>(&mut app, receiver);

    if let Some(exit_rx) = exit_rx {
        app.insert_resource(ExitReceiver {
            inner: Mutex::new(exit_rx),
        });
        app.add_systems(Update, poll_exit);
    }

    app.add_systems(Startup, setup::<B>);
    app.add_systems(
        Update,
        (update_pan_zoom_bounds, pan_zoom_input, apply_pan_zoom).chain(),
    );
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

#[cfg(all(not(target_arch = "wasm32"), feature = "cli"))]
fn handle_window_close(
    mut events: MessageReader<WindowCloseRequested>,
    stop: Res<StopSignal>,
    mut exit: MessageWriter<AppExit>,
) {
    if events.read().next().is_some() {
        stop.flag.store(true, Ordering::Relaxed);
        exit.write(AppExit::Success);
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

    commands.spawn(Camera2d);

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
                PanelKind::UnitsWrites,
                layout.history,
                layout.units_height,
                device,
                &mut images,
            );
            spawn_panel::<B>(
                root,
                PanelKind::UnitsY,
                layout.history,
                layout.units_height,
                device,
                &mut images,
            );
            spawn_panel::<B>(
                root,
                PanelKind::UnitsXY,
                layout.history,
                layout.units_height,
                device,
                &mut images,
            );
            spawn_panel::<B>(
                root,
                PanelKind::UnitsRho,
                layout.history,
                layout.units_height,
                device,
                &mut images,
            );
        });
}

fn spawn_panel<B: Backend<Device = WgpuDevice>>(
    parent: &mut ChildSpawnerCommands,
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
                Text::new(kind.label()),
                TextFont {
                    font: Handle::default(),
                    font_size: 12.0,
                    ..default()
                },
                TextColor(Color::srgb(0.8, 0.8, 0.8)),
            ));
            panel
                .spawn((
                    Node {
                        width: percent(100.0),
                        flex_grow: 1.0,
                        position_type: PositionType::Relative,
                        overflow: Overflow::clip(),
                        ..default()
                    },
                    PanZoomViewport,
                ))
                .with_children(|viewport| {
                    viewport.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(0.0),
                            top: px(0.0),
                            width: px(width as f32),
                            height: px(height as f32),
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
                        PanZoomImage,
                    ));
                });
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
    img.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        mag_filter: ImageFilterMode::Nearest,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Nearest,
        ..Default::default()
    });

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
        handle.tensor = kind.select_tensor(&frame).clone();
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
        handle.tensor = kind.select_tensor(&frame).clone();
        handle.upload = true;
    }
}

fn update_pan_zoom_bounds(
    mut state: ResMut<PanZoomState>,
    texture: Res<PanZoomTexture>,
    viewports: Query<&ComputedNode, With<PanZoomViewport>>,
) {
    let Some(node) = viewports.iter().find(|node| !node.is_empty()) else {
        return;
    };
    let viewport = node.size();
    let inverse_scale_factor = node.inverse_scale_factor();
    if viewport.x <= 0.0 || viewport.y <= 0.0 {
        return;
    }

    let scale_x = viewport.x / texture.size.x.max(1.0);
    let scale_y = viewport.y / texture.size.y.max(1.0);
    let min_scale = scale_x.min(scale_y).max(0.0001);
    let max_scale = (min_scale * 64.0).max(min_scale);

    let viewport_changed = state.viewport_size != viewport
        || (state.inverse_scale_factor - inverse_scale_factor).abs() > f32::EPSILON;

    state.viewport_size = viewport;
    state.inverse_scale_factor = inverse_scale_factor;
    state.min_scale = min_scale;
    state.max_scale = max_scale;

    if !state.initialized {
        let init_scale = scale_x.clamp(min_scale, max_scale);
        state.scale = init_scale;
        state.offset = default_offset_top(viewport, texture.size * state.scale);
        state.initialized = true;
        return;
    }

    if state.scale < min_scale || viewport_changed {
        state.scale = state.scale.max(min_scale).min(state.max_scale);
        state.offset = clamp_offset(state.offset, viewport, texture.size, state.scale);
    }
}

fn pan_zoom_input(
    mut state: ResMut<PanZoomState>,
    texture: Res<PanZoomTexture>,
    windows: Query<&Window, With<PrimaryWindow>>,
    viewports: Query<(&ComputedNode, &UiGlobalTransform), With<PanZoomViewport>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut scroll_events: MessageReader<MouseWheel>,
) {
    if !state.initialized {
        return;
    }

    let window: &Window = match windows.single() {
        Ok(window) => window,
        Err(_) => return,
    };
    let Some(cursor) = window.physical_cursor_position() else {
        state.last_cursor = None;
        state.dragging = false;
        return;
    };

    let active = active_viewport(cursor, &viewports);
    if let Some(active) = active {
        state.viewport_size = active.size;
        state.inverse_scale_factor = active.inverse_scale_factor;
    }
    let cursor_local = active.map(|active| active.cursor_local);
    let cursor_in_viewport = cursor_local.is_some();
    let viewport_size = active.map(|active| active.size).unwrap_or(state.viewport_size);

    let mut scroll = 0.0f32;
    for event in scroll_events.read() {
        let delta = match event.unit {
            MouseScrollUnit::Line => event.y,
            MouseScrollUnit::Pixel => event.y / 100.0,
        };
        scroll += delta;
    }

    if scroll.abs() > f32::EPSILON && cursor_in_viewport {
        let zoom_factor = 1.1_f32.powf(scroll);
        let next_scale = (state.scale * zoom_factor).clamp(state.min_scale, state.max_scale);
        if (next_scale - state.scale).abs() > f32::EPSILON {
            let pivot = cursor_local.expect("cursor in viewport");
            let image_pos = (pivot - state.offset) / state.scale;
            state.scale = next_scale;
            state.offset = pivot - image_pos * state.scale;
            state.offset = clamp_offset(state.offset, viewport_size, texture.size, state.scale);
        }
    }

    let left_pressed = buttons.pressed(MouseButton::Left);
    if !left_pressed {
        state.dragging = false;
    } else if !state.dragging && cursor_in_viewport {
        state.dragging = true;
    }

    if state.dragging {
        if let Some(last) = state.last_cursor {
            let delta = cursor - last;
            state.offset += delta;
            state.offset = clamp_offset(state.offset, viewport_size, texture.size, state.scale);
        }
    }

    state.last_cursor = Some(cursor);
}

fn apply_pan_zoom(
    state: Res<PanZoomState>,
    texture: Res<PanZoomTexture>,
    mut images: Query<&mut Node, With<PanZoomImage>>,
) {
    if !state.initialized {
        return;
    }
    let scale_factor = state.inverse_scale_factor;
    if scale_factor <= 0.0 {
        return;
    }
    let scaled = texture.size * state.scale;
    let scaled_logical = scaled * scale_factor;
    let offset_logical = state.offset * scale_factor;
    for mut node in &mut images {
        node.width = px(scaled_logical.x);
        node.height = px(scaled_logical.y);
        node.left = px(offset_logical.x);
        node.top = px(offset_logical.y);
    }
}

#[derive(Clone, Copy, Debug)]
struct ActiveViewport {
    cursor_local: Vec2,
    size: Vec2,
    inverse_scale_factor: f32,
}

fn active_viewport(
    cursor: Vec2,
    viewports: &Query<(&ComputedNode, &UiGlobalTransform), With<PanZoomViewport>>,
) -> Option<ActiveViewport> {
    for (node, transform) in viewports.iter() {
        if node.contains_point(*transform, cursor) {
            let Some(local) = transform.try_inverse().map(|affine| affine.transform_point2(cursor))
            else {
                continue;
            };
            let size = node.size();
            let local_top_left = local + size * 0.5;
            return Some(ActiveViewport {
                cursor_local: local_top_left,
                size,
                inverse_scale_factor: node.inverse_scale_factor(),
            });
        }
    }
    None
}

fn clamp_offset(offset: Vec2, viewport: Vec2, texture: Vec2, scale: f32) -> Vec2 {
    let scaled = texture * scale;
    let mut out = offset;
    if scaled.x <= viewport.x {
        out.x = (viewport.x - scaled.x) * 0.5;
    } else {
        let min_x = viewport.x - scaled.x;
        out.x = out.x.clamp(min_x, 0.0);
    }
    if scaled.y <= viewport.y {
        out.y = (viewport.y - scaled.y) * 0.5;
    } else {
        let min_y = viewport.y - scaled.y;
        out.y = out.y.clamp(min_y, 0.0);
    }
    out
}

fn default_offset_top(viewport: Vec2, scaled: Vec2) -> Vec2 {
    Vec2::new((viewport.x - scaled.x) * 0.5, 0.0)
}

#[cfg(all(test, feature = "viz", feature = "cli"))]
mod tests {
    use super::*;
    use bevy::asset::Assets;
    use burn::tensor::Tensor;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use super::super::frame::{LAYER_GAP, VizFrame};

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

        let units_height = 2 * 4 + LAYER_GAP;
        app.world_mut()
            .spawn((PanelKind::UnitsWrites, make_handle(units_height, 3)));
        app.world_mut()
            .spawn((PanelKind::UnitsY, make_handle(units_height, 3)));
        app.world_mut()
            .spawn((PanelKind::UnitsXY, make_handle(units_height, 3)));
        app.world_mut()
            .spawn((PanelKind::UnitsRho, make_handle(units_height, 3)));

        let frame = VizFrame::<Backend> {
            units_x: Tensor::<Backend, 3>::zeros([units_height, 3, 4], &device),
            units_y: Tensor::<Backend, 3>::zeros([units_height, 3, 4], &device),
            units_xy: Tensor::<Backend, 3>::zeros([units_height, 3, 4], &device),
            units_rho: Tensor::<Backend, 3>::zeros([units_height, 3, 4], &device),
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
                PanelKind::UnitsWrites
                | PanelKind::UnitsY
                | PanelKind::UnitsXY
                | PanelKind::UnitsRho => {
                    assert_eq!(dims, [units_height, 3, 4]);
                }
            }
        }
        assert_eq!(count, 4);
    }
}
