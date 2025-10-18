#[derive(Clone, Debug)]
pub struct Viz;

pub enum VizMode {
    Off,
}

impl Viz {
    pub fn new(_: VizMode) -> Self {
        Self
    }

    pub fn enabled(&self) -> bool {
        false
    }

    pub fn push(&self, _: frame::TokenFrame) {}
}

pub mod frame {
    #[derive(Clone, Debug)]
    pub struct TokenFrame;
}

pub mod collect {
    use burn::tensor::backend::Backend;
    use std::marker::PhantomData;

    use super::frame::TokenFrame;

    pub struct HeadRow<B: Backend> {
        _marker: PhantomData<B>,
        pub head: u8,
    }

    pub struct LayerTap<B: Backend> {
        pub layer: u8,
        _marker: PhantomData<B>,
    }

    pub struct StepTaps<B: Backend> {
        _marker: PhantomData<B>,
        pub t: u32,
        pub token_id: u32,
        pub token_text: String,
        pub layers: Vec<LayerTap<B>>,
    }

    pub struct CollectKnobs {
        pub k_attn: usize,
        pub k_neuron: usize,
    }

    pub fn to_frame<B: Backend>(_taps: StepTaps<B>, _knobs: &CollectKnobs) -> TokenFrame {
        TokenFrame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_viz_is_always_disabled() {
        let viz = Viz::new(VizMode::Off);
        assert!(!viz.enabled());
    }

    #[test]
    fn noop_token_frame_is_constructible() {
        let frame = frame::TokenFrame;
        let _ = frame.clone();
    }
}
