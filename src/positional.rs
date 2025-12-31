use burn::module::{
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RotaryEmbedding {
    Rope,
    #[default]
    Alibi,
    Pope,
}

impl RotaryEmbedding {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rope => "rope",
            Self::Pope => "pope",
            Self::Alibi => "alibi",
        }
    }
}

impl std::fmt::Display for RotaryEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<B: Backend> Module<B> for RotaryEmbedding {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for RotaryEmbedding {
    type InnerModule = RotaryEmbedding;

    fn valid(&self) -> Self::InnerModule {
        *self
    }
}

impl ModuleDisplayDefault for RotaryEmbedding {
    fn content(&self, content: Content) -> Option<Content> {
        let summary = format!("rotary_embedding={self}");
        content
            .set_top_level_type("RotaryEmbedding")
            .add_formatted(&summary)
            .optional()
    }
}

impl ModuleDisplay for RotaryEmbedding {}
