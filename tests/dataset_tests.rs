use std::fs;
use std::sync::Arc;

use burn::data::dataloader::DataLoader;
use burn::tensor::backend::Backend as BackendTrait;
use burn_dragon_hatchling::dataset::{
    DatasetSplit, RandomDataLoader, ShakespeareDataset, ShakespeareSplit, TokenSequenceDataset,
};
use burn_dragon_hatchling::tokenizer::{ByteTokenizerConfig, TokenizerConfig, TokenizerKind};
use burn_ndarray::NdArray;
use tempfile::tempdir;

#[test]
fn dataset_batches_match_expected_shape() {
    let dir = tempdir().expect("tempdir");
    let cache_dir = dir.path();
    let file_path = cache_dir.join("tinyshakespeare.txt");
    let content = b"The quick brown fox jumps over the lazy dog. ".repeat(512);
    fs::write(&file_path, content).expect("write dataset");

    let block_size = 32;
    let batch_size = 4;
    let tokenizer = TokenizerConfig::default();
    let dataset = ShakespeareDataset::new(cache_dir, block_size, batch_size, 0.8, &tokenizer)
        .expect("create dataset");

    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    <Backend as BackendTrait>::seed(&device, 0);

    let batch = dataset.sample_batch::<Backend>(ShakespeareSplit::Train, &device);
    assert_eq!(batch.inputs.shape().dims(), [batch_size, block_size]);
    assert_eq!(batch.targets.shape().dims(), [batch_size, block_size]);
}

#[test]
fn byte_tokenizer_dataset_initializes() {
    let dir = tempdir().expect("tempdir");
    let cache_dir = dir.path();
    let file_path = cache_dir.join("tinyshakespeare.txt");
    let content = b"The quick brown fox jumps over the lazy dog. ".repeat(128);
    fs::write(&file_path, content).expect("write dataset");

    let tokenizer = TokenizerConfig {
        vocab_path: None,
        kind: TokenizerKind::Byte(ByteTokenizerConfig::default()),
    };

    let dataset = ShakespeareDataset::new(cache_dir, 16, 2, 0.75, &tokenizer)
        .expect("create dataset with byte tokenizer");
    let tokenizer = dataset.tokenizer();
    assert!(tokenizer.len() >= 256);
}

#[test]
fn streamed_batches_continue_across_steps() {
    let dir = tempdir().expect("tempdir");
    let cache_dir = dir.path();
    let file_path = cache_dir.join("tinyshakespeare.txt");
    let content = "abcdefghijklmnopqrstuvwxyz".repeat(128);
    fs::write(&file_path, content).expect("write dataset");

    let block_size = 8;
    let batch_size = 4;
    let tokenizer = TokenizerConfig::default();
    let dataset = ShakespeareDataset::new(cache_dir, block_size, batch_size, 0.9, &tokenizer)
        .expect("dataset");
    let tokens: Vec<u32> = dataset.tokens().to_vec();

    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let dataset = Arc::new(dataset);
    let loader = RandomDataLoader::<Backend>::new(
        Arc::clone(&dataset),
        DatasetSplit::Train,
        &device,
        2,
        Some(2),
        0.5,
        Some(block_size),
    );

    let mut iter = loader.iter();
    let first = iter.next().expect("first batch");
    let second = iter.next().expect("second batch");

    let stream_first = first.stream.as_ref().expect("stream metadata present");
    let stream_second = second.stream.as_ref().expect("stream metadata present");
    let idx = stream_first
        .entries
        .iter()
        .position(|entry| entry.is_some())
        .expect("streamed slot exists");

    let handle_first = stream_first.entries[idx].as_ref().unwrap();
    let handle_second = stream_second.entries[idx].as_ref().unwrap();

    assert_eq!(handle_first.slot, handle_second.slot);
    assert!(Arc::ptr_eq(&handle_first.pool, &handle_second.pool));
    assert_eq!(stream_first.max_context, Some(block_size));

    if handle_first.id == handle_second.id {
        assert_eq!(handle_second.offset, handle_first.offset + block_size);
    } else {
        assert_ne!(handle_first.id, handle_second.id);
    }

    let first_tokens: Vec<i64> = first
        .inputs
        .clone()
        .slice_dim(0, idx..idx + 1)
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("tokens vec");
    let second_tokens: Vec<i64> = second
        .inputs
        .clone()
        .slice_dim(0, idx..idx + 1)
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("tokens vec");

    let expected_first: Vec<i64> = tokens[handle_first.offset..handle_first.offset + block_size]
        .iter()
        .map(|&t| t as i64)
        .collect();
    let expected_second: Vec<i64> = tokens[handle_second.offset..handle_second.offset + block_size]
        .iter()
        .map(|&t| t as i64)
        .collect();

    assert_eq!(first_tokens, expected_first);
    assert_eq!(second_tokens, expected_second);
}

#[test]
fn stream_respects_document_boundaries() {
    #[derive(Clone)]
    struct DocDataset {
        tokens: Vec<u32>,
        doc_ids: Vec<u64>,
        block: usize,
        batch: usize,
    }

    impl TokenSequenceDataset for DocDataset {
        fn tokenizer(&self) -> burn_dragon_hatchling::tokenizer::SharedTokenizer {
            panic!("tokenizer not needed")
        }
        fn tokens(&self) -> &[u32] {
            &self.tokens
        }
        fn doc_ids(&self) -> Option<&[u64]> {
            Some(&self.doc_ids)
        }
        fn train_len(&self) -> usize {
            self.tokens.len()
        }
        fn block_size(&self) -> usize {
            self.block
        }
        fn batch_size(&self) -> usize {
            self.batch
        }
        fn train_split_ratio(&self) -> f32 {
            1.0
        }
    }

    let block_size = 4;
    let batch_size = 2;
    // Three documents of length 8 each: [0..8), [8..16), [16..24)
    let tokens: Vec<u32> = (0..24).collect();
    let mut doc_ids = Vec::with_capacity(tokens.len());
    doc_ids.extend(std::iter::repeat(0u64).take(8));
    doc_ids.extend(std::iter::repeat(1u64).take(8));
    doc_ids.extend(std::iter::repeat(2u64).take(8));

    let dataset = DocDataset {
        tokens,
        doc_ids: doc_ids.clone(),
        block: block_size,
        batch: batch_size,
    };

    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let loader = RandomDataLoader::<Backend>::new(
        Arc::new(dataset),
        DatasetSplit::Train,
        &device,
        6,
        Some(6),
        1.0,
        Some(block_size),
    );

    let mut iter = loader.iter();
    for _ in 0..6 {
        let batch = iter.next().expect("streamed batch");
        let stream = batch.stream.as_ref().expect("stream metadata");
        let handle = stream
            .entries
            .iter()
            .find_map(|entry| entry.as_ref())
            .expect("stream handle");
        let offset = handle.offset;

        // All tokens in this sample should share a single doc id.
        let sample_doc_ids = &doc_ids[offset..offset + block_size];
        assert!(
            sample_doc_ids.windows(2).all(|w| w[0] == w[1]),
            "streamed sample should not cross doc boundary"
        );
    }
}
