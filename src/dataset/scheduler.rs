use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use rand::prelude::*;
use rand::seq::SliceRandom;

use crate::model::RecurrentStateStore;
use crate::tokenizer::SharedTokenizer;

use super::DatasetSplit;

/// Abstraction over text corpora that can be converted into BDH-compatible batches.
pub trait TokenSequenceDataset: Send + Sync {
    /// Return a shared tokenizer handle (cloned per call).
    fn tokenizer(&self) -> SharedTokenizer;

    /// Return the full sequence of token ids representing the corpus.
    fn tokens(&self) -> &[u32];

    /// Optional per-token document identifiers used to reset streams at boundaries.
    fn doc_ids(&self) -> Option<&[u64]> {
        None
    }

    /// Number of tokens reserved for the training split from the start of `tokens`.
    fn train_len(&self) -> usize;

    /// Maximum sequence length per sample.
    fn block_size(&self) -> usize;

    /// Number of sequences per batch.
    fn batch_size(&self) -> usize;

    /// Ratio used when determining train/validation split boundaries.
    fn train_split_ratio(&self) -> f32;

    /// Provide the offset and span of the requested split.
    fn split_offset_and_span(&self, split: DatasetSplit) -> (usize, usize) {
        match split {
            DatasetSplit::Train => (0, self.train_len()),
            DatasetSplit::Val => {
                let tokens = self.tokens();
                let train_len = self.train_len();
                let remaining = tokens.len().saturating_sub(train_len);
                if remaining <= self.block_size() + 1 {
                    (0, train_len)
                } else {
                    (train_len, remaining)
                }
            }
        }
    }

    /// Number of steps per epoch for a given split (defaults derived from token counts).
    fn steps_per_epoch(&self, split: DatasetSplit) -> usize {
        let (_offset, span) = self.split_offset_and_span(split);
        let tokens_per_step = self.block_size() * self.batch_size();
        if tokens_per_step == 0 {
            return 1;
        }
        let steps = span.div_ceil(tokens_per_step);
        steps.max(1)
    }

    /// Decode token ids back into text.
    fn decode(&self, tokens: &[i64]) -> String {
        let ids: Vec<u32> = tokens
            .iter()
            .filter_map(|&tok| (tok >= 0).then_some(tok as u32))
            .collect();
        self.tokenizer().decode(&ids)
    }
}

/// Sample a random batch from any dataset implementing [`TokenSequenceDataset`].
pub fn sample_batch<B: Backend, T: TokenSequenceDataset + ?Sized>(
    dataset: &T,
    split: DatasetSplit,
    device: &B::Device,
) -> SequenceBatch<B> {
    let tokens = dataset.tokens();
    let (offset, span) = dataset.split_offset_and_span(split);

    let mut rng = thread_rng();
    let mut inputs = vec![0i64; dataset.batch_size() * dataset.block_size()];
    let mut targets = vec![0i64; dataset.batch_size() * dataset.block_size()];

    for batch_idx in 0..dataset.batch_size() {
        let max_start = span.saturating_sub(dataset.block_size() + 1);
        let start_offset = if max_start == 0 {
            0
        } else {
            rng.gen_range(0..=max_start)
        };
        let start = offset + start_offset;
        for t in 0..dataset.block_size() {
            let data_idx = start + t;
            inputs[batch_idx * dataset.block_size() + t] = tokens[data_idx] as i64;
            targets[batch_idx * dataset.block_size() + t] = tokens[data_idx + 1] as i64;
        }
    }

    let inputs_tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(inputs, [dataset.batch_size(), dataset.block_size()]),
        device,
    );
    let targets_tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(targets, [dataset.batch_size(), dataset.block_size()]),
        device,
    );

    SequenceBatch::new(inputs_tensor, targets_tensor)
}

/// Batched token inputs and targets for language modeling.
#[derive(Clone)]
pub struct SequenceBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
    pub stream: Option<StreamBatchState<B>>,
}

impl<B: Backend> SequenceBatch<B> {
    pub fn new(inputs: Tensor<B, 2, Int>, targets: Tensor<B, 2, Int>) -> Self {
        Self {
            inputs,
            targets,
            stream: None,
        }
    }

    pub fn with_stream(
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        stream: StreamBatchState<B>,
    ) -> Self {
        Self {
            inputs,
            targets,
            stream: Some(stream),
        }
    }
}

#[derive(Clone)]
pub struct StreamHandle<B: Backend> {
    pub id: u64,
    pub offset: usize,
    pub slot: usize,
    pub pool: Arc<Mutex<RecurrentStateStore<B>>>,
}

#[derive(Clone)]
pub struct StreamBatchState<B: Backend> {
    pub entries: Vec<Option<StreamHandle<B>>>,
    pub max_context: Option<usize>,
    pub pool: Arc<Mutex<RecurrentStateStore<B>>>,
}

impl<B: Backend> StreamBatchState<B> {
    pub fn has_streams(&self) -> bool {
        self.entries.iter().any(|entry| entry.is_some())
    }
}

/// Data loader that produces random sequences from any `TokenSequenceDataset`.
pub struct RandomDataLoader<B: Backend> {
    dataset: Arc<dyn TokenSequenceDataset>,
    split: DatasetSplit,
    device: B::Device,
    steps_per_epoch: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
    stream_retain_pct: f32,
    stream_context_len: Option<usize>,
}

impl<B: Backend> Clone for RandomDataLoader<B> {
    fn clone(&self) -> Self {
        Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            stream_retain_pct: self.stream_retain_pct,
            stream_context_len: self.stream_context_len,
        }
    }
}

impl<B: Backend> RandomDataLoader<B> {
    pub fn new<T>(
        dataset: Arc<T>,
        split: DatasetSplit,
        device: &B::Device,
        steps_per_epoch: usize,
        total_steps: Option<usize>,
        stream_retain_pct: f32,
        stream_context_len: Option<usize>,
    ) -> Self
    where
        T: TokenSequenceDataset + 'static,
    {
        let dataset: Arc<dyn TokenSequenceDataset> = dataset;
        let steps_per_epoch = steps_per_epoch.max(1);
        let total_steps = total_steps.filter(|value| *value > 0);
        let consumed_steps = total_steps.as_ref().map(|_| Arc::new(AtomicUsize::new(0)));
        let stream_retain_pct = stream_retain_pct.clamp(0.0, 1.0);

        Self {
            dataset,
            split,
            device: device.clone(),
            steps_per_epoch,
            total_steps,
            consumed_steps,
            stream_retain_pct,
            stream_context_len,
        }
    }
}

impl<B> DataLoader<B, SequenceBatch<B>> for RandomDataLoader<B>
where
    B: Backend + 'static,
    B::Device: Clone,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<SequenceBatch<B>> + 'a> {
        let steps_total =
            if let (Some(limit), Some(consumed)) = (self.total_steps, &self.consumed_steps) {
                let used = consumed.load(Ordering::Relaxed);
                if used >= limit {
                    0
                } else {
                    (limit - used).min(self.steps_per_epoch)
                }
            } else {
                self.steps_per_epoch
            };

        let (split_offset, split_span) = self.dataset.split_offset_and_span(self.split);
        let block_size = self.dataset.block_size();
        let batch_size = self.dataset.batch_size();
        let doc_ranges =
            build_doc_ranges(self.dataset.doc_ids(), split_offset, split_span, block_size);
        let stream_pool =
            if self.stream_retain_pct > 0.0 && matches!(self.split, DatasetSplit::Train) {
                let max_stream = ((batch_size as f32) * self.stream_retain_pct).floor() as usize;
                if max_stream == 0 {
                    None
                } else {
                    Some(StreamPool::new(
                        max_stream,
                        self.stream_context_len,
                        split_offset,
                        split_span,
                        block_size,
                        doc_ranges.clone(),
                    ))
                }
            } else {
                None
            };

        Box::new(RandomIterator {
            dataset: Arc::clone(&self.dataset),
            device: self.device.clone(),
            steps_total,
            step: 0,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.clone(),
            stream_pool,
            split_offset,
            split_span,
            block_size,
            batch_size,
            doc_ranges,
        })
    }

    fn num_items(&self) -> usize {
        self.steps_per_epoch * self.dataset.batch_size()
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: device.clone(),
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            stream_retain_pct: self.stream_retain_pct,
            stream_context_len: self.stream_context_len,
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
        let end = end.min(self.steps_per_epoch);
        let start = start.min(end);
        let steps = (end - start).max(1);

        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            steps_per_epoch: steps,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            stream_retain_pct: self.stream_retain_pct,
            stream_context_len: self.stream_context_len,
        })
    }
}

struct RandomIterator<B: Backend> {
    dataset: Arc<dyn TokenSequenceDataset>,
    device: B::Device,
    steps_total: usize,
    step: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
    stream_pool: Option<StreamPool<B>>,
    split_offset: usize,
    split_span: usize,
    block_size: usize,
    batch_size: usize,
    doc_ranges: Option<Vec<Range<usize>>>,
}

#[derive(Clone)]
struct StreamCursor<B: Backend> {
    offset: usize,
    slot: usize,
    pool: Arc<Mutex<RecurrentStateStore<B>>>,
    id: u64,
    range_idx: Option<usize>,
}

struct StreamPool<B: Backend> {
    slots: Vec<StreamCursor<B>>,
    max_stream: usize,
    next_id: u64,
    split_offset: usize,
    split_span: usize,
    block_size: usize,
    context_len: Option<usize>,
    doc_ranges: Option<Vec<Range<usize>>>,
    state_pool: Arc<Mutex<RecurrentStateStore<B>>>,
}

impl<B: Backend> StreamPool<B> {
    fn new(
        max_stream: usize,
        context_len: Option<usize>,
        split_offset: usize,
        split_span: usize,
        block_size: usize,
        doc_ranges: Option<Vec<Range<usize>>>,
    ) -> Self {
        Self {
            slots: Vec::new(),
            max_stream,
            next_id: 0,
            split_offset,
            split_span,
            block_size,
            context_len,
            doc_ranges,
            state_pool: Arc::new(Mutex::new(RecurrentStateStore::new(max_stream))),
        }
    }

    fn sample_start(&self, rng: &mut ThreadRng) -> (usize, Option<usize>, Option<usize>) {
        if let Some(ranges) = &self.doc_ranges {
            if let Some(offset) = sample_from_ranges(ranges, self.block_size, rng) {
                let idx = find_range_index(ranges, offset).unwrap_or(0);
                let end = ranges.get(idx).map(|r| r.end);
                return (offset, end, Some(idx));
            }
        }

        let max_start = self
            .split_span
            .saturating_sub(self.block_size + 1)
            .min(self.split_span);
        let relative = if max_start == 0 {
            0
        } else {
            rng.gen_range(0..=max_start)
        };
        (relative, None, None)
    }

    fn prepare_batch(&mut self, batch_size: usize, rng: &mut ThreadRng) -> StreamBatchState<B> {
        let desired = self.max_stream.min(batch_size);
        while self.slots.len() < desired {
            let (start, _range_end, range_idx) = self.sample_start(rng);
            let id = self.alloc_id();
            self.slots.push(StreamCursor {
                offset: start,
                slot: self.slots.len(),
                pool: Arc::clone(&self.state_pool),
                id,
                range_idx,
            });
        }

        let mut entries = Vec::with_capacity(batch_size);
        for idx in 0..desired {
            let cursor = self.slots.get_mut(idx).expect("stream cursor initialized");
            let abs_offset = self.split_offset + cursor.offset;
            entries.push(Some(StreamHandle {
                id: cursor.id,
                offset: abs_offset,
                slot: cursor.slot,
                pool: Arc::clone(&cursor.pool),
            }));
            cursor.advance(
                &mut self.next_id,
                self.block_size,
                self.split_span,
                rng,
                self.doc_ranges.as_ref(),
            );
        }
        while entries.len() < batch_size {
            entries.push(None);
        }

        StreamBatchState {
            entries,
            max_context: self.context_len,
            pool: Arc::clone(&self.state_pool),
        }
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        id
    }
}

impl<B: Backend> StreamCursor<B> {
    fn advance(
        &mut self,
        next_id: &mut u64,
        block_size: usize,
        span: usize,
        rng: &mut ThreadRng,
        doc_ranges: Option<&Vec<Range<usize>>>,
    ) {
        let can_advance = if let Some(ranges) = doc_ranges
            && let Some(idx) = self.range_idx
            && let Some(range) = ranges.get(idx)
        {
            let next_start = self.offset + block_size;
            let limit = range.end;
            let exceeds = next_start + block_size + 1 > limit;
            !exceeds
        } else {
            let max_start = span.saturating_sub(block_size + 1);
            let next_start = self.offset + block_size;
            next_start + block_size + 1 <= span && max_start > 0
        };

        if can_advance {
            self.offset += block_size;
            return;
        }

        let (start, _range_end, range_idx) = if let Some(ranges) = doc_ranges {
            match sample_from_ranges(ranges, block_size, rng) {
                Some(offset) => {
                    let idx = find_range_index(ranges, offset).unwrap_or(0);
                    let end = ranges.get(idx).map(|r| r.end);
                    (offset, end, Some(idx))
                }
                None => (0, None, None),
            }
        } else {
            let max_start = span.saturating_sub(block_size + 1);
            let start = if max_start == 0 {
                0
            } else {
                rng.gen_range(0..=max_start)
            };
            (start, None, None)
        };

        self.offset = start;
        self.range_idx = range_idx;
        *next_id = (*next_id).wrapping_add(1);
        self.id = *next_id;
        if let Ok(mut guard) = self.pool.lock() {
            guard.reset_slot(self.slot);
        }
    }
}

impl<B: Backend> Iterator for RandomIterator<B> {
    type Item = SequenceBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step >= self.steps_total {
            return None;
        }
        self.step += 1;

        if let Some(counter) = &self.consumed_steps {
            if let Some(limit) = self.total_steps {
                let previous = counter.fetch_add(1, Ordering::Relaxed);
                if previous >= limit {
                    return None;
                }
            } else {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }

        let mut rng = thread_rng();

        let stream_state = self
            .stream_pool
            .as_mut()
            .map(|pool| pool.prepare_batch(self.batch_size, &mut rng));

        let mut starts = Vec::with_capacity(self.batch_size);
        for idx in 0..self.batch_size {
            if let Some(handle) = stream_state
                .as_ref()
                .and_then(|state| state.entries.get(idx))
                .and_then(|entry| entry.as_ref())
            {
                starts.push(handle.offset);
            } else {
                starts.push(self.random_offset(&mut rng));
            }
        }

        let mut inputs = vec![0i64; self.batch_size * self.block_size];
        let mut targets = vec![0i64; self.batch_size * self.block_size];

        for (batch_idx, start) in starts.into_iter().enumerate() {
            let base = batch_idx * self.block_size;
            for t in 0..self.block_size {
                let data_idx = start + t;
                inputs[base + t] = self.dataset.tokens()[data_idx] as i64;
                targets[base + t] = self.dataset.tokens()[data_idx + 1] as i64;
            }
        }

        let inputs_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(inputs, [self.batch_size, self.block_size]),
            &self.device,
        );
        let targets_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(targets, [self.batch_size, self.block_size]),
            &self.device,
        );

        let has_streams = stream_state
            .as_ref()
            .map(|state| state.has_streams())
            .unwrap_or(false);

        Some(if has_streams {
            SequenceBatch::with_stream(
                inputs_tensor,
                targets_tensor,
                stream_state.expect("stream state available"),
            )
        } else {
            SequenceBatch::new(inputs_tensor, targets_tensor)
        })
    }
}

impl<B: Backend> DataLoaderIterator<SequenceBatch<B>> for RandomIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(
            self.step * self.dataset.batch_size(),
            self.steps_total * self.dataset.batch_size(),
        )
    }
}

impl<B: Backend> RandomIterator<B> {
    fn random_offset(&self, rng: &mut ThreadRng) -> usize {
        if let Some(ranges) = &self.doc_ranges {
            return sample_from_ranges(ranges, self.block_size, rng)
                .map(|offset| self.split_offset + offset)
                .unwrap_or(self.split_offset);
        }

        let max_start = self
            .split_span
            .saturating_sub(self.block_size + 1)
            .min(self.split_span);
        let relative = if max_start == 0 {
            0
        } else {
            rng.gen_range(0..=max_start)
        };
        self.split_offset + relative
    }
}

fn build_doc_ranges(
    doc_ids: Option<&[u64]>,
    split_offset: usize,
    split_span: usize,
    block_size: usize,
) -> Option<Vec<Range<usize>>> {
    let ids = doc_ids?;
    if split_offset >= ids.len() {
        return None;
    }
    let end = (split_offset + split_span).min(ids.len());
    if end <= split_offset || block_size == 0 {
        return None;
    }

    let mut ranges = Vec::new();
    let mut current_start = split_offset;
    let mut current_id = ids.get(split_offset)?;

    for idx in (split_offset + 1)..end {
        let doc_id = &ids[idx];
        if doc_id != current_id {
            ranges.push(current_start - split_offset..idx - split_offset);
            current_start = idx;
            current_id = doc_id;
        }
    }
    ranges.push(current_start - split_offset..end - split_offset);

    ranges.retain(|range| range.len() >= block_size + 1);
    if ranges.is_empty() {
        None
    } else {
        Some(ranges)
    }
}

fn sample_from_ranges(
    ranges: &[Range<usize>],
    block_size: usize,
    rng: &mut ThreadRng,
) -> Option<usize> {
    let eligible: Vec<&Range<usize>> = ranges
        .iter()
        .filter(|range| range.len() >= block_size + 1)
        .collect();
    let range = eligible.choose(rng)?;
    let max_start = range.end.saturating_sub(block_size + 1);
    Some(rng.gen_range(range.start..=max_start))
}

fn find_range_index(ranges: &[Range<usize>], offset: usize) -> Option<usize> {
    ranges
        .iter()
        .position(|range| offset >= range.start && offset < range.end)
}
