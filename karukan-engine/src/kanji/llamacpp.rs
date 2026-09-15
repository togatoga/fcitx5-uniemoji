//! llama.cpp based GGUF inference for kanji conversion
//!
//! This module provides an alternative to Candle's GGUF implementation using
//! llama.cpp's optimized inference engine via the llama-cpp-2 crate.
//!
//! Enable with the `llamacpp` feature flag.

use super::error::KanjiError;
type Result<T> = super::error::Result<T>;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::collections::HashSet;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

/// Global llama.cpp backend (can only be initialized once)
static LLAMA_BACKEND: OnceLock<std::result::Result<LlamaBackend, String>> = OnceLock::new();

/// Get or initialize the global llama.cpp backend
fn get_backend() -> Result<&'static LlamaBackend> {
    let result = LLAMA_BACKEND.get_or_init(|| {
        let mut backend = LlamaBackend::init().map_err(|e| e.to_string())?;
        backend.void_logs();
        Ok(backend)
    });
    match result {
        Ok(backend) => Ok(backend),
        Err(e) => Err(KanjiError::ModelLoad(
            format!("Failed to initialize llama.cpp backend: {}", e).into(),
        )),
    }
}

/// Beam ceiling: live + scratch slots must stay within llama.cpp's
/// LLAMA_MAX_SEQ (256), which throws a C++ exception that aborts across FFI.
const MAX_BEAM_SIZE: usize = 128;

/// Spare KV cells beyond the computed prompt + generation rows.
const KV_HEADROOM_CELLS: usize = 64;

/// Spare batch slots beyond the computed prompt / sequence rows.
const BATCH_HEADROOM: usize = 8;

/// Pack `s` into the fixed-size buffer llama.cpp reads override values out
/// of: a NUL-terminated C string in a 128-byte array (`val_str` in
/// `llama_model_kv_override`). The array starts zeroed, so the bytes left
/// after `s` are the terminator; input longer than the buffer is truncated
/// one byte short so the terminator always survives.
fn kv_override_str(s: &str) -> [std::os::raw::c_char; 128] {
    let mut buf = [0; 128];
    let writable = buf.len() - 1;
    for (dst, &byte) in buf.iter_mut().zip(s.as_bytes()).take(writable) {
        *dst = byte as std::os::raw::c_char;
    }
    buf
}

/// Wrap any llama.cpp error as [`KanjiError::Inference`].
fn inference_err(e: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> KanjiError {
    KanjiError::Inference(e.into())
}

/// Convert bytes to hex display format for partial UTF-8 sequences
fn bytes_to_hex_display(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("<{:02X}>", b)).collect()
}

/// Whether an added-token string is a byte-fallback token like `<0xCE>`.
/// tokenizer.json marks them `special: true`, but they carry real output
/// bytes and must never be skipped as special. The check mirrors the
/// tokenizer's own `ByteFallback` decoder match exactly.
fn is_byte_fallback_token(token: &str) -> bool {
    token.len() == 6
        && token.starts_with("<0x")
        && token.ends_with('>')
        && u8::from_str_radix(&token[3..5], 16).is_ok()
}

/// Load and configure an external HuggingFace tokenizer from a `tokenizer.json` file.
fn load_tokenizer<P: AsRef<Path>>(path: P) -> Result<tokenizers::Tokenizer> {
    let mut tokenizer =
        tokenizers::Tokenizer::from_file(path.as_ref()).map_err(KanjiError::TokenizerLoad)?;
    // Disable padding and truncation — we handle sequence length ourselves
    // and padding tokens would corrupt the model input.
    tokenizer.with_padding(None);
    tokenizer.with_truncation(None).ok();
    Ok(tokenizer)
}

/// A beam candidate with generated tokens and cumulative score
#[derive(Clone)]
struct BeamState {
    tokens: Vec<LlamaToken>,
    score: f32,
}

/// KV state the greedy path carries between calls: a persistent context and
/// the token sequence its cache currently represents (last prompt +
/// generated tokens). Typing grows the prompt at the tail, so the next call
/// usually only has to decode the few tokens past the common prefix.
struct GreedySession {
    /// Borrows the `Box<LlamaModel>` next to it in [`LlamaCppModel`]; the
    /// lifetime is erased there under the invariants documented on the field.
    ctx: LlamaContext<'static>,
    cached: Vec<LlamaToken>,
}

// SAFETY: the context is only touched while holding the owning model's
// mutex, so accesses never overlap; llama.cpp allows moving a context
// between threads when calls are serialized.
unsafe impl Send for GreedySession {}

/// Length of the longest common prefix of two token sequences.
fn common_prefix_len(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// llama.cpp based GPT-2 model for GGUF inference
pub struct LlamaCppModel {
    /// Persistent context for the greedy path. Declared before `model` so it
    /// drops first: its context borrows the model behind the box.
    greedy_session: Mutex<Option<GreedySession>>,
    /// Boxed for a stable address — `greedy_session` holds a context whose
    /// model reference points into this allocation, so the box must never be
    /// replaced after construction.
    model: Box<LlamaModel>,
    n_ctx: u32,
    /// External HuggingFace tokenizer (always required).
    /// `tokenize()` and `decode()` use this instead of llama.cpp's built-in tokenizer.
    external_tokenizer: tokenizers::Tokenizer,
    /// Token ids `decode(_, skip_special_tokens=true)` removes before
    /// detokenizing: added tokens with `special: true`, minus the
    /// byte-fallback tokens (see [`is_byte_fallback_token`]).
    special_token_ids: HashSet<u32>,
    /// Number of threads for inference (0 = use llama.cpp default)
    n_threads: u32,
}

/// llama.cpp aborts the process when the model file is absent, so check
/// first and fail as an ordinary error the caller can degrade on.
fn ensure_model_file_exists(path: &Path) -> Result<()> {
    if !path.exists() {
        return Err(KanjiError::ModelLoad(
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("model file not found: {}", path.display()),
            )
            .into(),
        ));
    }
    Ok(())
}

impl LlamaCppModel {
    /// Load a GGUF model using llama.cpp with an external tokenizer.
    ///
    /// GPT-2 models use CPU only (Metal has issues with GPT-2).
    pub fn from_file<P: AsRef<Path>, T: AsRef<Path>>(path: P, tokenizer_json: T) -> Result<Self> {
        Self::from_file_with_n_ctx(path, tokenizer_json, 256)
    }

    /// Load a GGUF model with a pre-tokenizer type override.
    ///
    /// Some models use custom pre-tokenizer types (e.g., `gpt2-small-japanese-char`)
    /// that llama.cpp doesn't recognize. This method overrides the `tokenizer.ggml.pre`
    /// metadata key to a compatible type before loading.
    pub fn from_file_with_pre_tokenizer_override<P: AsRef<Path>, T: AsRef<Path>>(
        path: P,
        tokenizer_json: T,
        pre_tokenizer: &str,
    ) -> Result<Self> {
        use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
        use std::ffi::CString;
        use std::pin::pin;

        ensure_model_file_exists(path.as_ref())?;
        let backend = get_backend()?;

        let mut params = pin!(LlamaModelParams::default().with_n_gpu_layers(0));

        let key =
            CString::new("tokenizer.ggml.pre").map_err(|e| KanjiError::ModelLoad(e.into()))?;
        params.as_mut().append_kv_override(
            &key,
            ParamOverrideValue::Str(kv_override_str(pre_tokenizer)),
        );

        let model = LlamaModel::load_from_file(backend, path.as_ref(), &params)
            .map_err(|e| KanjiError::ModelLoad(e.into()))?;
        Self::finish(model, tokenizer_json, 256)
    }

    /// Load a GGUF model with explicit context window size
    pub fn from_file_with_n_ctx<P: AsRef<Path>, T: AsRef<Path>>(
        path: P,
        tokenizer_json: T,
        n_ctx: u32,
    ) -> Result<Self> {
        ensure_model_file_exists(path.as_ref())?;
        let backend = get_backend()?;

        // GPT-2 has Metal issues, use CPU
        let model_params = LlamaModelParams::default().with_n_gpu_layers(0);

        let model = LlamaModel::load_from_file(backend, path.as_ref(), &model_params)
            .map_err(|e| KanjiError::ModelLoad(e.into()))?;
        Self::finish(model, tokenizer_json, n_ctx)
    }

    /// Load the external tokenizer and construct the model wrapper.
    fn finish<T: AsRef<Path>>(model: LlamaModel, tokenizer_json: T, n_ctx: u32) -> Result<Self> {
        let external_tokenizer = load_tokenizer(tokenizer_json)?;
        let special_token_ids = external_tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter(|(_, tok)| tok.special && !is_byte_fallback_token(&tok.content))
            .map(|(id, _)| id)
            .collect();

        Ok(Self {
            greedy_session: Mutex::new(None),
            model: Box::new(model),
            n_ctx,
            external_tokenizer,
            special_token_ids,
            n_threads: 0,
        })
    }

    /// Set the number of threads for inference.
    /// 0 means use llama.cpp default (typically all cores).
    pub fn set_n_threads(&mut self, n: u32) {
        self.n_threads = n;
        // The session's context was built with the old thread count.
        *self.lock_greedy_session() = None;
    }

    fn lock_greedy_session(&self) -> std::sync::MutexGuard<'_, Option<GreedySession>> {
        self.greedy_session
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Build LlamaContextParams with configured n_threads
    fn context_params(&self) -> LlamaContextParams {
        self.context_params_with_n_ctx(self.n_ctx)
    }

    /// New inference context on the global backend.
    fn new_context(&self, params: LlamaContextParams) -> Result<LlamaContext<'_>> {
        let backend = get_backend()?;
        self.model
            .new_context(backend, params)
            .map_err(inference_err)
    }

    /// Build LlamaContextParams with configured n_threads and an explicit KV
    /// cache size. Beam search needs more cells than a greedy run: every beam
    /// keeps its own generated tokens alongside the shared prompt.
    fn context_params_with_n_ctx(&self, n_ctx: u32) -> LlamaContextParams {
        let params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(n_ctx).unwrap_or(NonZeroU32::MIN)));
        if self.n_threads > 0 {
            params
                .with_n_threads(self.n_threads as i32)
                .with_n_threads_batch(self.n_threads as i32)
        } else {
            params
        }
    }

    /// Tokenize a string using the external tokenizer
    pub fn tokenize(&self, text: &str) -> Result<Vec<LlamaToken>> {
        let encoding = self
            .external_tokenizer
            .encode(text, false)
            .map_err(KanjiError::Inference)?;
        let tokens: Vec<LlamaToken> = encoding
            .get_ids()
            .iter()
            .map(|&id| LlamaToken(id as i32))
            .collect();
        Ok(tokens)
    }

    /// Decode tokens via the external tokenizer. Special tokens are
    /// filtered here by id, not by the tokenizer's own skip flag — that
    /// would also drop byte-fallback tokens (`<0xCE>`…) before its
    /// ByteFallback decoder can fuse them back into characters.
    pub fn decode(&self, tokens: &[LlamaToken], skip_special_tokens: bool) -> Result<String> {
        let ids: Vec<u32> = tokens
            .iter()
            .map(|t| t.0 as u32)
            .filter(|id| !(skip_special_tokens && self.special_token_ids.contains(id)))
            .collect();
        let text = self
            .external_tokenizer
            .decode(&ids, false)
            .map_err(KanjiError::Inference)?;
        Ok(text)
    }

    /// Decode a single token for display purposes.
    ///
    /// For byte-level BPE tokens that represent partial UTF-8 sequences,
    /// this returns a hex representation like `<0xE3>` instead of replacement characters.
    pub fn decode_token_for_display(&self, token: LlamaToken) -> String {
        match self.model.token_to_piece_bytes(token, 32, true, None) {
            Ok(bytes) => {
                if let Ok(s) = std::str::from_utf8(&bytes) {
                    // Valid UTF-8, return as-is (escape control chars)
                    if s.chars().all(|c| !c.is_control() || c == ' ' || c == '\n') {
                        s.to_string()
                    } else {
                        // Has control characters, show hex
                        bytes_to_hex_display(&bytes)
                    }
                } else {
                    // Invalid UTF-8 (partial sequence), show hex
                    bytes_to_hex_display(&bytes)
                }
            }
            Err(_) => format!("<{}>", token.0),
        }
    }

    /// Generate tokens with greedy decoding.
    ///
    /// Runs on a persistent context and re-decodes only the part of the
    /// prompt past the longest common prefix with the previous call, so a
    /// prompt that grows keystroke by keystroke costs a few tokens instead
    /// of a full prefill. Falls back to a one-shot context when the prompt
    /// cannot fit the session.
    pub fn generate(
        &self,
        input_tokens: &[LlamaToken],
        max_new_tokens: usize,
        eos_token_id: Option<i32>,
    ) -> Result<Vec<LlamaToken>> {
        if input_tokens.is_empty() || input_tokens.len() + max_new_tokens >= self.n_ctx as usize {
            return self.generate_with_sampler(
                input_tokens,
                max_new_tokens,
                eos_token_id,
                LlamaSampler::greedy(),
            );
        }

        let mut guard = self.lock_greedy_session();
        if guard.is_none() {
            *guard = Some(self.new_greedy_session()?);
        }
        let session = guard.as_mut().expect("session initialized above");
        let result = self.run_greedy(session, input_tokens, max_new_tokens, eos_token_id);
        if result.is_err() {
            // The cache no longer matches `cached`; rebuild next call.
            *guard = None;
        }
        result
    }

    /// A fresh persistent context for the greedy path. Batches are sized to
    /// n_ctx: unlike the one-shot paths this allocation happens once, not
    /// per conversion, so there is nothing to win by shrinking it.
    fn new_greedy_session(&self) -> Result<GreedySession> {
        let params = self
            .context_params()
            .with_n_batch(self.n_ctx)
            .with_n_ubatch(self.n_ctx);
        let ctx = self.new_context(params)?;
        // SAFETY: the context borrows the model inside `self.model`'s box,
        // whose address is stable and which is never replaced; the session
        // lives in a field declared before `model`, so it drops first.
        let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
        Ok(GreedySession {
            ctx,
            cached: Vec::new(),
        })
    }

    /// Greedy generation on the session context, reusing the KV of the
    /// longest common prefix between `input_tokens` and the previous call.
    fn run_greedy(
        &self,
        session: &mut GreedySession,
        input_tokens: &[LlamaToken],
        max_new_tokens: usize,
        eos_token_id: Option<i32>,
    ) -> Result<Vec<LlamaToken>> {
        let n = input_tokens.len();
        let common = common_prefix_len(&session.cached, input_tokens);
        // Re-decode at least the final prompt token: sampling needs its
        // logits even when the whole prompt is already cached.
        let start = common.min(n - 1);

        session
            .ctx
            .kv_cache_seq_rm(0, Some(start as u32), None)
            .map_err(inference_err)?;
        // Invalidate before decoding: a failed decode leaves the KV cache in
        // an unknown state, and `generate` then discards the session.
        session.cached.clear();

        let mut batch = LlamaBatch::new(self.n_ctx as usize, 1);
        for (i, token) in input_tokens.iter().enumerate().skip(start) {
            batch
                .add(*token, i as i32, &[0], i == n - 1)
                .map_err(inference_err)?;
        }
        session.ctx.decode(&mut batch).map_err(inference_err)?;

        let mut sampler = LlamaSampler::greedy();
        let model_eos = self.model.token_eos();
        let mut generated = input_tokens.to_vec();
        for n_cur in (n..).take(max_new_tokens) {
            let new_token = sampler.sample(&session.ctx, -1);
            if self.is_eos_token(new_token, eos_token_id, model_eos) {
                break;
            }
            generated.push(new_token);
            batch.clear();
            batch
                .add(new_token, n_cur as i32, &[0], true)
                .map_err(inference_err)?;
            session.ctx.decode(&mut batch).map_err(inference_err)?;
        }

        session.cached = generated.clone();
        Ok(generated)
    }

    /// Depth-1 beam: pick the top-k initial tokens, then continue each
    /// sequence greedily and independently, all sharing one prompt KV cache.
    /// Faster than true beam search but may miss globally optimal
    /// candidates. Sorted by initial token probability.
    pub fn generate_beam_search_d1_greedy(
        &self,
        input_tokens: &[LlamaToken],
        max_new_tokens: usize,
        eos_token_id: Option<i32>,
        beam_size: usize,
    ) -> Result<Vec<(Vec<LlamaToken>, f32)>> {
        // Set n_batch and n_ubatch large enough to avoid batch splitting
        // which causes "coupled sequences" error
        let batch_size = input_tokens
            .len()
            .saturating_mul(beam_size)
            .saturating_add(64)
            .min(u32::MAX as usize) as u32;
        let mut ctx = self.new_context(
            self.context_params()
                .with_n_seq_max(beam_size.try_into().unwrap_or(32))
                .with_n_batch(batch_size)
                .with_n_ubatch(batch_size),
        )?;

        let model_eos = self.model.token_eos();
        let input_len = input_tokens.len();

        // Step 1: Process input tokens for ALL sequences in one batch
        // Add each token separately for each sequence (not coupled)
        let mut batch = LlamaBatch::new(512, 1);

        for (i, token) in input_tokens.iter().enumerate() {
            for seq_id in 0..beam_size as i32 {
                let is_last = i == input_len - 1 && seq_id == 0; // Only first seq needs logits
                batch
                    .add(*token, i as i32, &[seq_id], is_last)
                    .map_err(inference_err)?;
            }
        }
        ctx.decode(&mut batch).map_err(inference_err)?;

        // Step 2: Get top-k initial tokens (from any seq, all have same logits at this point)
        let top = self.get_top_k_tokens(ctx.get_logits(), beam_size);

        // Step 3: Initialize beam state. A beam whose initial token is
        // already EOS is finished before it generates anything.
        let mut beam_tokens: Vec<Vec<LlamaToken>> = top.iter().map(|&(t, _)| vec![t]).collect();
        let beam_scores: Vec<f32> = top.iter().map(|&(_, score)| score).collect();
        let mut beam_finished: Vec<bool> = top
            .iter()
            .map(|&(token, _)| self.is_eos_token(token, eos_token_id, model_eos))
            .collect();

        // Step 4: Add initial tokens to each beam's sequence
        batch.clear();
        for (beam_idx, &(token, _)) in top.iter().enumerate() {
            if !beam_finished[beam_idx] {
                batch
                    .add(token, input_len as i32, &[beam_idx as i32], true)
                    .map_err(inference_err)?;
            }
        }

        if batch.n_tokens() > 0 {
            ctx.decode(&mut batch).map_err(inference_err)?;
        }

        // Step 5: Generate tokens for all beams in parallel. Greedy sampling
        // is stateless, so one sampler serves every beam.
        let mut sampler = LlamaSampler::greedy();

        for _step in 0..(max_new_tokens - 1) {
            if beam_finished.iter().all(|&finished| finished) {
                break;
            }

            // Sample the next token per active beam. The logit row is the
            // beam's position in the previous batch, i.e. its index among the
            // active beams.
            let sampled: Vec<(usize, LlamaToken)> = beam_finished
                .iter()
                .enumerate()
                .filter(|&(_, &finished)| !finished)
                .enumerate()
                .map(|(logit_idx, (beam_idx, _))| {
                    (beam_idx, sampler.sample(&ctx, logit_idx as i32))
                })
                .collect();

            // Process sampled tokens
            batch.clear();
            for &(beam_idx, new_token) in &sampled {
                if self.is_eos_token(new_token, eos_token_id, model_eos) {
                    beam_finished[beam_idx] = true;
                } else {
                    beam_tokens[beam_idx].push(new_token);
                    let pos = (input_len + beam_tokens[beam_idx].len() - 1) as i32;
                    batch
                        .add(new_token, pos, &[beam_idx as i32], true)
                        .map_err(inference_err)?;
                }
            }

            // Decode all active beams at once
            if batch.n_tokens() == 0 {
                break;
            }
            ctx.decode(&mut batch).map_err(inference_err)?;
        }

        Ok(beam_tokens.into_iter().zip(beam_scores).collect())
    }

    /// True beam search: tracks cumulative probabilities at every step and
    /// keeps the globally best `beam_size` candidates, highest first.
    ///
    /// Runs on a single context with a reused KV cache — the prompt is
    /// decoded once into slot 0 and shared via cache copies, each live beam
    /// owns one sequence slot, and a step decodes only one new token per
    /// beam. Must return the same candidates as
    /// [`Self::generate_beam_search_full_eval`], the re-prefilling reference
    /// the equivalence test compares against.
    ///
    /// KV-cache-reuse formulation contributed by
    /// [kazuph/karukan@707eb10](https://github.com/kazuph/karukan/commit/707eb101f5de7b210f993b74723cf0ba9cc8c2af).
    pub fn generate_beam_search(
        &self,
        input_tokens: &[LlamaToken],
        max_new_tokens: usize,
        eos_token_id: Option<i32>,
        beam_size: usize,
    ) -> Result<Vec<(Vec<LlamaToken>, f32)>> {
        // max_new_tokens == 0 means "generate nothing": return before paying
        // for the prompt decode instead of emitting one-token pseudo-beams.
        if beam_size == 0 || max_new_tokens == 0 || input_tokens.is_empty() {
            return Ok(Vec::new());
        }
        let beam_size = beam_size.min(MAX_BEAM_SIZE);
        let model_eos = self.model.token_eos();
        let input_len = input_tokens.len();

        // Sequence slots 0..beam_size hold the live beams. The scratch slots
        // above them stage a permutation: several new beams can descend from
        // one parent, and a parent slot is itself a destination, so copying
        // straight into the live slots would clobber a prefix still needed.
        let n_seq = beam_size * 2;
        // Cells hold the prompt (shared by every beam) plus one row of
        // generated tokens per beam. Doubled to cover the scratch staging, in
        // case the backend materializes a copy rather than sharing cells.
        let n_cells = input_len + 2 * beam_size * (max_new_tokens + 1) + KV_HEADROOM_CELLS;
        // n_batch also caps the context's output buffer (n_outputs_max), which
        // llama.cpp reserves for n_seq_max rows — so it must cover n_seq too.
        let batch_cap = input_len.max(n_seq) + BATCH_HEADROOM;

        let mut ctx = self.new_context(
            self.context_params_with_n_ctx(n_cells as u32)
                .with_n_seq_max(n_seq.try_into().unwrap_or(u32::MAX))
                .with_n_batch(batch_cap as u32)
                .with_n_ubatch(batch_cap as u32),
        )?;

        // Step 1: decode the prompt once into slot 0 and read its logits.
        let mut batch = LlamaBatch::new(batch_cap, 1);
        self.add_input_tokens(&mut batch, input_tokens)?;
        ctx.decode(&mut batch).map_err(inference_err)?;

        let top = self.get_top_k_tokens(ctx.get_logits(), beam_size);
        let (mut beams, mut finished_beams) =
            self.partition_initial_beams(top, eos_token_id, model_eos);

        // Every surviving beam starts from the same prompt prefix.
        for slot in 1..beams.len() {
            ctx.copy_kv_cache_seq(0, slot as i32, None, None)
                .map_err(inference_err)?;
        }

        // Expansion factor
        let expand_k = beam_size.max(4);
        let mut candidates: Vec<(usize, LlamaToken, f32)> =
            Vec::with_capacity(beam_size * expand_k);
        let mut next: Vec<(usize, BeamState)> = Vec::with_capacity(beam_size);

        // Step 2: Main beam search loop
        for _step in 0..max_new_tokens.saturating_sub(1) {
            if beams.is_empty() || Self::beam_search_converged(&finished_beams, &beams, beam_size) {
                break;
            }

            // Decode one token per beam: the token chosen last step, appended
            // to that beam's cached prefix. Batch row `i` is beam slot `i`, so
            // the logits row index below is the slot index.
            batch.clear();
            for (slot, beam) in beams.iter().enumerate() {
                let last = *beam
                    .tokens
                    .last()
                    .expect("a live beam always has at least one token");
                let pos = (input_len + beam.tokens.len() - 1) as i32;
                batch
                    .add(last, pos, &[slot as i32], true)
                    .map_err(inference_err)?;
            }
            ctx.decode(&mut batch).map_err(inference_err)?;

            // Collect (parent slot, token, score) for every expansion — the
            // slot lets the cache follow the selection, and the token Vecs
            // are materialized only for the survivors below.
            candidates.clear();
            for (slot, beam) in beams.iter().enumerate() {
                let logits = ctx.get_logits_ith(slot as i32);
                for (token, log_prob) in self.get_top_k_tokens(logits, expand_k) {
                    candidates.push((slot, token, beam.score + log_prob));
                }
            }

            // Sort and keep top beam_size candidates
            candidates.sort_by(|a, b| b.2.total_cmp(&a.2));
            candidates.truncate(beam_size);

            // Partition into finished and active beams
            next.clear();
            for &(parent, token, score) in &candidates {
                let mut tokens = beams[parent].tokens.clone();
                tokens.push(token);
                let candidate = BeamState { tokens, score };
                if self.is_eos_token(token, eos_token_id, model_eos) {
                    finished_beams.push(candidate);
                } else {
                    next.push((parent, candidate));
                }
            }

            let parents: Vec<usize> = next.iter().map(|(parent, _)| *parent).collect();
            Self::permute_beam_kv(&mut ctx, &parents, beam_size)?;

            beams = next.drain(..).map(|(_, beam)| beam).collect();
        }

        Ok(Self::finalize_beams(finished_beams, beams, beam_size))
    }

    /// Reference beam search: a fresh context and a full re-prefill per beam per
    /// step. This is what [`Self::generate_beam_search`] replaced; it is kept so
    /// the equivalence test can assert the fast path still selects the same
    /// candidates. Far too slow for the input path — do not call it there.
    #[cfg(test)]
    pub(crate) fn generate_beam_search_full_eval(
        &self,
        input_tokens: &[LlamaToken],
        max_new_tokens: usize,
        eos_token_id: Option<i32>,
        beam_size: usize,
    ) -> Result<Vec<(Vec<LlamaToken>, f32)>> {
        // Same degenerate-input contract as the fast path.
        if beam_size == 0 || max_new_tokens == 0 || input_tokens.is_empty() {
            return Ok(Vec::new());
        }
        let model_eos = self.model.token_eos();

        let initial_logits = self.eval_sequence(input_tokens)?;
        let top = self.get_top_k_tokens(&initial_logits, beam_size);
        let (mut beams, mut finished_beams) =
            self.partition_initial_beams(top, eos_token_id, model_eos);

        let expand_k = beam_size.max(4);

        for _step in 0..max_new_tokens.saturating_sub(1) {
            if beams.is_empty() || Self::beam_search_converged(&finished_beams, &beams, beam_size) {
                break;
            }

            let mut candidates: Vec<BeamState> = Vec::new();

            for beam in &beams {
                // The whole point of the reference: re-prefill the full
                // sequence in a fresh context instead of reusing a KV cache.
                let mut full_seq: Vec<LlamaToken> = input_tokens.to_vec();
                full_seq.extend(&beam.tokens);

                let logits = self.eval_sequence(&full_seq)?;
                for (token, log_prob) in self.get_top_k_tokens(&logits, expand_k) {
                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(token);

                    candidates.push(BeamState {
                        tokens: new_tokens,
                        score: beam.score + log_prob,
                    });
                }
            }

            candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
            candidates.truncate(beam_size);

            beams.clear();
            for candidate in candidates {
                let Some(&last_token) = candidate.tokens.last() else {
                    continue;
                };
                if self.is_eos_token(last_token, eos_token_id, model_eos) {
                    finished_beams.push(candidate);
                } else {
                    beams.push(candidate);
                }
            }
        }

        Ok(Self::finalize_beams(finished_beams, beams, beam_size))
    }

    /// Add input tokens to a batch as a single sequence, requesting logits
    /// only for the last token.
    fn add_input_tokens(&self, batch: &mut LlamaBatch, tokens: &[LlamaToken]) -> Result<()> {
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(inference_err)?;
        }
        Ok(())
    }

    /// Process a token sequence and return the logits at the last position.
    ///
    /// Creates a fresh context for each call, so it costs a full prefill every
    /// time. Only the reference beam search
    /// ([`Self::generate_beam_search_full_eval`]) still works this way.
    #[cfg(test)]
    fn eval_sequence(&self, tokens: &[LlamaToken]) -> Result<Vec<f32>> {
        let mut ctx = self.new_context(self.context_params())?;

        let mut batch = LlamaBatch::new(512, 1);
        self.add_input_tokens(&mut batch, tokens)?;
        ctx.decode(&mut batch).map_err(inference_err)?;

        Ok(ctx.get_logits().to_vec())
    }

    /// Top-k tokens with their log probabilities, best first.
    fn get_top_k_tokens(&self, logits: &[f32], k: usize) -> Vec<(LlamaToken, f32)> {
        if k == 0 || logits.is_empty() {
            return Vec::new();
        }
        // log-softmax normalizer. Subtracting a constant never changes the
        // ranking, so top-k selection runs on the raw logits and only the k
        // winners are normalized.
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = logits
            .iter()
            .map(|&x| (x - max_logit).exp())
            .sum::<f32>()
            .ln()
            + max_logit;

        // Single-pass top-k (k is the beam width, a handful): allocating and
        // fully sorting a vocab-sized vector here ran per beam per step and
        // dominated the beam-search loop. Ties keep the lower token id first,
        // matching the stable full sort this replaces.
        let k = k.min(logits.len());
        let mut top: Vec<(usize, f32)> = Vec::with_capacity(k + 1);
        for (i, &x) in logits.iter().enumerate() {
            if top.len() == k && x <= top[k - 1].1 {
                continue;
            }
            let pos = top.partition_point(|&(_, s)| s >= x);
            top.insert(pos, (i, x));
            top.truncate(k);
        }

        top.into_iter()
            .map(|(i, x)| (LlamaToken(i as i32), x - log_sum_exp))
            .collect()
    }

    /// Seed beams from the prompt's top-k tokens, routing tokens that are
    /// already EOS into the finished set. Returns (active, finished).
    fn partition_initial_beams(
        &self,
        top: Vec<(LlamaToken, f32)>,
        eos_token_id: Option<i32>,
        model_eos: LlamaToken,
    ) -> (Vec<BeamState>, Vec<BeamState>) {
        let mut active = Vec::with_capacity(top.len());
        let mut finished = Vec::new();
        for (token, log_prob) in top {
            let beam = BeamState {
                tokens: vec![token],
                score: log_prob,
            };
            if self.is_eos_token(token, eos_token_id, model_eos) {
                finished.push(beam);
            } else {
                active.push(beam);
            }
        }
        (active, finished)
    }

    /// Whether the search can stop: enough beams have finished and none of
    /// the active ones can still outscore the best finished beam (scores only
    /// decrease as tokens are appended).
    fn beam_search_converged(
        finished: &[BeamState],
        active: &[BeamState],
        beam_size: usize,
    ) -> bool {
        if finished.len() < beam_size {
            return false;
        }
        let best = |beams: &[BeamState]| {
            beams
                .iter()
                .map(|b| b.score)
                .fold(f32::NEG_INFINITY, f32::max)
        };
        best(active) < best(finished)
    }

    /// Final candidate list: every beam, best score first, capped at
    /// `beam_size`.
    fn finalize_beams(
        finished: Vec<BeamState>,
        active: Vec<BeamState>,
        beam_size: usize,
    ) -> Vec<(Vec<LlamaToken>, f32)> {
        let mut all: Vec<(Vec<LlamaToken>, f32)> = finished
            .into_iter()
            .chain(active)
            .map(|b| (b.tokens, b.score))
            .collect();
        all.sort_by(|a, b| b.1.total_cmp(&a.1));
        all.truncate(beam_size);
        all
    }

    /// Move each surviving beam's KV prefix into its new slot. Several beams
    /// can descend from one parent and a parent slot is itself a destination,
    /// so the copies stage through scratch slots (`beam_size..`) instead of
    /// writing straight into the live slots and clobbering a prefix still in
    /// use. Clearing every live slot in between also reclaims the cells of
    /// beams that just finished.
    fn permute_beam_kv(ctx: &mut LlamaContext, parents: &[usize], beam_size: usize) -> Result<()> {
        for (j, &parent) in parents.iter().enumerate() {
            let scratch = (beam_size + j) as i32;
            ctx.clear_kv_cache_seq(Some(scratch as u32), None, None)
                .map_err(inference_err)?;
            ctx.copy_kv_cache_seq(parent as i32, scratch, None, None)
                .map_err(inference_err)?;
        }
        for slot in 0..beam_size {
            ctx.clear_kv_cache_seq(Some(slot as u32), None, None)
                .map_err(inference_err)?;
        }
        for j in 0..parents.len() {
            let scratch = (beam_size + j) as i32;
            ctx.copy_kv_cache_seq(scratch, j as i32, None, None)
                .map_err(inference_err)?;
            ctx.clear_kv_cache_seq(Some(scratch as u32), None, None)
                .map_err(inference_err)?;
        }
        Ok(())
    }

    /// Check if a token is an EOS token.
    ///
    /// Uses the model's own EOS/EOG metadata rather than hardcoded token IDs.
    fn is_eos_token(
        &self,
        token: LlamaToken,
        eos_token_id: Option<i32>,
        model_eos: LlamaToken,
    ) -> bool {
        eos_token_id.is_some_and(|eos| token.0 == eos)
            || token == model_eos
            || self.model.is_eog_token(token)
    }

    /// Generate tokens with a custom sampler
    fn generate_with_sampler(
        &self,
        input_tokens: &[LlamaToken],
        max_new_tokens: usize,
        eos_token_id: Option<i32>,
        mut sampler: LlamaSampler,
    ) -> Result<Vec<LlamaToken>> {
        // Size the batch to the prompt instead of inheriting llama.cpp's
        // defaults (n_batch 2048 / n_ubatch 512). The compute buffers are
        // allocated from those numbers on every `new_context`, and a
        // conversion only ever decodes the prompt plus one token at a time —
        // paying for a 512-token ubatch made each conversion allocate (and
        // zero) an order of magnitude more scratch memory than it uses.
        let batch_cap = input_tokens.len().max(1) + BATCH_HEADROOM;
        let mut ctx = self.new_context(
            self.context_params()
                .with_n_batch(batch_cap as u32)
                .with_n_ubatch(batch_cap as u32),
        )?;

        let mut batch = LlamaBatch::new(batch_cap, 1);
        let mut generated = input_tokens.to_vec();

        // Process input tokens
        self.add_input_tokens(&mut batch, input_tokens)?;

        ctx.decode(&mut batch).map_err(inference_err)?;

        // Get model's EOS token for comparison
        let model_eos = self.model.token_eos();

        // Generate new tokens
        for n_cur in (input_tokens.len()..).take(max_new_tokens) {
            let new_token = sampler.sample(&ctx, -1);

            // Check for EOS
            if self.is_eos_token(new_token, eos_token_id, model_eos) {
                break;
            }

            generated.push(new_token);

            // Prepare next batch with just the new token
            batch.clear();
            batch
                .add(new_token, n_cur as i32, &[0], true)
                .map_err(inference_err)?;

            ctx.decode(&mut batch).map_err(inference_err)?;
        }

        Ok(generated)
    }

    /// Get the EOS token ID from the model
    pub fn eos_token_id(&self) -> LlamaToken {
        self.model.token_eos()
    }
}

/// Reusable NLL scorer that keeps a single llama.cpp context alive.
///
/// Creating a `LlamaContext` is expensive. This struct amortizes the cost by
/// creating one context and clearing the KV cache between calls.
/// Use one `NllScorer` per thread for parallel scoring.
pub struct NllScorer<'a> {
    model: &'a LlamaCppModel,
    ctx: llama_cpp_2::context::LlamaContext<'a>,
    vocab_size: usize,
}

impl<'a> NllScorer<'a> {
    /// Create a new NLL scorer with a reusable context.
    pub fn new(model: &'a LlamaCppModel, n_ctx: u32) -> Result<Self> {
        // Via the model's own params builder so the configured n_threads
        // applies here too.
        let ctx = model.new_context(model.context_params_with_n_ctx(n_ctx))?;
        let vocab_size = model.model.n_vocab() as usize;

        Ok(Self {
            model,
            ctx,
            vocab_size,
        })
    }

    /// Compute per-character NLL for a single (reading, surface) pair.
    ///
    /// Reuses the internal context by clearing the KV cache between calls.
    pub fn compute_nll(&mut self, reading_katakana: &str, surface: &str) -> Result<f32> {
        let prompt = super::build_jinen_prompt(reading_katakana, "");
        let full_text = format!("{}{}", prompt, crate::kana::normalize_nfkc(surface));

        let prompt_tokens = self.model.tokenize(&prompt)?;
        let full_tokens = self.model.tokenize(&full_text)?;

        if full_tokens.len() <= prompt_tokens.len() {
            return Ok(100.0);
        }

        let n_tokens = full_tokens.len();

        self.ctx.clear_kv_cache();

        let mut batch = LlamaBatch::new(n_tokens.max(512), 1);
        batch
            .add_sequence(&full_tokens, 0, true)
            .map_err(inference_err)?;

        self.ctx.decode(&mut batch).map_err(inference_err)?;

        let start_pos = prompt_tokens.len() - 1;
        let end_pos = n_tokens - 1;
        let mut total_nll: f32 = 0.0;
        let mut n_scored = 0;

        for pos in start_pos..end_pos {
            let logits = self.ctx.get_logits_ith(pos as i32);

            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp: f32 = logits
                .iter()
                .take(self.vocab_size)
                .map(|&x| (x - max_logit).exp())
                .sum::<f32>()
                .ln()
                + max_logit;

            let target = full_tokens[pos + 1].0 as usize;
            if target < self.vocab_size {
                total_nll -= logits[target] - log_sum_exp;
            }
            n_scored += 1;
        }

        if n_scored == 0 {
            return Ok(100.0);
        }

        let n_chars = surface.chars().count().max(1);
        Ok(total_nll / n_chars as f32)
    }
}

#[cfg(test)]
mod missing_file_tests {
    use super::*;

    #[test]
    fn missing_model_file_is_err_not_panic() {
        let result = LlamaCppModel::from_file("/nonexistent/model.gguf", "/nonexistent/tok.json");
        assert!(matches!(result, Err(KanjiError::ModelLoad(_))));
    }
}

#[cfg(test)]
mod byte_fallback_token_tests {
    use super::is_byte_fallback_token;

    #[test]
    fn byte_fallback_tokens_are_recognized() {
        assert!(is_byte_fallback_token("<0x00>"));
        assert!(is_byte_fallback_token("<0xCE>"));
        assert!(is_byte_fallback_token("<0xff>"));
    }

    #[test]
    fn control_tokens_are_not_byte_fallback() {
        assert!(!is_byte_fallback_token("<s>"));
        assert!(!is_byte_fallback_token("</s>"));
        assert!(!is_byte_fallback_token("<unk>"));
        assert!(!is_byte_fallback_token("<pad>"));
        assert!(!is_byte_fallback_token("<0xGG>"));
        assert!(!is_byte_fallback_token("<0x123>"));
        assert!(!is_byte_fallback_token("0xCE"));
    }
}

#[cfg(test)]
mod beam_search_tests {
    use super::*;
    use crate::kanji::build_jinen_prompt;
    use crate::kanji::hf_download::{get_path_by_id, get_tokenizer_path_by_id};
    use crate::kanji::model_config::registry;

    /// Load the default registry model, or `None` when it isn't available
    /// locally (the tests are skipped rather than failing offline).
    fn load_model() -> Option<LlamaCppModel> {
        let reg = registry();
        let path = get_path_by_id(&reg.default_model).ok()?;
        let tok_path = get_tokenizer_path_by_id(&reg.default_model).ok()?;
        LlamaCppModel::from_file(&path, &tok_path).ok()
    }

    fn tokens_for(model: &LlamaCppModel, katakana: &str) -> Vec<LlamaToken> {
        let prompt = build_jinen_prompt(katakana, "");
        model.tokenize(&prompt).expect("tokenize failed")
    }

    /// The KV-cache-reusing beam search must pick the same candidates as the
    /// reference that re-prefills a fresh context for every beam at every step.
    /// This is the guard that makes the optimization a pure speedup: if the
    /// cache bookkeeping (slot assignment, parent permutation, positions) were
    /// wrong, the beams would attend to the wrong prefix and diverge here.
    #[test]
    fn matches_full_eval_reference() {
        let Some(model) = load_model() else {
            eprintln!("model unavailable, skipping");
            return;
        };
        let eos = Some(model.eos_token_id().0);

        for katakana in ["ヘンカン", "カンジ", "キョウハイイテンキデスネ", "コウエン"]
        {
            let tokens = tokens_for(&model, katakana);
            for beam_size in [1usize, 2, 3, 5] {
                let fast = model
                    .generate_beam_search(&tokens, 20, eos, beam_size)
                    .expect("fast beam search failed");
                let reference = model
                    .generate_beam_search_full_eval(&tokens, 20, eos, beam_size)
                    .expect("reference beam search failed");

                let decode = |results: &[(Vec<LlamaToken>, f32)]| -> Vec<String> {
                    results
                        .iter()
                        .map(|(t, _)| model.decode(t, true).unwrap_or_default())
                        .collect()
                };
                assert_eq!(
                    decode(&fast),
                    decode(&reference),
                    "candidates diverged for {katakana} at beam_size={beam_size}"
                );
            }
        }
    }

    /// A beam wider than the surviving beam count, and a beam of 1, both used
    /// to sit on `max_new_tokens - 1` underflow / empty-slot edges.
    #[test]
    fn handles_edge_case_widths() {
        let Some(model) = load_model() else {
            eprintln!("model unavailable, skipping");
            return;
        };
        let eos = Some(model.eos_token_id().0);
        let tokens = tokens_for(&model, "ヘンカン");

        assert!(
            model
                .generate_beam_search(&tokens, 0, eos, 3)
                .expect("max_new_tokens=0 must not panic")
                .is_empty(),
            "max_new_tokens=0 must generate nothing"
        );
        assert!(
            !model
                .generate_beam_search(&tokens, 20, eos, 9)
                .expect("wide beam failed")
                .is_empty()
        );
        assert!(
            model
                .generate_beam_search(&[], 20, eos, 3)
                .expect("empty input must not panic")
                .is_empty()
        );
        // beam_size is clamped to 128: 2*200 sequence slots would exceed
        // llama.cpp's LLAMA_MAX_SEQ (256) and abort across the FFI boundary.
        assert!(
            !model
                .generate_beam_search(&tokens, 3, eos, 200)
                .expect("oversized beam must be clamped, not abort")
                .is_empty()
        );
    }
}

#[cfg(test)]
mod kv_override_tests {
    use super::kv_override_str;

    #[test]
    fn packs_a_short_name_and_leaves_the_rest_zeroed() {
        let buf = kv_override_str("gpt2");
        assert_eq!(&buf[..4], b"gpt2".map(|b| b as std::os::raw::c_char));
        assert!(buf[4..].iter().all(|&b| b == 0), "must stay NUL-filled");
    }

    #[test]
    fn truncation_keeps_the_terminator() {
        let buf = kv_override_str(&"x".repeat(200));
        assert_eq!(buf[126], b'x' as std::os::raw::c_char);
        assert_eq!(buf[127], 0, "last byte must remain the NUL terminator");
    }
}

#[cfg(test)]
mod kv_reuse_tests {
    use super::*;
    use crate::kanji::build_jinen_prompt;
    use crate::kanji::hf_download::{get_path_by_id, get_tokenizer_path_by_id};
    use crate::kanji::model_config::registry;

    #[test]
    fn common_prefix_len_basics() {
        let t = |v: &[i32]| v.iter().map(|&x| LlamaToken(x)).collect::<Vec<_>>();
        assert_eq!(common_prefix_len(&t(&[1, 2, 3]), &t(&[1, 2, 4])), 2);
        assert_eq!(common_prefix_len(&t(&[]), &t(&[1])), 0);
        assert_eq!(common_prefix_len(&t(&[1, 2]), &t(&[1, 2])), 2);
        assert_eq!(common_prefix_len(&t(&[1, 2, 3]), &t(&[1, 2])), 2);
        assert_eq!(common_prefix_len(&t(&[9]), &t(&[1, 2])), 0);
    }

    /// Load the default registry model, or `None` when it isn't available
    /// locally (the tests are skipped rather than failing offline).
    fn load_model() -> Option<LlamaCppModel> {
        let reg = registry();
        let path = get_path_by_id(&reg.default_model).ok()?;
        let tok_path = get_tokenizer_path_by_id(&reg.default_model).ok()?;
        LlamaCppModel::from_file(&path, &tok_path).ok()
    }

    /// The session must produce exactly what a fresh context produces, over
    /// the call sequences typing actually generates: prompts growing one
    /// kana at a time, backspace to a shorter prefix, an exact repeat, a
    /// switch to an unrelated prompt, and a left-context change (which
    /// flips the very front of the prompt).
    #[test]
    fn matches_fresh_context_across_call_sequences() {
        let Some(model) = load_model() else {
            eprintln!("model unavailable, skipping");
            return;
        };
        let eos = Some(model.eos_token_id().0);

        let cases: [(&str, &str); 9] = [
            ("ワ", ""),
            ("ワタ", ""),
            ("ワタシ", ""),
            ("ワタシハ", ""),
            ("ワタシハガクセイ", ""),
            ("ワタシ", ""),             // backspace
            ("ワタシ", ""),             // repeat
            ("キョウハイイテンキ", ""), // unrelated switch
            ("ハシ", "箸を持つ。"),     // context change
        ];
        for (katakana, context) in cases {
            let prompt = build_jinen_prompt(katakana, context);
            let tokens = model.tokenize(&prompt).expect("tokenize failed");
            let session = model.generate(&tokens, 20, eos).expect("generate failed");
            let fresh = model
                .generate_with_sampler(&tokens, 20, eos, LlamaSampler::greedy())
                .expect("fresh generate failed");
            assert_eq!(
                session, fresh,
                "diverged for 「{katakana}」 ctx 「{context}」"
            );
        }
    }
}
