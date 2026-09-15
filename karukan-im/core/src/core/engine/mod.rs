//! IME Engine - the core state machine and input processing
//!
//! This module contains the main `InputMethodEngine` struct that coordinates between
//! the romaji converter, kanji converter, and manages the IME state.

mod cache;
mod chunk;
mod conversion;
mod cursor;
mod display;
mod filter;
mod init;
mod input;
mod input_buffer;
mod mode;
mod model;
mod strategy;
mod types;

pub use types::*;

use std::collections::HashSet;

use cache::{ConversionCache, ConversionCacheKey, ModelRole};
use input_buffer::InputBuffer;

#[cfg(test)]
mod tests;

use karukan_engine::{
    DateRewriter, Dictionary, EmojiRewriter, KanaKanjiConverter, LearningCache, LearningConfig,
    RewriteOutput, Rewriter, RewriterChain, RomajiConverter,
};
use tracing::{debug, trace};

use super::candidate::{Candidate, CandidateList, CandidateSource};
use super::keycode::{KeyEvent, Keysym};
use super::preedit::Preedit;
use super::state::InputState;
use crate::config::settings::{Settings, SpaceStyle};

/// A conversion candidate tagged with its source and an optional description.
///
/// Built up internally during candidate construction; later mapped onto the
/// public `Candidate`, which carries the `source` itself and derives its
/// presentation (aux label, deletability) from it on read.
#[derive(Debug, Clone)]
struct AnnotatedCandidate {
    text: String,
    source: CandidateSource,
    /// Override reading (e.g. from prefix_lookup where the full reading differs from input)
    reading: Option<String>,
    /// Per-candidate description (e.g. `三点リーダ` for `…`,
    /// `[全]英大文字` for `ＡＢＣ`). Surfaced as the mozc-style right-side
    /// comment on the candidate; never contains a source label.
    description: Option<String>,
}

impl AnnotatedCandidate {
    fn new(text: impl Into<String>, source: CandidateSource) -> Self {
        Self {
            text: text.into(),
            source,
            reading: None,
            description: None,
        }
    }

    fn with_reading(mut self, reading: Option<String>) -> Self {
        self.reading = reading;
        self
    }

    fn with_description(mut self, description: Option<String>) -> Self {
        self.description = description;
        self
    }

    /// Into the public [`Candidate`], falling back to `reading` for
    /// candidates that don't carry one of their own (predictive results do).
    /// The source rides along; its presentation is derived on read.
    fn into_candidate(self, reading: &str) -> Candidate {
        Candidate {
            text: self.text,
            reading: Some(self.reading.unwrap_or_else(|| reading.to_string())),
            source: Some(self.source),
            description: self.description,
        }
    }
}

/// Resolve a model variant id from settings.
///
/// - `model` is None or empty → default variant from registry
/// - `model` matches a known variant id → that variant
/// - otherwise → error (unknown variant)
pub fn resolve_variant_id(model: Option<&str>) -> anyhow::Result<String> {
    let reg = karukan_engine::kanji::registry();
    match model {
        Some(id) if !id.is_empty() => {
            if reg.find_variant(id).is_some() {
                Ok(id.to_string())
            } else {
                anyhow::bail!("unknown model variant: {}", id)
            }
        }
        _ => Ok(reg.default_model.clone()),
    }
}

/// Keep at most the last `n` characters of `s`.
fn keep_last_chars(s: &str, n: usize) -> String {
    let char_count = s.chars().count();
    if char_count > n {
        s.chars().skip(char_count - n).collect()
    } else {
        s.to_string()
    }
}

/// Keep at most the first `n` characters of `s`.
fn keep_first_chars(s: &str, n: usize) -> String {
    let char_count = s.chars().count();
    if char_count > n {
        s.chars().take(n).collect()
    } else {
        s.to_string()
    }
}

/// The main IME engine
pub struct InputMethodEngine {
    /// Current input state
    state: InputState,
    /// Converters (romaji, kanji, light kanji)
    converters: Converters,
    /// Surrounding text context from the editor (text around cursor)
    surrounding_context: Option<SurroundingContext>,
    /// Engine configuration
    config: EngineConfig,
    /// Conversion timing and adaptive model metrics
    metrics: ConversionMetrics,
    /// Current input mode plus the mode to come back to when a temporary
    /// mode (Emoji, Alphabet) ends — see [`ModeState`]
    mode: ModeState,
    /// Composition record: per-display-char elements plus the caret
    input_buf: InputBuffer,
    /// Live conversion state
    live: LiveConversion,
    /// Internal chunking of the composing buffer built by
    /// `chunked_auto_suggest`: the current per-chunk conversions, rebuilt from
    /// scratch on every keystroke (per-chunk model calls are deduplicated by
    /// `conversion_cache`, so unchanged chunks cost a lookup, not an
    /// inference). Empty when not composing.
    chunks: Vec<ComposingChunk>,
    /// Manual chunk boundaries (reading char positions) inserted with Ctrl+J.
    /// Shifted along with edits (`edit_with_chunk_breaks`) and cleared when
    /// the composition ends.
    chunk_breaks: Vec<usize>,
    /// LRU cache of model conversion results keyed by reading + lctx +
    /// strategy. Content-addressed, so it survives commits and resets.
    conversion_cache: ConversionCache,
    /// Suppresses the auto-suggest model call while a conversion detours
    /// through the composing path and throws the render away. Owned by
    /// `in_composing`, which sets and clears it around the one call.
    suppress_suggest: bool,
    /// Mirror of the suggestion window shown while composing — what Ctrl+digit
    /// selects. The Conversion state owns its own list; this covers the
    /// Composing state, which renders candidates without holding them.
    shown_suggestions: CandidateList,
    /// Dictionaries (system, user)
    dicts: Dictionaries,
    /// Learning cache (user conversion history)
    learning: Option<LearningCache>,
    /// Receiver for the background model-loading thread: model resolution
    /// can block on the network, so it never runs on the key-event thread.
    /// Drained by `poll_loaded_models` at the top of `process_key`; until
    /// then (or if loading failed) the engine runs dictionary/kana-only.
    model_loading: Option<std::sync::mpsc::Receiver<init::LoadedConverters>>,
}

impl InputMethodEngine {
    /// Create a new IME engine
    pub fn new() -> Self {
        Self {
            state: InputState::Empty,
            converters: Converters {
                romaji: RomajiConverter::new(),
                kanji: None,
                light_kanji: None,
                rewriters: RewriterChain::default_chain(),
                date: DateRewriter::new(karukan_engine::DateConfig::default()),
            },
            surrounding_context: None,
            config: EngineConfig::default(),
            metrics: ConversionMetrics::default(),
            mode: ModeState::default(),
            input_buf: InputBuffer::new(),
            live: LiveConversion::default(),
            chunks: Vec::new(),
            chunk_breaks: Vec::new(),
            conversion_cache: ConversionCache::default(),
            suppress_suggest: false,
            shown_suggestions: CandidateList::default(),
            dicts: Dictionaries::default(),
            learning: None,
            model_loading: None,
        }
    }

    /// Create with configuration
    pub fn with_config(config: EngineConfig) -> Self {
        let mut engine = Self {
            live: LiveConversion::new(config.live_conversion),
            ..Self::new()
        };
        // The symbol style is baked into the rule trie, so the converter is
        // rebuilt rather than configured after the fact. It carries the
        // width rules too: a keystroke settles at the width in force when
        // it was typed.
        engine.converters.romaji = RomajiConverter::with_rules(config.symbol, config.width);
        engine.converters.date = DateRewriter::new(config.date.clone());
        engine.config = config;
        engine
    }

    /// Conversion (inference) time of the last `process_key` /
    /// `select_candidate_on_page` call in milliseconds; 0 when that call
    /// ran no conversion.
    pub fn last_conversion_ms(&self) -> u64 {
        self.metrics.conversion_ms
    }

    /// Get last process_key time in milliseconds (input to result, end-to-end)
    pub fn last_process_key_ms(&self) -> u64 {
        self.metrics.process_key_ms
    }

    /// Get the model name being used
    pub fn model_name(&self) -> String {
        let main = self
            .converters
            .kanji
            .as_ref()
            .map(|c| c.model_display_name());
        let sub = self
            .converters
            .light_kanji
            .as_ref()
            .map(|c| c.model_display_name());
        match (main, sub) {
            (Some(m), Some(s)) => format!("{}+{}", m, s),
            (Some(m), None) => m.to_string(),
            _ if self.model_loading.is_some() => "loading".to_string(),
            _ => "unknown".to_string(),
        }
    }

    /// Get the current state
    pub fn state(&self) -> &InputState {
        &self.state
    }

    /// Get the current preedit
    pub fn preedit(&self) -> Option<&Preedit> {
        self.state.preedit()
    }

    /// Get the current candidates
    pub fn candidates(&self) -> Option<&CandidateList> {
        self.state.candidates()
    }

    /// Candidates per page of the candidate window (layout-dependent).
    /// Bounds the `select_candidate` page index on the JSON-RPC server.
    pub fn candidate_page_size(&self) -> usize {
        self.config.candidate_page_size()
    }

    /// Columns of the grid candidate layout; `None` for the vertical list.
    /// Rides on every `ShowCandidates` so the frontend renders to match.
    pub fn candidate_grid_columns(&self) -> Option<usize> {
        self.config.candidate_grid_columns()
    }

    /// Reset the engine state
    /// Note: surrounding_context is intentionally NOT cleared here.
    /// It is set once at activate() time and should persist through
    /// the session. fcitx5 may send reset events between activate
    /// and the first keyEvent, which would wipe the context.
    pub fn reset(&mut self) {
        self.state = InputState::Empty;
        self.mode = ModeState::default();
        self.clear_composition();
        self.metrics = ConversionMetrics::default();
    }

    /// Drop all composition-scoped state in one place: the input buffer, the
    /// live-conversion display flag, the chunks, and the manual chunk breaks.
    /// Every path that ends (or freshly starts) a composition goes through
    /// here, so a new composition-scoped field is a one-line change.
    pub(super) fn clear_composition(&mut self) {
        self.input_buf.clear();
        self.live.shown = false;
        self.chunks.clear();
        self.chunk_breaks.clear();
        self.shown_suggestions = CandidateList::default();
    }

    /// End the composition: clear the buffer, live display, and chunks,
    /// return to Empty, and exit any temporary mode. Every commit/cancel/
    /// erase-to-empty path must go through here so no piece of the teardown
    /// is forgotten.
    fn end_composition(&mut self) {
        self.clear_composition();
        self.state = InputState::Empty;
        self.mode.exit_temporary();
    }

    /// The candidate list currently on screen: the conversion's own list, or
    /// the suggestion list shown while composing. One accessor so keys that
    /// act on "what the user is looking at" — digit selection — do not need
    /// to know which state produced it.
    fn shown_candidates_mut(&mut self) -> Option<&mut CandidateList> {
        match &mut self.state {
            InputState::Conversion { candidates, .. } => Some(candidates),
            InputState::Composing { .. } => Some(&mut self.shown_suggestions),
            InputState::Empty => None,
        }
    }

    /// Ctrl+1..9: commit the numbered candidate from the list on screen,
    /// whether it came from a conversion or from the composing suggestions.
    /// Consumed even with nothing to select, so the chord never leaks to the
    /// application.
    pub(super) fn select_shown_candidate(&mut self, digit: usize) -> EngineResult {
        let Some(candidates) = self.shown_candidates_mut() else {
            return EngineResult::not_consumed();
        };
        if candidates.select_on_page(digit).is_none() {
            return EngineResult::consumed();
        }
        let Some(selected) = candidates.selected() else {
            return EngineResult::consumed();
        };
        let text = selected.text.clone();
        let reading = selected.reading.clone();
        let source = selected.source;
        if text.is_empty() {
            return EngineResult::consumed();
        }

        // A suggestion always carries its reading; fall back to the buffer
        // so a candidate built without one still records under a key. A
        // non-learnable source (a date is stale tomorrow) records nothing.
        let reading = if source.is_none_or(|s| s.is_learnable()) {
            reading.or_else(|| Some(self.input_buf.reading()))
        } else {
            None
        };
        self.finish_conversion(&text, &reading);

        EngineResult::consumed()
            .with_action(EngineAction::Commit(text))
            .with_action(EngineAction::HideCandidates)
            .with_action(EngineAction::HideAuxText)
    }

    /// If the composition is empty, reset to Empty state and return the result.
    /// Returns None if elements remain (caller should continue normally).
    fn try_reset_if_empty(&mut self) -> Option<EngineResult> {
        if self.input_buf.is_empty() {
            self.end_composition();
            Some(
                EngineResult::consumed()
                    .with_action(EngineAction::UpdatePreedit(Preedit::new()))
                    .with_action(EngineAction::HideCandidates)
                    .with_action(EngineAction::HideAuxText),
            )
        } else {
            None
        }
    }

    /// Update state to Composing with the current preedit, returning it.
    /// Automatically uses live conversion display when `live_text()` is non-empty.
    fn set_composing_state(&mut self) -> Preedit {
        let preedit = self.build_composing_preedit();
        self.state = InputState::Composing {
            preedit: preedit.clone(),
        };
        preedit
    }

    /// Convert hiragana in input_buf to katakana permanently.
    /// Called when leaving Katakana mode so the preedit doesn't revert.
    fn bake_katakana(&mut self) {
        self.input_buf.bake_katakana();
    }

    /// Settle the pending romaji segment into the composed text at the cursor
    fn settle_romaji(&mut self) {
        self.edit_with_chunk_breaks(|e| e.input_buf.settle_romaji(&e.converters.romaji));
    }

    /// Set surrounding context from the full text plus a cursor offset in
    /// Unicode scalar values (the unit both fcitx5 and the JSON-RPC
    /// protocol deliver). Splits at the cursor and delegates to
    /// [`Self::set_surrounding_context`].
    pub fn set_surrounding_text_at(&mut self, text: &str, cursor_chars: usize) {
        let byte_offset = text
            .char_indices()
            .nth(cursor_chars)
            .map(|(i, _)| i)
            .unwrap_or(text.len());
        let (left, right) = text.split_at(byte_offset);
        self.set_surrounding_context(left, right);
    }

    /// Set both left and right context from surrounding text (from editor)
    /// left_context: text before cursor
    /// right_context: text after cursor
    pub fn set_surrounding_context(&mut self, left_context: &str, right_context: &str) {
        debug!(
            "set_surrounding_context: left=\"{}\" right=\"{}\"",
            left_context, right_context
        );

        // Strip to current line: left = text after last newline.
        // If cursor is right after a newline, left context is empty.
        let left_context = match left_context.rsplit_once('\n') {
            Some((_, after)) => after,
            None => left_context,
        };
        let right_context = right_context
            .split_once('\n')
            .map_or(right_context, |(before, _)| before);

        if left_context.is_empty() && right_context.is_empty() {
            self.surrounding_context = None;
            return;
        }

        // Truncate left context to max length (keep end)
        let left = if left_context.is_empty() {
            None
        } else {
            Some(keep_last_chars(left_context, self.config.context_chars))
        };

        // Truncate right context to max length (keep beginning)
        let right = if right_context.is_empty() {
            None
        } else {
            Some(keep_first_chars(right_context, self.config.context_chars))
        };

        self.surrounding_context = Some(SurroundingContext { left, right });
    }

    /// Handle mode toggle keys (Right Alt/Super/Meta/Hyper and the JIS 変換
    /// key): one-way non-Hiragana → Hiragana.
    /// Returns `Some(result)` if the key was handled, `None` if not a mode toggle key.
    fn handle_mode_toggle_key(&mut self, key: &KeyEvent) -> Option<EngineResult> {
        if !key.keysym.is_mode_toggle_key() {
            return None;
        }
        // 変換 is an ordinary key, not a modifier: a modified chord
        // (Ctrl+変換 etc.) may be an app or fcitx5 shortcut, so only the
        // bare press acts as the toggle. The right-modifier keysyms are
        // exempt — their events routinely carry their own modifier state.
        if key.keysym == Keysym::HENKAN && key.modifiers.any() {
            return None;
        }
        // While a conversion is in flight (candidate window open) the kana
        // modes cannot toggle: switching would katakana-bake the conversion
        // *reading* (not the preedit) and defeat the Emoji-mode learning
        // guard — the commit path checks the current mode to decide whether
        // the reading is safe to record in the kana-keyed learning cache.
        // Alphabet is exempt: it only says how the next keystroke is read,
        // and Shift+letter can enter it here (typing refines the reading
        // instead of committing), so this is the only way back out.
        if matches!(self.state, InputState::Conversion { .. })
            && self.mode.current() != InputMode::Alphabet
        {
            return Some(EngineResult::not_consumed());
        }
        // Only consume the key when actually switching; otherwise pass through
        // so the system can properly track modifier state.
        if key.is_press && self.mode.current() != InputMode::Hiragana {
            // Bake katakana before switching so preedit doesn't revert. No
            // settling otherwise — the mode switch must not touch the
            // elements, so live romaji (`ky` typed before an alphabet word)
            // still combines after coming back to kana mode.
            if self.mode.current() == InputMode::Katakana {
                self.settle_romaji();
                self.bake_katakana();
            }
            self.mode.set(InputMode::Hiragana);
            // An open candidate window keeps its own line, mode indicator
            // included: a composing line here would hide the source-filter
            // header mid-view.
            let aux = match &self.state {
                InputState::Conversion {
                    reading,
                    candidates,
                    ..
                } => self.format_aux_conversion(reading, candidates),
                _ => self.format_aux_composing(),
            };
            if matches!(self.state, InputState::Composing { .. }) {
                let preedit = self.set_composing_state();
                return Some(
                    EngineResult::consumed()
                        .with_action(EngineAction::UpdatePreedit(preedit))
                        .with_action(EngineAction::UpdateAuxText(aux)),
                );
            }
            return Some(EngineResult::consumed().with_action(EngineAction::UpdateAuxText(aux)));
        }
        Some(EngineResult::not_consumed())
    }

    /// Text the engine did not type, at the width its groups are
    /// configured for: the model's answers and the candidates built from
    /// them, the dictionaries, the learning cache.
    ///
    /// Typed text settles earlier, as each character leaves the romaji
    /// converter, so it keeps the width in force when it was typed —
    /// switching to alphabet input mid-word leaves 「（」 alone. This one is
    /// for text that arrives already converted: the model is prompted with
    /// NFKC and answers in half-width whatever was typed, so its output has
    /// to be settled on the way in or the setting would never survive a
    /// conversion.
    ///
    /// Never applied to a candidate the user picked — that is already the
    /// width they chose, and a second pass would fold `＜＞１２３４` back to
    /// `＜＞1234` on commit.
    fn settle_text(&self, text: &str) -> String {
        match self.mode.current() {
            InputMode::Alphabet | InputMode::Emoji => text.to_string(),
            InputMode::Hiragana | InputMode::Katakana => {
                // Spaces follow `[symbol] space` the way symbols follow the
                // width rules: NFKC flattens `　` to ` ` in the prompt, so
                // the model can only ever answer with the half-width one.
                let space = self.space_char();
                self.config
                    .width
                    .apply_str(text)
                    .chars()
                    .map(|c| if c.is_whitespace() { space } else { c })
                    .collect()
            }
        }
    }

    /// Build the candidate list a window shows and a selection indexes,
    /// with the model's answers settled at the configured width.
    ///
    /// Only the model's. Its width is an artifact — the prompt is
    /// NFKC-normalized, so the answer comes back half-width whatever was
    /// typed — while every other source carries a width someone chose: a
    /// dictionary surface is spelled the way its author spelled it
    /// (`Yahoo!` with a half-width `!`), learning replays what the user
    /// committed, the rewriter's variants *are* the width choice, and the
    /// kana fallbacks settled as they were typed.
    ///
    /// Folding creates duplicates (with full-width digits both `1` and the
    /// rewriter's `１` come out `１`) and only the first survives — dropped
    /// here rather than at display time, since this list is also what
    /// Ctrl+digit indexes and commit reads.
    fn settle_candidates(&self, candidates: Vec<Candidate>) -> CandidateList {
        let mut seen = HashSet::new();
        let settled = candidates
            .into_iter()
            .filter_map(|mut candidate| {
                if candidate.source == Some(CandidateSource::Model) {
                    candidate.text = self.settle_text(&candidate.text);
                }
                seen.insert(candidate.text.clone()).then_some(candidate)
            })
            .collect();
        CandidateList::with_page_size(settled, self.config.candidate_page_size())
    }

    /// Process a key event
    pub fn process_key(&mut self, key: &KeyEvent) -> EngineResult {
        // Install converters the background loader has finished; never blocks.
        self.poll_loaded_models();

        // Log modifier key events for debugging key mapping issues
        if key.keysym.is_modifier() {
            debug!(
                "modifier key: keysym=0x{:04x} press={} modifiers={:?}",
                key.keysym.0, key.is_press, key.modifiers
            );
        }

        // Right Alt/Super/Meta/Hyper: one-way non-Hiragana → Hiragana switch
        if let Some(result) = self.handle_mode_toggle_key(key) {
            return result;
        }

        // Modifier-only keys (Shift, Ctrl, Alt_L, Super_L, etc.): pass through
        if key.keysym.is_modifier() {
            return EngineResult::not_consumed();
        }

        // Only process key presses
        if !key.is_press {
            return EngineResult::not_consumed();
        }

        // Ctrl+Shift+L: toggle live conversion (works in all states)
        if key.modifiers.control_key
            && key.modifiers.shift_key
            && (key.keysym == Keysym::KEY_L || key.keysym == Keysym::KEY_L_UPPER)
        {
            return self.toggle_live_conversion();
        }

        // Ctrl+Shift+V: toggle the verbose aux line (works in all states)
        if key.modifiers.control_key
            && key.modifiers.shift_key
            && (key.keysym == Keysym::KEY_V || key.keysym == Keysym::KEY_V_UPPER)
        {
            return self.toggle_verbose();
        }

        // Reset adaptive model flag when starting a new word (first key in Empty state)
        if matches!(self.state, InputState::Empty) {
            self.metrics.adaptive_use_light_model = false;
        }

        // trace, not debug: this logs what the user types (keysyms), so it
        // must stay out of ordinary debug logging.
        trace!(
            "process_key: keysym=0x{:04x} modifiers={:?} state={}",
            key.keysym.0,
            key.modifiers,
            match &self.state {
                InputState::Empty => "Empty",
                InputState::Composing { .. } => "Composing",
                InputState::Conversion { .. } => "Conversion",
            }
        );

        let start = std::time::Instant::now();
        // conversion_ms reports this key only: 0 unless a conversion runs below
        self.metrics.conversion_ms = 0;

        let shift_active = key.modifiers.shift_key;

        let result = match &self.state {
            InputState::Empty => self.process_key_empty(key, shift_active),
            InputState::Composing { .. } => self.process_key_composing(key, shift_active),
            InputState::Conversion { .. } => self.process_key_conversion(key, shift_active),
        };

        self.metrics.process_key_ms = start.elapsed().as_millis() as u64;

        result
    }

    /// Commit any pending input and return the text. Shares the resolution
    /// and teardown with the Enter-commit paths, so a focus-out commit can
    /// never diverge from what Enter would have produced.
    pub fn commit(&mut self) -> String {
        let text = match &self.state {
            InputState::Empty => String::new(),
            InputState::Composing { .. } => {
                let (reading, text) = self.resolve_composing_commit();
                self.record_learning(&reading, &text);
                self.end_composition();
                text
            }
            InputState::Conversion { .. } => {
                let (text, reading) = self
                    .selected_conversion_info()
                    .expect("state is Conversion");
                self.finish_conversion(&text, &reading);
                text
            }
        };
        self.surrounding_context = None;
        text
    }

    /// Commit any pending input as an [`EngineResult`], emitting the same
    /// UI cleanup actions as the key-driven commit path (Enter), so
    /// frontends don't have to pair [`Self::commit`] with manual
    /// preedit/candidate-window teardown.
    pub fn commit_result(&mut self) -> EngineResult {
        let text = self.commit();
        let mut result = EngineResult::consumed();
        if !text.is_empty() {
            result = result.with_action(EngineAction::Commit(text));
        } else {
            result = result.with_action(EngineAction::UpdatePreedit(Preedit::new()));
        }
        result
            .with_action(EngineAction::HideCandidates)
            .with_action(EngineAction::HideAuxText)
    }

    /// Save the learning cache to disk if it has unsaved changes.
    pub fn save_learning(&mut self) {
        if let Some(cache) = &mut self.learning
            && cache.is_dirty()
            && let Some(path) = Settings::learning_file()
        {
            if let Err(e) = cache.save(&path) {
                debug!("Failed to save learning cache: {}", e);
            } else {
                debug!("Learning cache saved to {:?}", path);
            }
        }
    }
}

impl Default for InputMethodEngine {
    fn default() -> Self {
        Self::new()
    }
}
