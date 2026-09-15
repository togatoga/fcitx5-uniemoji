//! Type definitions for the IME engine

use karukan_engine::{
    DateConfig, DateRewriter, Dictionary, KanaKanjiConverter, RewriterChain, RomajiConverter,
    SymbolStyle, WidthRules,
};

use crate::config::settings::{CandidateLayout, SpaceStyle, StrategyMode};

use super::super::candidate::CandidateList;
use super::super::preedit::Preedit;

/// Action to be performed by the framework/UI layer
#[derive(Debug, Clone)]
pub enum EngineAction {
    /// Update the preedit display
    UpdatePreedit(Preedit),
    /// Show the candidate window with candidates
    ShowCandidates(CandidateList),
    /// Hide the candidate window
    HideCandidates,
    /// Commit text to the application
    Commit(String),
    /// Update auxiliary text (e.g., reading hint, mode indicator)
    UpdateAuxText(String),
    /// Hide auxiliary text
    HideAuxText,
}

/// Result of processing a key event
#[derive(Debug, Clone, Default)]
pub struct EngineResult {
    /// Whether the key was consumed by the IME
    pub consumed: bool,
    /// Actions to perform
    pub actions: Vec<EngineAction>,
}

impl EngineResult {
    pub fn consumed() -> Self {
        Self {
            consumed: true,
            actions: Vec::new(),
        }
    }

    pub fn not_consumed() -> Self {
        Self {
            consumed: false,
            actions: Vec::new(),
        }
    }

    pub fn with_action(mut self, action: EngineAction) -> Self {
        self.actions.push(action);
        self
    }
}

/// Surrounding text context from the editor (text around the cursor)
#[derive(Debug, Clone)]
pub(in crate::core) struct SurroundingContext {
    /// Text before the cursor (None if empty)
    pub left: Option<String>,
    /// Text after the cursor (None if empty)
    pub right: Option<String>,
}

/// Configuration for the IME engine
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Number of conversion candidates for explicit conversion (Space key)
    pub num_candidates: usize,
    /// Maximum context length to display
    pub display_context_chars: usize,
    /// Maximum context length for API calls (to avoid overflow)
    pub context_chars: usize,
    /// Maximum reading length (chars) converted by the model in a single call.
    /// The composing buffer is split into chunks of at most this many chars so
    /// live-conversion latency stays bounded for long input. See
    /// [`ComposingChunk`] and `chunked_auto_suggest`.
    pub chunk_chars: usize,
    /// Maximum non-Japanese chars (symbols/digits) a Japanese chunk absorbs
    /// in total during live conversion; the absorption rules live in
    /// `group_chunks`.
    pub chunk_symbols: usize,
    /// Digits a chunk containing Japanese keeps (0 = split at every run).
    pub chunk_digits: usize,
    /// Alphabet chars a chunk containing Japanese keeps (0 = split at every
    /// run, which also keeps the romaji tail out of the model).
    pub chunk_alphabets: usize,
    /// Chars the beam covers, snapped to chunk boundaries.
    pub beam_chars: usize,
    /// Beam width: how many alternatives the beam returns
    pub beam_width: usize,
    /// Maximum acceptable latency in milliseconds for auto-suggest (0 = disabled)
    /// When a main model conversion exceeds this, the engine adaptively switches to light_model
    pub max_latency_ms: u64,
    /// Conversion strategy mode (adaptive, light, main)
    pub strategy: StrategyMode,
    /// Show the detailed aux line (Ctrl+Shift+V toggles it).
    pub verbose: bool,
    /// Candidate window layout: vertical list, or a grid carrying its own
    /// columns × rows.
    pub candidate_layout: CandidateLayout,
    /// Whether live conversion is enabled at engine startup
    pub live_conversion: bool,
    /// Which symbol the `,` `.` `/` `[` `]` keys type
    pub symbol: SymbolStyle,
    /// The width kana input comes out at, per character group
    pub width: WidthRules,
    /// The space the Space key inputs
    pub space: SpaceStyle,
    /// Date/time phrases and their formats
    pub date: DateConfig,
}

impl EngineConfig {
    /// Build an engine config from user settings (config.toml).
    /// Shared by the fcitx5 FFI and the stdio JSON-RPC server.
    pub fn from_settings(settings: &crate::config::Settings) -> Self {
        Self {
            num_candidates: settings.conversion.num_candidates,
            display_context_chars: 10,
            context_chars: if settings.conversion.use_context {
                settings.conversion.context_chars
            } else {
                0
            },
            chunk_chars: settings.conversion.chunk_chars,
            chunk_symbols: settings.conversion.chunk_symbols,
            chunk_digits: settings.conversion.chunk_digits,
            chunk_alphabets: settings.conversion.chunk_alphabets,
            beam_chars: settings.conversion.beam_chars,
            beam_width: settings.conversion.beam_width,
            max_latency_ms: settings.conversion.max_latency_ms,
            strategy: settings.conversion.strategy,
            verbose: settings.display.verbose,
            candidate_layout: settings.display.candidate_layout,
            live_conversion: settings.conversion.live_conversion,
            symbol: settings.symbol.style(),
            width: settings.width,
            space: settings.symbol.space,
            date: settings.date.clone(),
        }
    }

    /// Candidates per page of the candidate window: the grid's rows ×
    /// columns, or the classic 9-row column. A zero dimension in the config
    /// is treated as 1 so the window can never go empty.
    pub fn candidate_page_size(&self) -> usize {
        match self.candidate_layout {
            CandidateLayout::Vertical => CandidateList::DEFAULT_PAGE_SIZE,
            CandidateLayout::Grid { columns, rows } => columns.max(1) * rows.max(1),
        }
    }

    /// Columns of the grid layout; `None` for the vertical list. This is
    /// what the frontends key their rendering on, and what makes ↑/↓ (and
    /// Ctrl+N/P) move by a row instead of a candidate.
    pub fn candidate_grid_columns(&self) -> Option<usize> {
        match self.candidate_layout {
            CandidateLayout::Vertical => None,
            CandidateLayout::Grid { columns, .. } => Some(columns.max(1)),
        }
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            num_candidates: 3, // Space conversion: beam search with 3 candidates
            display_context_chars: 10,
            context_chars: 10,
            chunk_chars: 30,
            chunk_symbols: 1,
            chunk_digits: 0,
            chunk_alphabets: 0,
            beam_chars: 30,
            beam_width: 3,
            max_latency_ms: 100,
            strategy: StrategyMode::default(),
            verbose: false,
            candidate_layout: CandidateLayout::default(),
            live_conversion: false,
            symbol: SymbolStyle::default(),
            width: WidthRules::default(),
            space: SpaceStyle::default(),
            date: DateConfig::default(),
        }
    }
}

/// Converter bundle: romaji → hiragana, kana → kanji (main + light)
pub(in crate::core) struct Converters {
    /// Romaji to hiragana converter
    pub romaji: RomajiConverter,
    /// Kanji converter (lazy loaded)
    pub kanji: Option<KanaKanjiConverter>,
    /// Light model for beam search
    pub light_kanji: Option<KanaKanjiConverter>,
    /// Candidate rewriters (half-width katakana, symbol variants)
    pub rewriters: RewriterChain,
    /// Date/time phrase rewriter. Held beside the chain, not in it, so its
    /// candidates keep their own source (`CandidateSource::Date`) and stay
    /// out of the learning cache.
    pub date: DateRewriter,
}

/// Input mode for the IME engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum InputMode {
    /// Hiragana mode (default) — romaji is converted to hiragana
    #[default]
    Hiragana,
    /// Katakana mode — preedit displays katakana instead of hiragana
    Katakana,
    /// Alphabet (direct input) mode — characters bypass romaji conversion
    Alphabet,
    /// Emoji shortcode mode — entered by typing `:` from Empty. Behaves
    /// like [`InputMode::Alphabet`] but auto-exits to the prior mode on
    /// commit/cancel; `EmojiRewriter` surfaces candidates for the query.
    Emoji,
}

/// Current [`InputMode`] plus the mode to come back to when a *temporary*
/// mode (Emoji, Alphabet) ends. `comeback` always holds the last
/// non-temporary mode and equals `current` while none is active, so
/// exiting is an unconditional `current = comeback` — even a hop between
/// two temporary modes exits to the user's real kana mode. Fields are
/// private so every transition maintains the invariant.
#[derive(Debug, Default)]
pub(crate) struct ModeState {
    /// Current input mode.
    current: InputMode,
    /// The last non-temporary mode; what [`ModeState::exit_temporary`]
    /// restores. Equal to `current` whenever `current` is not temporary.
    comeback: InputMode,
}

impl ModeState {
    /// Whether `mode` is a temporary, per-composition mode.
    fn is_temporary(mode: InputMode) -> bool {
        matches!(mode, InputMode::Emoji | InputMode::Alphabet)
    }

    /// The current input mode.
    pub(crate) fn current(&self) -> InputMode {
        self.current
    }

    /// Switch directly to `mode`. The user explicitly picked it, so it
    /// also becomes the comeback target.
    pub(crate) fn set(&mut self, mode: InputMode) {
        debug_assert!(
            !Self::is_temporary(mode),
            "temporary mode {mode:?} must be entered via enter_temporary"
        );
        self.current = mode;
        self.comeback = mode;
    }

    /// Enter a *temporary* mode, remembering the current one for
    /// [`ModeState::exit_temporary`]. A hop between two temporary modes
    /// keeps the original comeback target.
    pub(crate) fn enter_temporary(&mut self, mode: InputMode) {
        debug_assert!(
            Self::is_temporary(mode),
            "enter_temporary called with non-temporary mode {mode:?}"
        );
        if !Self::is_temporary(self.current) {
            self.comeback = self.current;
        }
        self.current = mode;
    }

    /// End any temporary mode: come back to the last non-temporary mode.
    /// No-op when none is active, so the commit/cancel/erase exit sites
    /// call it unconditionally. This is what returns the next word to kana
    /// without an explicit toggle key (issue #37).
    pub(crate) fn exit_temporary(&mut self) {
        self.current = self.comeback;
    }
}

/// One internal chunk of the composing buffer with its cached model
/// conversion. Chunks are invisible — the user sees the concatenation of
/// every `converted` as one continuous preedit; splitting only bounds each
/// model call for long input. The lctx a chunk was converted with is
/// derived on demand (`chunk_lctx`), never stored.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(in crate::core) struct ComposingChunk {
    /// Hiragana reading for this chunk (≤ N chars).
    pub reading: String,
    /// Model conversion of `reading` — this chunk's slice of the live preedit.
    /// Falls back to `reading` when the model yields nothing.
    pub converted: String,
}

/// Live conversion state. The displayed text itself is not stored: it is
/// derived from the current chunks (`live_text`), so it can never go stale
/// against them.
#[derive(Debug, Clone, Default)]
pub(in crate::core) struct LiveConversion {
    /// Whether live conversion is enabled (toggled via Ctrl+Shift+L)
    pub enabled: bool,
    /// Whether the live suggestion is currently shown in the preedit.
    /// Cleared by gestures that fall back to the kana display (cursor moves,
    /// first Escape, mode switches) without discarding the chunks.
    pub shown: bool,
}

impl LiveConversion {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            shown: false,
        }
    }
}

/// Dictionary store: system, user, and future cache dictionaries
#[derive(Default)]
pub(in crate::core) struct Dictionaries {
    /// System dictionary for yada double-array trie lookup
    pub system: Option<Dictionary>,
    /// User dictionary (merged from user_dict_paths)
    pub user: Option<Dictionary>,
}

/// Conversion model dispatch strategy based on input length
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(in crate::core) enum ConversionStrategy {
    /// Short input: main model greedy + light model beam search (parallel)
    ParallelBeam { beam_width: usize },
    /// Long input: light model greedy only (skip slow main model)
    LightModelOnly,
    /// Latency-downgraded beam: the light half of [`Self::ParallelBeam`]
    /// alone, so a slow main model costs the beam its quality but not its
    /// candidate count
    LightModelBeam { beam_width: usize },
    /// No light model: main model greedy only
    MainModelOnly,
    /// Main model beam search (used in Light strategy mode where light model occupies main slot)
    MainModelBeam { beam_width: usize },
}

/// Which way Ctrl+R / Ctrl+T rotates the source-filter cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::core) enum FilterDirection {
    Forward,
    Backward,
}

/// Whether a conversion consults the learning cache. Tab asks for
/// [`Self::Skip`] so a noisy history can be escaped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::core) enum LearningLookup {
    Use,
    Skip,
}

/// Timing and adaptive model selection metrics for conversion
#[derive(Debug, Clone, Default)]
pub(in crate::core) struct ConversionMetrics {
    /// Conversion time of the current call in milliseconds (inference only);
    /// reset to 0 at the start of each key/selection so it never carries
    /// over from a previous keystroke
    pub conversion_ms: u64,
    /// Last process_key time in milliseconds (input to result, end-to-end)
    pub process_key_ms: u64,
    /// Display name of the model used for the last conversion
    pub model_name: String,
    /// Adaptive flag: set when the main model exceeded max_latency_ms
    /// Reset when a new word begins (Empty state)
    pub adaptive_use_light_model: bool,
}
