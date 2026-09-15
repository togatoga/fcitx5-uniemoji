//! Settings configuration
//!
//! Manages user-configurable settings for the IME.
//! Default values are defined in `config/default.toml`.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use directories::ProjectDirs;
use karukan_engine::{
    BracketStyle, DateConfig, PunctuationStyle, SlashStyle, SymbolStyle, WidthRules,
};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// Default configuration TOML embedded from config/default.toml
const DEFAULT_CONFIG_TOML: &str = include_str!("../../config/default.toml");

/// Configuration settings for the IME
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Conversion settings
    pub conversion: ConversionSettings,
    /// Learning cache settings
    pub learning: LearningSettings,
    /// What the aux line shows
    pub display: DisplaySettings,
    /// Which symbol each configurable key types
    pub symbol: SymbolSettings,
    /// The width kana input comes out at, per character group
    pub width: WidthRules,
    /// Date/time phrases and their formats
    pub date: DateConfig,
}

/// The space the Space key inputs while typing kana. Alphabet and emoji
/// input always take the ASCII one, like the width rules leave direct input
/// alone.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpaceStyle {
    /// The ASCII space.
    #[default]
    Half,
    /// The ideographic space `　`.
    Full,
}

/// Which symbol the keys with more than one conventional output type. The
/// width these settle at is `[width]`, not this section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolSettings {
    /// What `,` and `.` type: 、。 ，． 、． ，。
    pub punctuation: PunctuationStyle,
    /// What `[` and `]` type: 「」 or []
    pub bracket: BracketStyle,
    /// What `/` types: ・ or /
    pub slash: SlashStyle,
    /// The space Space inputs while typing kana
    pub space: SpaceStyle,
}

impl SymbolSettings {
    /// The key-to-symbol style the romaji converter is built with. Space is
    /// not part of it: no romaji rule types a space.
    pub fn style(&self) -> SymbolStyle {
        SymbolStyle {
            punctuation: self.punctuation,
            bracket: self.bracket,
            slash: self.slash,
        }
    }
}

/// How the candidate window lays out one page. A tagged value, so each
/// layout carries exactly its own parameters (like `[date]`'s phrases):
/// `{ type = "vertical" }` has none, `{ type = "grid", columns = 5,
/// rows = 5 }` carries its dimensions, defaulting to 5×5 when omitted.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CandidateLayout {
    /// One column of up to 9 numbered candidates (mozc-style).
    #[default]
    Vertical,
    /// A grid of `columns` × `rows` candidates per page, like the expanded
    /// table view of macOS 日本語入力 / MS-IME. More candidates per page,
    /// no numbering (Ctrl+digit still selects the first nine).
    Grid {
        #[serde(default = "default_grid_dimension")]
        columns: usize,
        #[serde(default = "default_grid_dimension")]
        rows: usize,
    },
}

fn default_grid_dimension() -> usize {
    5
}

/// Aux-line settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplaySettings {
    /// Start with the detailed aux line (Ctrl+Shift+V toggles it live):
    /// which part the alternatives cover, inference timing, the model that
    /// ran, and the context handed to it.
    pub verbose: bool,
    /// Candidate window layout: vertical list or grid.
    pub candidate_layout: CandidateLayout,
}

/// Conversion strategy mode
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StrategyMode {
    /// Adaptive: dynamically switch between main and light models based on latency
    #[default]
    Adaptive,
    /// Light: use light_model only (loaded into main slot, beam search on Space)
    Light,
    /// Main: use main model only (no light model loaded)
    Main,
}

/// Conversion-related settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionSettings {
    /// Conversion strategy mode (adaptive, light, main)
    pub strategy: StrategyMode,
    /// Number of candidates to show on Space conversion
    pub num_candidates: usize,
    /// Use surrounding text (text left of cursor) as context for conversion
    pub use_context: bool,
    /// Maximum number of surrounding text characters passed to the conversion API
    pub context_chars: usize,
    /// Maximum reading length (in characters) converted by the model in a single
    /// call during live conversion. The composing buffer is split into chunks
    /// of at most this many characters so per-keystroke latency stays bounded
    /// for long input; each chunk's left context is the converted text of the
    /// preceding chunks.
    pub chunk_chars: usize,
    /// Marks (、。！？…) a chunk containing Japanese keeps instead of
    /// splitting there, so a sentence keeps converting as one unit.
    pub chunk_symbols: usize,
    /// Digits a chunk containing Japanese keeps. The default 0 keeps them
    /// out of the converter entirely, which is what protects them: the
    /// model hallucinates on digit runs, dropping or duplicating figures.
    /// Raising it lets short runs (「だい3かい」) convert with the text
    /// around them; the digits that fit ride along and the rest open the
    /// next chunk.
    pub chunk_digits: usize,
    /// Alphabet chars a chunk containing Japanese keeps. The default 0 keeps
    /// latin text passthrough, and keeps the in-progress romaji tail out of
    /// the converter: while 「わせだd」 is being typed, the `d` is an unfired
    /// keystroke, not text. Raising it lets 「Rustで」 convert as one unit at
    /// the cost of feeding that tail to the model on every keystroke.
    pub chunk_alphabets: usize,
    /// Path to dictionary binary file (optional, defaults to data_dir/dict.bin)
    pub dict_path: Option<String>,
    /// Model variant id (optional, defaults to registry default)
    pub model: Option<String>,
    /// Beam search model variant id (used on Space conversion, default model if unset)
    pub light_model: Option<String>,
    /// Chars the beam covers, snapped to chunk boundaries: the trailing
    /// Japanese chunks fitting this budget, always at least the last one.
    /// A digit/symbol chunk and a manual break both stop the span.
    pub beam_chars: usize,
    /// Beam width: how many alternatives the beam returns
    pub beam_width: usize,
    /// Maximum acceptable latency in milliseconds for auto-suggest (0 = disabled)
    /// When a main model conversion exceeds this, the engine adaptively switches to light_model
    pub max_latency_ms: u64,
    /// Number of threads for llama.cpp inference (0 = all cores, llama.cpp default)
    pub n_threads: u32,
    /// Enable live conversion at startup (Ctrl+Shift+L still toggles at runtime)
    pub live_conversion: bool,
}

/// Learning cache settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSettings {
    /// Whether learning is enabled
    pub enabled: bool,
    /// Maximum number of total entries in the learning cache
    pub max_entries: usize,
    /// Maximum surface length (in characters) recorded into the learning
    /// cache; longer conversion results (e.g. whole live-converted
    /// sentences) are not learned
    pub max_surface_chars: usize,
}

impl Default for Settings {
    fn default() -> Self {
        toml::from_str(DEFAULT_CONFIG_TOML).expect("embedded default.toml must be valid")
    }
}

/// Recursively merge `overlay` TOML values on top of `base`.
fn merge_toml(base: &mut toml::Value, overlay: &toml::Value) {
    match (base, overlay) {
        (toml::Value::Table(base_table), toml::Value::Table(overlay_table)) => {
            for (key, value) in overlay_table {
                if let Some(base_value) = base_table.get_mut(key) {
                    merge_toml(base_value, value);
                } else {
                    base_table.insert(key.clone(), value.clone());
                }
            }
        }
        (base, _) => {
            *base = overlay.clone();
        }
    }
}

/// Parse user TOML content merged on top of default.toml.
fn parse_with_defaults(user_content: &str) -> Result<Settings> {
    let mut base: toml::Value = toml::from_str(DEFAULT_CONFIG_TOML)?;
    let user: toml::Value = toml::from_str(user_content)?;
    merge_toml(&mut base, &user);
    let settings: Settings = base.try_into()?;
    Ok(settings)
}

/// Get the project directories for karukan-im.
fn project_dirs() -> Option<ProjectDirs> {
    ProjectDirs::from("com", "karukan", "karukan-im")
}

impl Settings {
    /// Get the data directory path
    pub fn data_dir() -> Option<PathBuf> {
        project_dirs().map(|dirs| dirs.data_dir().to_path_buf())
    }

    /// Get the configuration directory path
    pub fn config_dir() -> Option<PathBuf> {
        project_dirs().map(|dirs| dirs.config_dir().to_path_buf())
    }

    /// Get the configuration file path
    pub fn config_file() -> Option<PathBuf> {
        Self::config_dir().map(|dir| dir.join("config.toml"))
    }

    /// Get the user dictionary directory path.
    ///
    /// All files in this directory are automatically loaded as user dictionaries.
    /// Default: `~/.local/share/karukan-im/user_dicts/`
    pub fn user_dict_dir() -> Option<PathBuf> {
        Self::data_dir().map(|dir| dir.join("user_dicts"))
    }

    /// Get the learning cache file path.
    ///
    /// Default: `~/.local/share/karukan-im/learning.tsv`
    pub fn learning_file() -> Option<PathBuf> {
        Self::data_dir().map(|dir| dir.join("learning.tsv"))
    }

    /// Load settings from the default configuration file.
    /// Falls back to embedded default.toml if the config file does not exist.
    pub fn load() -> Result<Self> {
        let Some(config_file) = Self::config_file() else {
            warn!("Could not determine config directory, using defaults");
            return Ok(Self::default());
        };

        if !config_file.exists() {
            debug!("Config file not found, using defaults");
            return Ok(Self::default());
        }

        debug!("Loading config from {:?}", config_file);
        Self::load_from(&config_file)
    }

    /// Load settings from a specific file, merged on top of defaults.
    pub fn load_from(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        parse_with_defaults(&content)
    }

    /// Save settings to the default configuration file
    pub fn save(&self) -> Result<()> {
        let Some(config_file) = Self::config_file() else {
            anyhow::bail!("Could not determine config directory");
        };

        debug!("Saving config to {:?}", config_file);
        self.save_to(&config_file)
    }

    /// Save settings to a specific file
    pub fn save_to(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use karukan_engine::Width;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_settings() {
        let settings = Settings::default();
        assert_eq!(settings.conversion.num_candidates, 9);
        assert!(settings.conversion.use_context);
        assert_eq!(settings.conversion.context_chars, 10);
        assert!(settings.learning.enabled);
        assert_eq!(settings.learning.max_entries, 10000);
        assert_eq!(settings.learning.max_surface_chars, 50);
    }

    #[test]
    fn test_default_symbol_and_width_settings() {
        // Shipped defaults: the Japanese symbols, and kana input that comes
        // out full-width apart from digits.
        let settings = Settings::default();
        assert_eq!(settings.symbol.punctuation, PunctuationStyle::KutenTouten);
        assert_eq!(settings.symbol.bracket, BracketStyle::Corner);
        assert_eq!(settings.symbol.slash, SlashStyle::MiddleDot);
        assert_eq!(settings.symbol.space, SpaceStyle::Half);
        assert_eq!(settings.width.kana_symbol, Width::Full);
        assert_eq!(settings.width.ascii_symbol, Width::Full);
        assert_eq!(settings.width.digit, Width::Half);
    }

    #[test]
    fn test_symbol_style_is_written_as_the_symbols() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
[symbol]
punctuation = "，．"
bracket = "[]"
slash = "/"
space = "half"
"#
        )
        .unwrap();

        let settings = Settings::load_from(file.path()).unwrap();
        assert_eq!(settings.symbol.punctuation, PunctuationStyle::CommaPeriod);
        assert_eq!(settings.symbol.bracket, BracketStyle::Square);
        assert_eq!(settings.symbol.slash, SlashStyle::Slash);
        assert_eq!(settings.symbol.space, SpaceStyle::Half);
    }

    #[test]
    fn test_width_partial_config() {
        // Setting one group leaves the others at their default.
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
[width]
ascii_symbol = "half"
digit = "full"
"#
        )
        .unwrap();

        let settings = Settings::load_from(file.path()).unwrap();
        assert_eq!(settings.width.ascii_symbol, Width::Half);
        assert_eq!(settings.width.digit, Width::Full);
        assert_eq!(settings.width.kana_symbol, Width::Full);
    }

    #[test]
    fn test_unknown_keys_are_ignored() {
        // A config written for an older version (e.g. the removed
        // short_input_threshold key) must still load: unknown keys are
        // ignored, the renamed setting falls back to its default, and
        // the other overrides keep applying.
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
[conversion]
short_input_threshold = 10
beam_width = 5
"#
        )
        .unwrap();

        let settings = Settings::load_from(file.path()).unwrap();
        assert_eq!(settings.conversion.chunk_chars, 30);
        assert_eq!(settings.conversion.beam_width, 5);
    }

    #[test]
    fn test_learning_partial_config() {
        // Overriding one learning key keeps the defaults for the others.
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
[learning]
max_surface_chars = 10
"#
        )
        .unwrap();

        let settings = Settings::load_from(file.path()).unwrap();
        assert_eq!(settings.learning.max_surface_chars, 10);
        assert!(settings.learning.enabled);
        assert_eq!(settings.learning.max_entries, 10000);
    }

    #[test]
    fn test_date_phrases_replace_the_shipped_list() {
        // phrases is an array: writing one replaces the shipped list
        // wholesale (extend by copying default.toml's list). offset_days
        // defaults to 0, and the shared [date] formats stay untouched.
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
[date]
phrases = [
    {{ reading = "らいしゅう", offset_days = 7 }},
    {{ reading = "きょう", formats = ["{{YEAR}}年"] }},
]
"#
        )
        .unwrap();

        let settings = Settings::load_from(file.path()).unwrap();
        let phrases = &settings.date.phrases;
        assert_eq!(phrases.len(), 2);
        assert_eq!(phrases[0].reading, "らいしゅう");
        assert_eq!(phrases[0].offset_days, 7);
        assert!(phrases[0].formats.is_none());
        assert_eq!(phrases[1].offset_days, 0);
        assert_eq!(
            phrases[1].formats.as_deref(),
            Some(["{YEAR}年".to_string()].as_slice())
        );
        assert!(!settings.date.formats.is_empty());
    }

    #[test]
    fn test_default_candidate_layout() {
        let settings = Settings::default();
        assert_eq!(settings.display.candidate_layout, CandidateLayout::Vertical);
    }

    #[test]
    fn test_candidate_layout_grid_defaults_to_5x5() {
        // The layout and its parameters travel together; omitted grid
        // dimensions fall back to 5.
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
[display]
candidate_layout = {{ type = "grid", rows = 4 }}
"#
        )
        .unwrap();

        let settings = Settings::load_from(file.path()).unwrap();
        assert_eq!(
            settings.display.candidate_layout,
            CandidateLayout::Grid {
                columns: 5,
                rows: 4
            }
        );
        assert!(!settings.display.verbose);
    }

    #[test]
    fn test_serialize_deserialize() {
        let settings = Settings::default();
        let toml_str = toml::to_string(&settings).unwrap();
        let loaded: Settings = toml::from_str(&toml_str).unwrap();
        assert_eq!(
            loaded.conversion.num_candidates,
            settings.conversion.num_candidates
        );
    }

    #[test]
    fn test_load_from_file() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
[conversion]
num_candidates = 5
use_context = false
"#
        )
        .unwrap();

        let path = file.path().to_path_buf();
        let settings = Settings::load_from(&path).unwrap();
        assert_eq!(settings.conversion.num_candidates, 5);
        assert!(!settings.conversion.use_context);
    }

    #[test]
    fn test_user_dict_dir() {
        let dir = Settings::user_dict_dir();
        // Should return Some on systems with a home directory
        if let Some(dir) = dir {
            assert!(dir.ends_with("user_dicts"));
        }
    }

    #[test]
    fn test_partial_config() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
[conversion]
num_candidates = 3
"#
        )
        .unwrap();

        let path = file.path().to_path_buf();
        let settings = Settings::load_from(&path).unwrap();
        assert_eq!(settings.conversion.num_candidates, 3);
        // Should use default for unspecified values
        assert!(settings.conversion.use_context);
        assert_eq!(settings.conversion.context_chars, 10);
    }

    #[test]
    fn test_strategy_default_when_unspecified() {
        // When strategy is not specified, it should default to Adaptive
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
[conversion]
num_candidates = 5
"#
        )
        .unwrap();

        let path = file.path().to_path_buf();
        let settings = Settings::load_from(&path).unwrap();
        assert_eq!(settings.conversion.strategy, StrategyMode::Adaptive);
    }

    #[test]
    fn test_strategy_light() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
[conversion]
strategy = "light"
"#
        )
        .unwrap();

        let path = file.path().to_path_buf();
        let settings = Settings::load_from(&path).unwrap();
        assert_eq!(settings.conversion.strategy, StrategyMode::Light);
    }

    #[test]
    fn test_strategy_main() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            r#"
[conversion]
strategy = "main"
"#
        )
        .unwrap();

        let path = file.path().to_path_buf();
        let settings = Settings::load_from(&path).unwrap();
        assert_eq!(settings.conversion.strategy, StrategyMode::Main);
    }
}
