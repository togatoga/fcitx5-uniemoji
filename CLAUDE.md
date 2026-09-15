# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

karukan is a Japanese Input Method system for Linux and macOS, consisting of four Rust crates and a Swift package. The IME core and its platform frontends are grouped under `karukan-im/`:

- **karukan-engine** (`karukan-engine/`): Core library — romaji-to-hiragana conversion, neural kana-kanji conversion via llama.cpp, system dictionary, learning cache, candidate rewriter (width/case/symbol variants)
- **karukan-cli** (`karukan-cli/`): CLI tools and server — dictionary builder, Sudachi converter, dict viewer, AJIMEE-Bench, HTTP API server
- **karukan-im** (`karukan-im/core/`): Shared IME engine state machine (Empty → Composing → Conversion) and `karukan-imserver` stdio JSON-RPC server (macOS binary bundled in the macOS frontend)
- **karukan-fcitx5** (`karukan-im/fcitx5/`): fcitx5 Linux frontend — C FFI (`src/ffi/`) and C++ addon (`fcitx5-addon/`) that wrap karukan-im
- **karukan-macos** (`karukan-im/macos/`): Swift/InputMethodKit frontend that spawns `karukan-imserver` as a bundled child process

## Build and Development Commands

This project uses a Cargo workspace. All commands are run from the repository root.

### Full workspace

```bash
cargo build --release       # Build all crates
cargo test --workspace      # Run all tests
```

### karukan-engine

```bash
cargo build -p karukan-engine --release
cargo test -p karukan-engine  # includes integration tests (model auto-downloaded on first run)
```

### karukan-cli

```bash
cargo build -p karukan-cli --release

# Start the server (auto-downloads models from HuggingFace)
cargo run --release --bin karukan-server

# Build dictionary from JSON or Mozc TSV
cargo run --release --bin karukan-dict -- build input.json -o dict.bin

# Build scored dictionary from Sudachi CSV
cargo run --release --bin sudachi-dict -- input.csv -o scored.json

# Dictionary viewer (web UI + CLI search)
cargo run --release --bin karukan-dict -- view dict.bin

# AJIMEE-Bench evaluation
cargo run --release --bin ajimee-bench -- evaluation_items.json
```

### karukan-im

```bash
cargo build -p karukan-im --release
cargo test -p karukan-im
```

### karukan-fcitx5

```bash
cargo build -p karukan-fcitx5 --release
cargo test -p karukan-fcitx5

# Build and install fcitx5 addon
cd karukan-im/fcitx5/fcitx5-addon

# Option A: System install (sudo required, no FCITX_ADDON_DIRS needed)
cmake -B build -DCMAKE_INSTALL_PREFIX=/usr
cmake --build build -j
sudo cmake --install build

# Option B: User-local install (no sudo, requires FCITX_ADDON_DIRS)
cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build build -j
cmake --install build
```

### karukan-macos

```bash
cd karukan-im/macos

make test      # Swift tests (incl. integration tests against a real karukan-imserver)
make install   # Build, assemble Karukan.app, install to ~/Library/Input Methods (auto-downloads dict.bin if missing and prefetches all models.toml models into the HF cache)
```

First install requires logout/login; afterwards `make install` + `killall KarukanIME` suffices.

### Code Quality

```bash
cargo fmt --all       # Format all crates
cargo clippy --workspace  # Lint all crates
```

## Architecture

### karukan-engine (`karukan-engine/src/`)

- `lib.rs` — Library entry point and re-exports
- `romaji/` — Romaji-to-hiragana conversion
  - `trie.rs` — Trie data structure
  - `rules.rs` — 200+ conversion rule
  - `style.rs` — `SymbolStyle`: which symbol the `,` `.` `/` `[` `]` keys type
  - `converter.rs` — Stateless converter (`convert`/`flush_pending`/`starts_rule`)
- `kanji/` — Kana-kanji conversion via llama.cpp
  - `backend.rs` — Backend + KanaKanjiConverter
  - `llamacpp.rs` — GGUF inference
  - `hf_download.rs` — HuggingFace model download
  - `model_config.rs` — models.toml registry
  - `error.rs` — KanjiError type
- `rewriter/` — Candidate rewriter system
  - `mod.rs` — Rewriter trait, RewriterChain, default_chain()
  - `alphabet.rs` — Alphabet width/case variants (e.g. `abc` → `ABC`, `ａｂｃ`, `ＡＢＣ`)
  - `date.rs` — Date/time phrases (`[date]` in config.toml): きょう/あした/いま render format strings against today±offset_days. The only rewriter carrying config; held beside the chain (`Converters::date`) so its candidates keep `CandidateSource::Date` and skip the learning cache
  - `half_katakana.rs` — Half-width katakana variants (e.g. `がっこう` → `ｶﾞｯｺｳ`)
  - `symbol.rs` — Symbol variant chains, reading→symbol lookup (Mozc symbol.tsv derived), and the whole-candidate width pair (`。` → `｡`, `(a1)` → `（ａ１）`) for candidates made entirely of characters that have both forms. The pair annotates in place when a chain already produced the same text, so nothing is lost either way: `｡` reads `[半]記号`, `＠` `[全]アットマーク`
  - `width.rs` (crate root) — the width groups themselves: `WidthRules` (what `[width]` configures) plus `to_half_width` / `to_full_width` / `has_width_pair`, shared by the rewriters
- `dict.rs` — Double-array trie system dictionary
- `learning.rs` — Learning cache (user conversion history, TSV persistence, recency+frequency scoring)
- `kana.rs` — Hiragana/katakana utilities, full-width/half-width conversion functions

### karukan-cli (`karukan-cli/src/`)

- `bin/dict.rs` — Dictionary tool: build (JSON or Mozc TSV → binary) and view (web UI + CLI search)
- `bin/sudachi_dict.rs` — Sudachi dictionary → scored JSON converter
- `bin/server.rs` — Axum HTTP API server
- `bin/ajimee_bench.rs` — AJIMEE-Bench evaluation
- `static/` — Web UI assets for server and dict-viewer

### karukan-im (`karukan-im/core/src/`)

- `core/engine/` — IMEEngine state machine (Empty → Composing → Conversion)
  - `mod.rs` — Main InputMethodEngine struct and core processing logic
  - `types.rs` — EngineConfig, EngineResult, EngineAction, Converters, ConversionStrategy
  - `input.rs` — Key input handling for Composing state
  - `input_buffer.rs` — Composition record: per-display-char element array (`Romaji`/`Converted`) + caret index; display/reading/pending are derived views
  - `conversion.rs` — Conversion mode handling (mixed candidate list, key handling, commit)
  - `model.rs` — Model dispatch and the cache in front of it (`run_kana_kanji_conversion`, `cached_convert`, `run_parallel_beam`, `model_candidates`)
  - `filter.rs` — Ctrl+R / Ctrl+T source-filtered candidate views (`FILTER_CYCLE`, `source_view`, `apply_candidate_filter`)
  - `chunk/split.rs` — Where the chunk boundaries fall, engine-free: every char is classified into a `Token`, and `group_chunks` walks them once, packing them into chunks under `ChunkLimits`. Pure functions, unit-tested in place
  - `chunk/mod.rs` — What the engine does with the chunks: `chunked_auto_suggest`, the chunk-grid conversion, and the manual breaks
  - `cache.rs` — LRU conversion cache keyed by the computation: (katakana reading, lctx, model role, beam width)
  - `cursor.rs` — Cursor movement
  - `display.rs` — Preedit text display
  - `mode.rs` — Mode switching (katakana, alphabet, live conversion)
  - `init.rs` — Dictionary setup, learning cache init, and background model loading (models load on a spawned thread; `poll_loaded_models` installs them on the next key event)
  - `strategy.rs` — Conversion strategy determination and adaptive model selection
  - `tests.rs` — Engine unit tests
- `core/preedit.rs` — Preedit composition with cursor support
- `core/candidate.rs` — Candidate list with pagination support
- `core/keycode.rs` — Key symbol definitions and key event handling
- `core/state.rs` — Engine state definitions
- `config/settings.rs` — User settings (`~/.config/karukan-im/config.toml` on Linux, `~/Library/Application Support/com.karukan.karukan-im/` on macOS)
- `server/` — stdio JSON-RPC 2.0 server for the macOS frontend (`protocol.rs` defines the wire format; `bin/karukan-imserver.rs` is the entry point)

### karukan-fcitx5 (`karukan-im/fcitx5/`)

Linux fcitx5 frontend. Wraps karukan-im via C FFI and exposes the engine to the C++ addon.

- `src/ffi/mod.rs` — `KarukanEngine` opaque struct, action dispatch, cache structs, FFI macros
- `src/ffi/lifecycle.rs` — `karukan_engine_new/init/free`
- `src/ffi/input.rs` — `karukan_engine_process_key/reset/set_surrounding_text`
- `src/ffi/query.rs` — All getter functions (preedit, commit, candidates, aux, timing)
- `include/karukan.h` — C header for the fcitx5 C++ addon
- `fcitx5-addon/src/karukan.cpp` — C++ fcitx5 wrapper

### karukan-macos (`karukan-im/macos/Sources/KarukanIME/`)

Swift/InputMethodKit frontend. All IME state lives in karukan-imserver (spawned as a bundled child process); Swift only adapts IMK events and renders UI.

- `main.swift` — IMKServer startup, engine process spawn, wake-from-sleep restart, SIGPIPE handling
- `KarukanInputController.swift` — IMKInputController; translates keys, applies engine actions (preedit/candidates/commit), JIS かな key and right-Command tap return to hiragana (exit katakana mode)
- `KeyCodeMap.swift` — NSEvent → XKB keysym translation (same keysym representation as fcitx5), RightCommandTapDetector
- `resources/*.tiff` — template menu icon (か), regenerated via `swift scripts/generate_icons.swift`; `resources/{ja,en}.lproj/InfoPlist.strings` localize the input mode name shown in the input menu
- `EngineProcess.swift` — child process lifecycle: crash restart with exponential backoff, EOF-based clean shutdown (lets the server save its learning cache)
- `EngineClient.swift` — JSON-RPC transport (sync for process_key, async for fire-and-forget)
- `EngineProtocol.swift` — Swift mirror of `karukan-im/core/src/server/protocol.rs` (keep in sync; protocol_version guards breaking changes)
- `CandidateWindowController.swift` — custom NSPanel candidate window (engine pre-paginates; renders the page as a numbered vertical list or, when `ShowCandidates` carries `grid_columns`, a measured-frame grid; cursor moves restyle cells in place instead of rebuilding)

## macOS Input Mode Design

`karukan-macos` registers **only the Japanese input mode** (`dev.togatoga.inputmethod.Karukan.Japanese`) in `Info.plist`. There is no Roman/英数 mode inside Karukan — if the user wants to type in Latin script they switch to the OS-level English input source (e.g. via Karabiner). Do not add a Roman mode back; it is intentionally absent.

The engine-internal `InputMode::Alphabet` (entered via Shift+letter on Linux/fcitx5) is a separate Rust engine concept unrelated to this macOS input mode registration. Do not conflate the two.

## Key Design Patterns

- IMEEngine uses a state machine: Empty → Composing → Conversion
- `input_buf: InputBuffer` in IMEEngine is the source of truth: an element array (one element per display character — `Romaji(char)` unfired keystroke / `Converted(char)` settled character: fired output, passthrough, or direct input) plus a caret index. All views (display, conversion reading, aux romaji tail) are derived from it
- `RomajiConverter` is stateless (pure `convert`/`flush_pending`); after each romaji keystroke the engine re-evaluates only the Romaji run ending at the caret (`evaluate_run`), re-recording fired keystrokes as `Converted`. `Converted` never re-enters evaluation, so settled text never reverts. Backspace/delete remove one element and then evaluate the run the removal joined, so the result always equals typing the remaining keystrokes fresh (`ykt` → BS → `o` → 「yこ」; `yt1t` minus the `1` → 「yっt」). Cursor moves and mode toggles never touch the array
- Models use jinen format with special Unicode tokens (U+EE00–U+EE02) from the Private Use Area; model input is katakana (hiragana is converted to katakana before inference)
- Model registry defined in `karukan-engine/models.toml`; default models use Q5_K_M quantization
- Model files resolve cache-first (`download_gguf`): a file already in the HuggingFace cache is returned without any network request, so a cached model resolves instantly offline; the network (short retry budget) is only touched on a cache miss. The flip side: an update pushed to the same repo/filename is not picked up once cached — model updates must change the filename. Model loading itself runs on a background thread (`spawn_model_loading`): `init` returns immediately, the engine runs dictionary/kana-only until `poll_loaded_models` (top of `process_key`) installs the converters, and a load failure (e.g. offline with an empty cache) leaves the IME running without a model instead of failing init
- Live conversion (auto-suggest) splits the composing buffer into internal chunks of at most `chunk_chars` reading chars (default 30). Splitting bounds each model call and freezes settled text: once a boundary is behind the caret that chunk's reading and lctx no longer change, so it stays a cache hit and its display stops flickering. Chunks are internal — the user sees one continuous preedit
- Chunk boundaries (`group_chunks`) open at the length cap, at a manual break (Ctrl+J → `insert_chunk_break`; boundaries live in `chunk_breaks` as reading positions, shift with edits via `edit_with_chunk_breaks`, and are cleared when the composition ends), and around non-Japanese chars (`is_japanese`: hiragana, katakana incl. `ー`, and kanji are Japanese; the middle dot `・` is special-cased as non-Japanese). A chunk containing Japanese keeps marks up to `chunk_symbols` (default 1) so 「おい、お前だよ」 keeps converting as one unit instead of freezing a premature 「老、」; digits up to `chunk_digits` (default 0, so digits stay out of the model — it hallucinates on digit runs, dropping or duplicating figures), filled greedily like the symbol budget, so the digits that fit ride along and the rest open the next chunk; and alphabet chars up to `chunk_alphabets` (default 0, so latin text stays passthrough and the unfired romaji tail — the `d` in 「わせだd」 — never reaches the model as part of the reading). A chunk with no Japanese is exempt from both caps and never reaches the model
- `chunked_auto_suggest` re-chunks the whole buffer from scratch on every keystroke and re-runs every chunk through `run_kana_kanji_conversion`, which caches results in an LRU keyed by the computation itself — (katakana reading, lctx, model role, beam width) rather than by the requesting strategy, so strategies sharing a computation share its entry (`ParallelBeam` is exactly a main-greedy plus a light-beam entry, and its halves are the same ones `MainModelOnly` / `LightModelBeam` compute). Unchanged chunks are cache hits and only chunks whose reading or left context changed reach the model (a middle edit therefore reconverts the chunks to its right with their updated lctx). Each chunk's left context (lctx) is the editor surrounding text plus the converted text of the preceding chunks, truncated to `context_chars`
- The aux line is quiet by default — mode, state, reading, candidate source, page. Both states show the same reading field, the caret's chunk with a `used/max` fill counter (「わせだd 3/30」, `0/30` right after a manual break, making the cut visible): the conversion line calls the composing line's own `aux_reading` (`conversion_reading`), so Space adds the conversion's fields instead of replacing what typing reported, and the chunk is split from the buffer on demand rather than read off `self.chunks` — typing inside a filtered view suppresses the suggestion, which clears the grid. A view that passes its own text (a predictive entry's 「query → reading」) keeps it and only takes the counter. The mode indicator leads the conversion line too (`[A][変換:🤖]`), since a candidate window is open for keystrokes (typing refines, Shift+letter switches to direct input) and nothing else would say which mode is in force. Ctrl+Shift+V (`toggle_verbose`, `[display] verbose` to start that way) adds the debug details, re-rendering the current state's line on the spot: the chunk is swapped for the beam span, labelled 🎯 and counted against `beam_chars` (`🎯 うえ 2/8`), plus inference timing, the model that ran, and the lctx handed to it (`conversion_chunk_reading`; the learning/dictionary views and a selected predictive candidate keep the plain reading, since neither is beamed as a span). User-facing doc: `docs/chunking.md`
- Dictionary lookup during composing/conversion is exact match plus predictive (prefix-extending) matches: readings that start with the typed reading surface as extra candidates (2+ typed chars required; up to 3 in the composing suggestion list, uncapped in the paged conversion list), ranked by `score + 500·ln(50·remaining_chars)` — dictionary scores are -500·log(p) costs, so longer completions are demoted on the same scale. Predictive candidates carry their full reading so committing records under the right key. While a romaji tail is unresolved, prediction is narrowed to the kana that tail can become (わせ + `d` keeps わせだ… and drops わせり…; a tail that cannot become kana suppresses prediction)
- Learning cache records user-selected conversions and boosts them on subsequent conversions; candidate priority: Learning → User Dictionary → Model → System Dictionary → Fallback → Rewriter. Surfaces longer than `max_surface_chars` (default 50, `[learning]` in config.toml; both learning limits travel as `LearningConfig`) are not recorded. During conversion, Ctrl+Backspace or Ctrl+Delete (the Mac "delete" key is Backspace) removes the selected learning candidate from the history, mozc-style (`DeleteSelectedCandidate`): the removal clears the entry's whole prefix fan-out (`LearningCache::remove_suggestion`) so deduped twins under longer readings can't resurface, and the conversion is then rebuilt in place (mozc's cancel-and-reconvert, minus the window blink) so a surface that the model/dictionary/fallback also produce survives as an ordinary candidate. With a non-learning candidate selected the chord is consumed but does nothing (mozc's `DoNothing`); cancelling stays on plain Backspace/Escape; the chord exists only in the Conversion state (Composing keeps plain char editing). Deletability is derived from `Candidate::source` (`CandidateSource::is_deletable`, mozc's `USER_HISTORY_PREDICTION` analogue), and while a learning candidate is selected the aux text shows the mozc-style footer hint 「Ctrl+Backspaceで履歴から削除」
- Typing a printable character during Conversion refines instead of committing (deliberate mozc deviation, incremental-search style): the engine drops back to the untouched composition and feeds the keystroke, so the reading grows and the live suggestion rewrites in place; inside a narrowed source view the filter survives the keystroke (`refine_through_composing` re-enters the conversion with the same source; the intermediate composing render is discarded, so its auto-suggest inference is suppressed for that one call — the view runs its own lookup) and plain Backspace mirrors it by shrinking the reading in place (unfiltered Backspace still cancels back to the composition). Ctrl+digit (1-9) selects a candidate — bare digits refine like any other printable char, so typing numbers never conflicts with selection — Ctrl+J (`rebreak_conversion`) inserts a chunk break at the caret and rebuilds in place, narrowing what the beam covers while keeping the active source filter, and Enter commits
- Source views query their own source: learning and dictionary use the live buffer's base reading + romaji tail so mid-word consonants narrow predictively — the tail constraint applies to learning predictions too, so a stale exact match can never swallow the tail on commit; the model and rewriter views use the state's settled reading (the exact text Enter commits)
- Model candidates — Space's mixed list and the AI view share the same split conversion (`model_candidates`) — beam only a span: the trailing chunks on the live-conversion grid fitting `beam_chars` (`beam_span_start` → `trailing_chunks_start` walks `group_chunks`' own output backwards, so it always lands on a boundary; a chunk with no Japanese and a manual break each wall the span, keeping digits out of the model and the frozen text frozen; cutting anywhere else would leave a prefix live conversion never converted, costing an extra inference and showing a seam the user never saw). Everything before the span converts top-1 on the grid, so the prefix is exactly the chunks typing already cached. Cost stays bounded and beam-width alternatives survive however long the reading grows; the adaptive latency downgrade (main model over `max_latency_ms`) lands on `LightModelBeam`, so a slow main model costs quality but never the candidate count. An empty model list means the model produced nothing; a candidate equal to the reading is a real answer (a kana-only word like きゃりーぱみゅぱみゅ converts to itself) and rides like any other
- The model list is headed by the whole-reading top-1 on the live-conversion chunk grid — the exact text live typing displays — and it costs no extra pass: the span is the last chunk, so its `ParallelBeam` main-greedy half *is* what the grid would compute for that chunk, and `prefix + that greedy` is the head. Running it as the beam's main half computes the head and the alternatives in parallel rather than one conversion before the other. The head is normally a pure cache replay, since the cache is keyed by computation and main greedy runs at most once per (reading, lctx) however live typing, Space, and the AI view interleave. A light-model request is additionally served by the main model's entry for the same reading and beam width when one exists (never the reverse — that would downgrade quality), so a latency downgrade doesn't re-infer what the main model already converted. The adaptive gate is fed the main model's own elapsed time — ParallelBeam times its main half separately, and a cache-served half reports no measurement — so an expensive one-shot beam can no longer spuriously downgrade the rest of the word
- The candidate window layout is an engine-owned display setting (`[display] candidate_layout`, a tagged value so each layout carries exactly its own parameters, like `[date]`'s phrases: `{ type = "vertical" }` (default) or `{ type = "grid", columns = 5, rows = 5 }`, dimensions defaulting to 5). The engine owns pagination, so the grid is mostly a page-size change: `EngineConfig::candidate_page_size` (columns × rows vs the classic 9) feeds `CandidateList::with_page_size` in `settle_candidates`, and the column count rides on every `ShowCandidates` (`grid_columns`, absent = vertical; additive field, protocol version unchanged) so the frontend renders to match — the macOS panel lays the page out row-major with explicitly measured cell frames (`CandidateGridView`; NSGridView's fitting size inside the stack proved unreliable and opened blank gaps), numberless, since the grid holds more candidates than Ctrl+digit selection can reach (Ctrl+1..9 still selects the first nine), with the selection highlight hugging the cell's own text rather than the column's full width. The panel renders incrementally — a cursor-only change restyles the affected cells and moves the highlight in place, never tearing the view tree down (the rebuild is what flickered) — and its stack is pinned to the panel's top only, so a sizing mismatch can never stretch gaps open inside the window. Navigation maps onto the grid in the Conversion state: ↓/↑ and Ctrl+N/P move by a row (`CandidateList::move_down`/`move_up`, column-preserving — crossing a page's last row lands on the next page in the same column, past the ends wraps within the column), ←/→ and Ctrl+B/F walk the row, Ctrl+A/E jump to the row's first/last occupied cell; in the vertical list all of those keep their existing meanings (↓/↑/Ctrl+N/P next/prev candidate, ←/→/Ctrl+B/F/A/E drop back to caret editing), and Space/Tab stay next-candidate with PageUp/PageDown paging in both layouts. fcitx5 lays candidates out itself, so there the setting only changes how many candidates a page holds
- Ctrl+I jumps straight to the model view from either Composing or Conversion (`source_for_key` → `jump_to_source`); it is the only view with a dedicated key, since the rest are a step or two along the cycle and are opened rarely. Ctrl+T / Ctrl+R in the Conversion state cycle a candidate-source filter forward / backward (dedicated keys, both keysym cases — direction never depends on the Shift bit, which some environments fold into an uppercase keysym; Ctrl+R is the reverse one to match readline's reverse-i-search and because R sits left of T) — Tab stays next-candidate and Shift+Tab (`ISO_LEFT_TAB` on X11) prev-candidate for mozc-compatible muscle memory (`FILTER_CYCLE` in filter.rs: a pure rotation grouped by what the user is after — taught (learning), looked up (📚 covers both dictionaries as one stop, the user's own entries first and deduped by surface, since which book a word came from is already in each candidate's own annotation), guessed (model), rewritten; the model sits late because Ctrl+I reaches it in one press — the full list is not a stop, it is what Space already shows and Esc→Space returns to it; the rewriter view carries the plain kana at its tail (skipped in emoji mode — the picker shows emojis only, and `rewriter_variants` runs the emoji rewriter alone there so a `:smile` query is never answered with another rewriter's width variant), derived from the reading so mixed-list dedup can't hide them, and sits last so Ctrl+R reaches it in one press). Exactly one step per press: an empty source shows an empty window with 「候補なし」 in the aux, never skipped, so the position stays predictable; Ctrl+T while Composing starts the conversion already narrowed one step, Ctrl+R at the cycle tail (`start_filtered_conversion`); the filter and the unfiltered list live in `InputState::Conversion` so they die with the state, and the aux header shows the active source's emoji (`[変換:📝]`). Inside the Conversion state every unbound non-Alt key is consumed as a no-op (mozc's TestSendKey does the same for Composition/Conversion), so chords like Ctrl+R never leak to the application mid-conversion — leaking them was the original bug that made browsers reload; Alt chords pass through before any binding matches (Alt+Return must not commit, Alt+Tab must not navigate) so desktop shortcuts keep working
- Symbols are two settings, not one. `[symbol]` picks *which* symbol a key types — `punctuation` (`、。` / `，．` / `、．` / `，。`), `bracket` (`「」` / `[]`), `slash` (`・` / `/`) — and is baked into the rule trie at construction (`RomajiConverter::with_style`), so every rule output stays non-ASCII and the converter's "an ASCII char in the output is a passed-through keystroke" contract holds whatever the style is. `[width]` picks how wide it comes out, per character group (`kana_symbol` 。、「」・, `ascii_symbol` for every ASCII symbol and its full-width twin including ？！ and ，．, `digit`) with `half` / `full` — two values, since a third meaning "leave it" would be a synonym for `full` on the kana symbols and for `half` on the digits, and on the ASCII symbols would only restore the accident. Three groups, cut by what a user asks for rather than by what a character is: the pair `kana_symbol` / `ascii_symbol` splits them by which character set they come from, which is what makes each name readable: `kana_symbol` is the five with no ASCII counterpart, whose only half-width forms `｡､｢｣･` are the ones that go with half-width katakana, and keeping them apart is what stops "every symbol half-width" from producing 「こんにちは｡」, and that separation is what lets ？！ sit with the ASCII symbols they are the full-width forms of. Splitting the symbols further — parentheses apart from operators apart from quotes — would answer a question nobody asks; that granularity only earns its keep when each group's width is learned from what the user last picked, and nothing here learns it. `space` (`half`, shipped, / `full`) governs the bare Space key while typing kana; direct input always takes the ASCII one, the same edge the width rules stop at. A half-width space is left unconsumed in the Empty state so the application keeps its own Space behaviour. Shift+Space splits by state: from Empty it commits a full-width space whatever the setting says (the escape hatch, committed directly rather than opening a composition a second Space would convert), while mid-composition it inserts the *configured* space (`input_space`) — bare Space converts there, so this is the only way to put a space into a composition, and the width wanted inside one is the everyday one, so a half-width space can be part of what gets converted. Spaces in the model's output settle with the rest (`settle_text`): NFKC flattens `　` to ` ` in the prompt, so the model can only ever answer with the half-width one
- The width rules describe kana input and stop at its edge (`width_for`): in Alphabet and Emoji mode nothing is converted, so `Hello, world.` typed with Shift stays ASCII and a `:smile` query stays typeable. Which is why the shipped defaults can be `kana_symbol`/`ascii_symbol` = `full`: kana input comes out full-width as it does in every Japanese IME, instead of the accident it used to be, where a symbol was full-width iff the rule table happened to carry it (`？！` full, `(` `@` `:` half). `digit` ships `half`: nothing here remembers the width last picked, so a full-width default would be one the user cannot take back. Letters have no setting at all — in kana input a latin char is either an unfired keystroke (the `d` in 「わせだd」, which a full-width rule would render 「わせだｄ」) or a dictionary surface, so `Group::Alphabet` exists only to give letters a half/full pair for the rewriter's variants and always resolves to `AsIs`
- The width applies in two places, split by who produced the text. **Typed text settles as it is typed**: `evaluate_run` gives a character its configured width the moment it stops being a live keystroke (the converter carries the rules, `RomajiConverter::with_rules`), so each character keeps the width in force when it was typed — `（` then Shift+A leaves 「（A」, where deciding by the current mode would have reverted the paren the moment direct input started. Live keystrokes are untouched, and so is direct input (alphabet/emoji), which is what keeps `Hello, world.` ASCII. **The model's answers settle on the way in** (`settle_text`, from `live_text` and `settle_candidates`). That half is not optional — the model is prompted with NFKC and answers in half-width whatever was typed, so a width applied only while typing would be undone by the next conversion. It is also the *only* text settled that way: the model's width is an artifact, while every other source carries a width someone chose, so the dictionaries, learning, the rewriter's variants and the kana fallbacks are left alone. Folding them would rewrite authored spellings — the system dictionary has `Yahoo!`, `P!NK`, `Ya-foo`. Nothing settles on the way *out*: a candidate the user picked is already the width they chose, and a second pass would fold `＜＞１２３４` back to `＜＞1234` on commit. Candidate lists settle at construction rather than at display because the stored list is what Ctrl+digit indexes and what commit reads; folding creates duplicates (with `digit = "full"` both `1` and the rewriter's `１` come out `１`) and only the first survives. The rewriter's whole-candidate width pair (`。` → `｡` `[半]記号`, `<>1234` → `＜＞１２３４` and `<>1234`) is the per-instance escape hatch from the setting. User-facing doc: `docs/symbols.md`
- Date/time conversion is a rewriter with config: `[date]` in config.toml lists phrases (きょう/あした/いま — `reading`, `offset_days`, optional `formats`) whose format strings render at conversion time. `phrases` is an array, so a user config writing it replaces the shipped list wholesale (`merge_toml` merges tables but replaces arrays — extend by copying default.toml's list, drop a phrase by omitting it; the first entry wins on a duplicated reading). Tokens are zero-padded (`{YEAR}/{MONTH}/{DATE}` → `2026/09/06`); `:bare` unpads (`9月`), `:kanji` renders kanji numerals — the year digit-by-digit (`二〇二六`), the rest read-style (`十六`), era year 1 as `元`. A format with an unknown token, or `{ERA}` before 明治, is skipped rather than shown broken; a phrase without its own `formats` shares the `[date]` list, so the shipped date phrases are one line each. `DateRewriter` is the only rewriter carrying config and the only clock-reading one — rendering lives in `candidates_at(reading, now)` so tests pin the clock — which is why it sits beside the chain (`Converters::date`, built in `with_config`) instead of in it: its candidates keep their own `CandidateSource::Date` (📅, no cycle stop — they head the Ctrl+R rewriter view and sit above the width variants in the mixed tail), and that source is what marks them non-learnable (`is_learnable`, checked at both commit paths). A committed date is never recorded: recorded once, yesterday's date would head every later きょう conversion at learning priority. User-facing doc: `docs/date.md`
- Data files (system dictionary `dict.bin`, user dictionaries `user_dicts/`, learning cache `learning.tsv`) live in the data directory: `~/.local/share/karukan-im/` on Linux, `~/Library/Application Support/com.karukan.karukan-im/` on macOS; a prebuilt `dict.tgz` is published on GitHub releases
- Learning cache is persisted as TSV (`learning.tsv` in the data directory); saved on deactivate and engine free, not on every commit
- Learning score uses recency-weighted formula (mozc-inspired): `recency * 10.0 + ln(1 + frequency)`; eviction removes lowest-score entries when over `max_entries` (default: 10,000)

## Training (karukan-jinen)

Model training is handled by the separate `karukan-jinen` Python project (not in this repository). It trains small language models (GPT-2 and Qwen3 based) for kana-kanji conversion using the jinen format, and outputs GGUF files for use with karukan-engine.
