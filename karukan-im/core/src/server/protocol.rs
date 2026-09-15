//! JSON-RPC 2.0 protocol types for the stdio IME server.
//!
//! The macOS (Swift) frontend spawns `karukan-imserver` as a child process
//! and exchanges newline-delimited JSON-RPC 2.0 messages over stdin/stdout.
//! One request per line, one response per line. stderr carries logs only.
//!
//! Methods:
//!
//! | method                 | params                                   | result            |
//! |------------------------|------------------------------------------|-------------------|
//! | `init`                 | `{}`                                     | [`InitResult`]    |
//! | `process_key`          | [`ProcessKeyParams`]                     | [`KeyResult`]     |
//! | `select_candidate`     | [`SelectCandidateParams`]                | [`KeyResult`]     |
//! | `commit`               | `{}`                                     | [`KeyResult`]     |
//! | `reset`                | `{}`                                     | `{}`              |
//! | `set_surrounding_text` | [`SurroundingTextParams`]                | `{}`              |
//! | `save_learning`        | `{}`                                     | `{}`              |
//! | `status`               | `{}`                                     | [`StatusResult`]  |
//!
//! Keysyms are XKB keysym values (the same representation used by the
//! fcitx5 FFI); the frontend is responsible for translating native key
//! events (e.g. `NSEvent`) into XKB keysyms.
//!
//! All positions (`caret`, attribute `start`/`end`, `cursor_pos`) are in
//! Unicode scalar values (Rust `char` counts), not bytes and not UTF-16
//! code units.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::keycode::KeyModifiers;

/// Protocol version reported by `init`. Bump on breaking changes.
pub const PROTOCOL_VERSION: u32 = 1;

// === JSON-RPC envelope ===

#[derive(Debug, Deserialize)]
pub struct Request {
    /// Absent for notifications (no response is sent).
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Serialize)]
pub struct Response {
    pub jsonrpc: &'static str,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

impl Response {
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn failure(id: Value, error: RpcError) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(error),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
}

impl RpcError {
    pub const PARSE_ERROR: i32 = -32700;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;
    /// Engine initialization failed (model/dictionary load error).
    pub const INIT_FAILED: i32 = -32000;

    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

// === Params ===

#[derive(Debug, Deserialize)]
pub struct ProcessKeyParams {
    /// XKB keysym value (e.g. 0x0061 = 'a', 0xff0d = Return).
    pub keysym: u32,
    /// Wire fields: `shift` / `control` / `alt` / `super`, all optional.
    #[serde(default)]
    pub modifiers: KeyModifiers,
    #[serde(default)]
    pub is_release: bool,
}

#[derive(Debug, Deserialize)]
pub struct SelectCandidateParams {
    /// 0-based index within the currently displayed page (0..page_size).
    pub page_index: usize,
}

#[derive(Debug, Deserialize)]
pub struct SurroundingTextParams {
    pub text: String,
    /// Cursor position within `text`, in Unicode scalar values.
    pub cursor_pos: usize,
}

// === Results ===

#[derive(Debug, Serialize)]
pub struct InitResult {
    pub protocol_version: u32,
    pub model_name: String,
}

/// Result of `process_key`, `select_candidate`, and `commit`.
#[derive(Debug, Serialize)]
pub struct KeyResult {
    pub consumed: bool,
    pub actions: Vec<Action>,
    /// Conversion (inference) time for this request in milliseconds;
    /// 0 when no conversion ran.
    pub conversion_ms: u64,
    /// End-to-end engine processing time for this request in milliseconds.
    pub process_key_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct StatusResult {
    pub initialized: bool,
    pub model_name: String,
    /// Current engine state: "empty", "composing", or "conversion".
    pub state: &'static str,
}

// === UI actions ===

/// UI update emitted by the engine, mirroring `EngineAction`.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    UpdatePreedit {
        text: String,
        /// Caret position in Unicode scalar values.
        caret: usize,
        attributes: Vec<PreeditAttr>,
    },
    /// Candidates for the currently visible page only; the engine handles
    /// paging internally (Page Up/Down keys move between pages).
    ShowCandidates {
        candidates: Vec<CandidateItem>,
        /// Selected index within this page.
        cursor: usize,
        /// Current page (0-based).
        page: usize,
        total_pages: usize,
        /// Columns of the grid layout (`[display] candidate_layout`): the
        /// page fills row-major, `candidates.len()` ≤ columns × rows.
        /// Absent for the classic vertical list.
        #[serde(skip_serializing_if = "Option::is_none")]
        grid_columns: Option<usize>,
    },
    HideCandidates,
    Commit {
        text: String,
    },
    UpdateAux {
        text: String,
    },
    HideAux,
}

#[derive(Debug, Serialize)]
pub struct PreeditAttr {
    /// Start position in Unicode scalar values (inclusive).
    pub start: usize,
    /// End position in Unicode scalar values (exclusive).
    pub end: usize,
    /// "underline" | "underline_double" | "highlight" | "reverse"
    pub style: &'static str,
}

#[derive(Debug, Serialize)]
pub struct CandidateItem {
    pub text: String,
    /// Mozc-style right-side comment (e.g. 三点リーダ). Unformatted; the
    /// frontend decides how to display it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}
