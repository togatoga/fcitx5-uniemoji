//! Stdio JSON-RPC server wrapping [`InputMethodEngine`].
//!
//! This is the platform-independent core of `karukan-imserver`, the engine
//! process spawned by the macOS (Swift/InputMethodKit) frontend. See
//! [`protocol`] for the wire format.

pub mod protocol;

use serde_json::{Value, json};

use crate::config::Settings;
use crate::core::engine::{EngineAction, EngineConfig, EngineResult, InputMethodEngine};
use crate::core::keycode::{KeyEvent, Keysym};
use crate::core::state::InputState;

use protocol::{
    Action, CandidateItem, InitResult, KeyResult, PROTOCOL_VERSION, PreeditAttr, ProcessKeyParams,
    Request, Response, RpcError, SelectCandidateParams, StatusResult, SurroundingTextParams,
};

/// JSON-RPC dispatcher owning the engine instance.
///
/// One server serves one frontend over one stdin/stdout pair; engine state
/// (Empty → Composing → Conversion) lives here across requests.
pub struct ImServer {
    engine: InputMethodEngine,
    /// Pending settings, consumed by the first successful `init`.
    settings: Option<Settings>,
    initialized: bool,
}

impl Default for ImServer {
    fn default() -> Self {
        Self::new()
    }
}

impl ImServer {
    /// Create a server with settings loaded from config.toml (or defaults).
    pub fn new() -> Self {
        let settings = Settings::load().unwrap_or_default();
        Self::with_settings(settings)
    }

    pub fn with_settings(settings: Settings) -> Self {
        let config = EngineConfig::from_settings(&settings);
        Self {
            engine: InputMethodEngine::with_config(config),
            settings: Some(settings),
            initialized: false,
        }
    }

    /// Save the learning cache (called on EOF/shutdown).
    pub fn save_learning(&mut self) {
        self.engine.save_learning();
    }

    /// Handle one request line. Returns the response line to write, or
    /// `None` for notifications (requests without an id).
    pub fn handle_line(&mut self, line: &str) -> Option<String> {
        let request: Request = match serde_json::from_str(line) {
            Ok(req) => req,
            Err(e) => {
                let resp = Response::failure(
                    Value::Null,
                    RpcError::new(RpcError::PARSE_ERROR, format!("parse error: {e}")),
                );
                return Some(serde_json::to_string(&resp).expect("response serialization"));
            }
        };

        let id = request.id.clone();
        let result = self.dispatch(&request.method, request.params);

        let id = id?; // notification: execute but don't respond
        let resp = match result {
            Ok(value) => Response::success(id, value),
            Err(error) => Response::failure(id, error),
        };
        Some(serde_json::to_string(&resp).expect("response serialization"))
    }

    fn dispatch(&mut self, method: &str, params: Value) -> Result<Value, RpcError> {
        match method {
            "init" => self.handle_init(),
            "process_key" => {
                let params: ProcessKeyParams = parse_params(params)?;
                let event =
                    KeyEvent::new(Keysym(params.keysym), params.modifiers, !params.is_release);
                let result = self.engine.process_key(&event);
                self.key_result(result)
            }
            "select_candidate" => {
                let params: SelectCandidateParams = parse_params(params)?;
                if params.page_index >= self.engine.candidate_page_size() {
                    return Err(RpcError::new(
                        RpcError::INVALID_PARAMS,
                        format!("page_index out of range: {}", params.page_index),
                    ));
                }
                let result = self.engine.select_candidate_on_page(params.page_index);
                self.key_result(result)
            }
            "commit" => {
                let result = self.engine.commit_result();
                self.key_result(result)
            }
            "reset" => {
                self.engine.reset();
                Ok(json!({}))
            }
            "set_surrounding_text" => {
                let params: SurroundingTextParams = parse_params(params)?;
                self.engine
                    .set_surrounding_text_at(&params.text, params.cursor_pos);
                Ok(json!({}))
            }
            "save_learning" => {
                self.engine.save_learning();
                Ok(json!({}))
            }
            "status" => {
                let state = match self.engine.state() {
                    InputState::Empty => "empty",
                    InputState::Composing { .. } => "composing",
                    InputState::Conversion { .. } => "conversion",
                };
                serde_json::to_value(StatusResult {
                    initialized: self.initialized,
                    model_name: self.engine.model_name(),
                    state,
                })
                .map_err(internal_error)
            }
            other => Err(RpcError::new(
                RpcError::METHOD_NOT_FOUND,
                format!("method not found: {other}"),
            )),
        }
    }

    fn handle_init(&mut self) -> Result<Value, RpcError> {
        if !self.initialized {
            let settings = self
                .settings
                .take()
                .unwrap_or_else(|| Settings::load().unwrap_or_default());
            if let Err(e) = self.engine.init_from_settings(&settings) {
                // Keep the settings so a retried `init` uses the same ones.
                self.settings = Some(settings);
                return Err(RpcError::new(RpcError::INIT_FAILED, format!("{e:#}")));
            }
            self.initialized = true;
        }
        serde_json::to_value(InitResult {
            protocol_version: PROTOCOL_VERSION,
            model_name: self.engine.model_name(),
        })
        .map_err(internal_error)
    }

    fn key_result(&self, result: EngineResult) -> Result<Value, RpcError> {
        let grid_columns = self.engine.candidate_grid_columns();
        let actions = result
            .actions
            .into_iter()
            .map(|a| to_action(a, grid_columns))
            .collect();
        serde_json::to_value(KeyResult {
            consumed: result.consumed,
            actions,
            conversion_ms: self.engine.last_conversion_ms(),
            process_key_ms: self.engine.last_process_key_ms(),
        })
        .map_err(internal_error)
    }
}

fn parse_params<T: serde::de::DeserializeOwned>(params: Value) -> Result<T, RpcError> {
    serde_json::from_value(params)
        .map_err(|e| RpcError::new(RpcError::INVALID_PARAMS, format!("invalid params: {e}")))
}

fn internal_error(e: serde_json::Error) -> RpcError {
    RpcError::new(RpcError::INTERNAL_ERROR, format!("internal error: {e}"))
}

fn to_action(action: EngineAction, grid_columns: Option<usize>) -> Action {
    use crate::core::preedit::AttributeType;

    match action {
        EngineAction::UpdatePreedit(preedit) => Action::UpdatePreedit {
            caret: preedit.caret(),
            attributes: preedit
                .attributes()
                .iter()
                .map(|a| PreeditAttr {
                    start: a.start,
                    end: a.end,
                    style: match a.attr_type {
                        AttributeType::Underline => "underline",
                        AttributeType::UnderlineDouble => "underline_double",
                        AttributeType::Highlight => "highlight",
                        AttributeType::Reverse => "reverse",
                    },
                })
                .collect(),
            text: preedit.text().to_string(),
        },
        EngineAction::ShowCandidates(list) => Action::ShowCandidates {
            candidates: list
                .page_candidates()
                .iter()
                .map(|c| CandidateItem {
                    text: c.text.clone(),
                    description: c.description.clone().filter(|s| !s.is_empty()),
                })
                .collect(),
            cursor: list.page_cursor(),
            page: list.current_page(),
            total_pages: list.total_pages(),
            grid_columns,
        },
        EngineAction::HideCandidates => Action::HideCandidates,
        EngineAction::Commit(text) => Action::Commit { text },
        EngineAction::UpdateAuxText(text) => Action::UpdateAux { text },
        EngineAction::HideAuxText => Action::HideAux,
    }
}

#[cfg(test)]
mod tests;
