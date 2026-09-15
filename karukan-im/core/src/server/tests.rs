use serde_json::{Value, json};

use super::ImServer;
use crate::config::Settings;
use crate::core::keycode::Keysym;

// XKB keysyms for common keys (u32 aliases for the JSON payloads below)
const XKB_KEY_K: u32 = Keysym::KEY_K.0;
const XKB_KEY_A: u32 = Keysym::KEY_A.0;
const XKB_KEY_LOWER_L: u32 = Keysym::KEY_L.0;
const XKB_KEY_RETURN: u32 = Keysym::RETURN.0;
const XKB_KEY_ESCAPE: u32 = Keysym::ESCAPE.0;
const XKB_KEY_SPACE: u32 = Keysym::SPACE.0;

fn test_server() -> ImServer {
    let mut server = ImServer::with_settings(Settings::default());
    // Disable live conversion (Ctrl+Shift+L) so the preedit stays as
    // hiragana; live conversion would require a loaded model.
    request(
        &mut server,
        json!({"jsonrpc":"2.0","id":0,"method":"process_key","params":{
            "keysym": XKB_KEY_LOWER_L,
            "modifiers": {"control": true, "shift": true}
        }}),
    );
    server
}

/// Send a request value and return the parsed response.
fn request(server: &mut ImServer, req: Value) -> Value {
    let line = serde_json::to_string(&req).unwrap();
    let resp = server.handle_line(&line).expect("expected a response");
    serde_json::from_str(&resp).unwrap()
}

fn press(server: &mut ImServer, keysym: u32) -> Value {
    request(
        server,
        json!({"jsonrpc":"2.0","id":1,"method":"process_key","params":{"keysym": keysym}}),
    )
}

/// Collect actions of a given type from a process_key response.
fn actions_of<'a>(resp: &'a Value, ty: &str) -> Vec<&'a Value> {
    resp["result"]["actions"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|a| a["type"] == ty)
        .collect()
}

#[test]
fn test_parse_error() {
    let mut server = test_server();
    let resp: Value = serde_json::from_str(&server.handle_line("not json").unwrap()).unwrap();
    assert_eq!(resp["error"]["code"], -32700);
    assert_eq!(resp["id"], Value::Null);
}

#[test]
fn test_method_not_found() {
    let mut server = test_server();
    let resp = request(
        &mut server,
        json!({"jsonrpc":"2.0","id":1,"method":"no_such_method"}),
    );
    assert_eq!(resp["error"]["code"], -32601);
    assert_eq!(resp["id"], 1);
}

#[test]
fn test_invalid_params() {
    let mut server = test_server();
    let resp = request(
        &mut server,
        json!({"jsonrpc":"2.0","id":2,"method":"process_key","params":{"keysym":"x"}}),
    );
    assert_eq!(resp["error"]["code"], -32602);
}

#[test]
fn test_notification_gets_no_response() {
    let mut server = test_server();
    let line = serde_json::to_string(
        &json!({"jsonrpc":"2.0","method":"process_key","params":{"keysym": XKB_KEY_K}}),
    )
    .unwrap();
    assert!(server.handle_line(&line).is_none());
    // The notification was still executed: 'k' is buffered, so 'a' yields か
    let resp = press(&mut server, XKB_KEY_A);
    let preedits = actions_of(&resp, "update_preedit");
    assert_eq!(preedits.last().unwrap()["text"], "か");
}

#[test]
fn test_typing_and_commit() {
    let mut server = test_server();

    let resp = press(&mut server, XKB_KEY_K);
    assert_eq!(resp["result"]["consumed"], true);
    let preedits = actions_of(&resp, "update_preedit");
    assert_eq!(preedits.last().unwrap()["text"], "k");

    let resp = press(&mut server, XKB_KEY_A);
    let preedits = actions_of(&resp, "update_preedit");
    let preedit = preedits.last().unwrap();
    assert_eq!(preedit["text"], "か");
    assert_eq!(preedit["caret"], 1);

    let resp = press(&mut server, XKB_KEY_RETURN);
    let commits = actions_of(&resp, "commit");
    assert_eq!(commits.last().unwrap()["text"], "か");
    assert!(actions_of(&resp, "update_preedit").is_empty());
}

#[test]
fn test_escape_cancels_composition() {
    let mut server = test_server();
    press(&mut server, XKB_KEY_K);
    press(&mut server, XKB_KEY_A);
    let resp = press(&mut server, XKB_KEY_ESCAPE);
    assert_eq!(resp["result"]["consumed"], true);
    let preedits = actions_of(&resp, "update_preedit");
    assert_eq!(preedits.last().unwrap()["text"], "");

    // Key after cancel is not consumed in Empty state unless printable;
    // Escape itself in Empty state passes through.
    let resp = press(&mut server, XKB_KEY_ESCAPE);
    assert_eq!(resp["result"]["consumed"], false);
}

#[test]
fn test_explicit_commit_method() {
    let mut server = test_server();
    press(&mut server, XKB_KEY_K);
    press(&mut server, XKB_KEY_A);
    let resp = request(
        &mut server,
        json!({"jsonrpc":"2.0","id":7,"method":"commit"}),
    );
    let commits = actions_of(&resp, "commit");
    assert_eq!(commits.last().unwrap()["text"], "か");
    assert!(actions_of(&resp, "update_preedit").is_empty());

    // Nothing left to commit afterwards
    let resp = request(
        &mut server,
        json!({"jsonrpc":"2.0","id":8,"method":"commit"}),
    );
    assert!(actions_of(&resp, "commit").is_empty());
    assert_eq!(
        actions_of(&resp, "update_preedit").last().unwrap()["text"],
        ""
    );
}

#[test]
fn test_select_candidate_commits_page_candidate() {
    let mut server = test_server();
    press(&mut server, XKB_KEY_K);
    press(&mut server, XKB_KEY_A);

    // Space starts conversion; without a model the candidates come from
    // the hiragana/katakana fallback and the rewriter.
    let resp = press(&mut server, XKB_KEY_SPACE);
    let shows = actions_of(&resp, "show_candidates");
    let first_text = shows.last().unwrap()["candidates"][0]["text"].clone();

    let resp = request(
        &mut server,
        json!({"jsonrpc":"2.0","id":20,"method":"select_candidate","params":{"page_index":0}}),
    );
    assert_eq!(resp["result"]["consumed"], true);
    let commits = actions_of(&resp, "commit");
    assert_eq!(commits.last().unwrap()["text"], first_text);
    assert!(actions_of(&resp, "update_preedit").is_empty());
    assert!(!actions_of(&resp, "hide_candidates").is_empty());

    let resp = request(
        &mut server,
        json!({"jsonrpc":"2.0","id":21,"method":"status"}),
    );
    assert_eq!(resp["result"]["state"], "empty");
}

#[test]
fn test_grid_layout_rides_on_show_candidates() {
    // With the grid layout configured, every show_candidates carries the
    // column count for the frontend; the vertical default omits the field.
    let mut settings = Settings::default();
    settings.display.candidate_layout = crate::config::settings::CandidateLayout::Grid {
        columns: 5,
        rows: 5,
    };
    let mut server = ImServer::with_settings(settings);
    press(&mut server, XKB_KEY_K);
    press(&mut server, XKB_KEY_A);
    let resp = press(&mut server, XKB_KEY_SPACE);
    let shows = actions_of(&resp, "show_candidates");
    assert_eq!(shows.last().unwrap()["grid_columns"], 5);

    // Grid page size (25) admits page indexes the vertical list rejects.
    let resp = request(
        &mut server,
        json!({"jsonrpc":"2.0","id":30,"method":"select_candidate","params":{"page_index":9}}),
    );
    assert!(resp["error"].is_null());

    let mut server = test_server();
    press(&mut server, XKB_KEY_K);
    press(&mut server, XKB_KEY_A);
    let resp = press(&mut server, XKB_KEY_SPACE);
    let shows = actions_of(&resp, "show_candidates");
    assert!(shows.last().unwrap().get("grid_columns").is_none());
}

#[test]
fn test_select_candidate_out_of_range() {
    let mut server = test_server();
    let resp = request(
        &mut server,
        json!({"jsonrpc":"2.0","id":22,"method":"select_candidate","params":{"page_index":9}}),
    );
    assert_eq!(resp["error"]["code"], -32602);
}

#[test]
fn test_select_candidate_without_candidates_not_consumed() {
    let mut server = test_server();
    let resp = request(
        &mut server,
        json!({"jsonrpc":"2.0","id":23,"method":"select_candidate","params":{"page_index":0}}),
    );
    assert_eq!(resp["result"]["consumed"], false);
}

#[test]
fn test_reset_clears_state() {
    let mut server = test_server();
    press(&mut server, XKB_KEY_K);
    let resp = request(
        &mut server,
        json!({"jsonrpc":"2.0","id":9,"method":"reset"}),
    );
    assert!(resp["error"].is_null());
    let resp = request(
        &mut server,
        json!({"jsonrpc":"2.0","id":10,"method":"status"}),
    );
    assert_eq!(resp["result"]["state"], "empty");
    assert_eq!(resp["result"]["initialized"], false);
}

#[test]
fn test_set_surrounding_text() {
    let mut server = test_server();
    let resp = request(
        &mut server,
        json!({"jsonrpc":"2.0","id":11,"method":"set_surrounding_text",
               "params":{"text":"こんにちは世界","cursor_pos":5}}),
    );
    assert!(resp["error"].is_null());
}

#[test]
fn test_status_before_init() {
    let mut server = test_server();
    let resp = request(
        &mut server,
        json!({"jsonrpc":"2.0","id":12,"method":"status"}),
    );
    assert_eq!(resp["result"]["initialized"], false);
    assert_eq!(resp["result"]["state"], "empty");
}
