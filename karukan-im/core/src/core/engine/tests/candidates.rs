use super::*;

// --- Candidate preservation tests ---

#[test]
fn test_live_text_preserved_in_conversion_via_down() {
    // When DOWN is pressed during live conversion, the AI inference result
    // (live_conversion_text) should appear in the candidate list.
    let mut engine = make_live_conversion_engine();

    // Simulate typing "あい" with live conversion active
    engine.process_key(&press('a'));
    engine.process_key(&press('i'));
    set_live_text(&mut engine, "愛");

    // Press DOWN → start_conversion()
    let result = engine.process_key(&press_key(Keysym::DOWN));
    assert!(result.consumed);
    assert!(matches!(engine.state(), InputState::Conversion { .. }));

    // The candidate list should contain "愛"
    let candidates = engine.state().candidates().unwrap();
    assert!(
        candidates.candidates().iter().any(|c| c.text == "愛"),
        "AI inference result '愛' should be in the candidate list"
    );
}

#[test]
fn test_live_text_not_duplicated_in_conversion() {
    // If the live_text matches the reading, it should not be duplicated
    let mut engine = make_live_conversion_engine();

    engine.process_key(&press('a'));
    engine.process_key(&press('i'));
    // live_conversion_text same as hiragana reading → should not be added
    set_live_text(&mut engine, "あい");

    let result = engine.process_key(&press_key(Keysym::DOWN));
    assert!(result.consumed);
    assert!(matches!(engine.state(), InputState::Conversion { .. }));

    // "あい" should not appear twice (it's same as reading, so live_text is skipped)
    let candidates = engine.state().candidates().unwrap();
    let count = candidates
        .candidates()
        .iter()
        .filter(|c| c.text == "あい")
        .count();
    assert_eq!(count, 1, "Reading should appear exactly once");
}

#[test]
fn test_suggest_result_preserved_in_start_conversion() {
    // When Space is pressed, the previous auto-suggest/live conversion result
    // should be preserved in the candidate list even if re-inference doesn't produce it.
    // (Without a kanji converter, build_conversion_candidates returns fallback only,
    // so the live_conversion_text would be lost without the preservation logic.)
    let mut engine = InputMethodEngine::new();

    engine.process_key(&press('a'));
    engine.process_key(&press('i'));
    set_live_text(&mut engine, "愛");

    // Press Space → start_conversion()
    let result = engine.process_key(&press_key(Keysym::SPACE));
    assert!(result.consumed);
    assert!(matches!(engine.state(), InputState::Conversion { .. }));

    // "愛" should be preserved in the candidate list
    let candidates = engine.state().candidates().unwrap();
    assert!(
        candidates.candidates().iter().any(|c| c.text == "愛"),
        "Previous suggest result '愛' should be preserved in candidates"
    );
}

#[test]
fn test_empty_live_text_not_added_to_candidates() {
    // When live_conversion_text is empty, no extra candidate should be added
    let mut engine = make_live_conversion_engine();

    engine.process_key(&press('a'));
    engine.process_key(&press('i'));
    // Force empty to test the "no live text" scenario
    engine.live.shown = false;

    // DOWN → start_conversion()
    let result = engine.process_key(&press_key(Keysym::DOWN));
    assert!(result.consumed);

    // Should have candidates but no empty-string candidate
    if let Some(candidates) = engine.state().candidates() {
        assert!(
            !candidates.candidates().iter().any(|c| c.text.is_empty()),
            "Empty candidate should not be in the list"
        );
    }
}

// --- Grid candidate layout ---

/// Engine with a 2×2 grid candidate window, no model.
fn grid_engine() -> InputMethodEngine {
    InputMethodEngine::with_config(EngineConfig {
        candidate_layout: crate::config::settings::CandidateLayout::Grid {
            columns: 2,
            rows: 2,
        },
        ..EngineConfig::default()
    })
}

/// Put `n` candidates c1..cn on a conversion, paged at the engine's
/// configured page size.
fn enter_grid_conversion(engine: &mut InputMethodEngine, n: usize) {
    let items = (1..=n).map(|i| Candidate::new(format!("c{}", i))).collect();
    engine.state = InputState::Conversion {
        preedit: Preedit::new(),
        candidates: CandidateList::with_page_size(items, engine.config.candidate_page_size()),
        reading: "あい".to_string(),
        filter: None,
    };
}

fn cursor_of(engine: &InputMethodEngine) -> usize {
    engine.state().candidates().unwrap().cursor()
}

#[test]
fn test_grid_arrow_and_emacs_navigation() {
    // 2 columns: c1 c2 / c3 c4 | c5 c6 / c7 c8
    let mut engine = grid_engine();
    enter_grid_conversion(&mut engine, 8);

    // ↓ moves a row down, not to the next candidate.
    engine.process_key(&press_key(Keysym::DOWN));
    assert_eq!(cursor_of(&engine), 2);

    // Ctrl+F walks the row; Ctrl+N drops a row, crossing onto page 2.
    engine.process_key(&press_ctrl(Keysym::KEY_F));
    assert_eq!(cursor_of(&engine), 3);
    engine.process_key(&press_ctrl(Keysym::KEY_N));
    assert_eq!(cursor_of(&engine), 5);
    assert_eq!(engine.state().candidates().unwrap().current_page(), 1);

    // Ctrl+P and Ctrl+B mirror them.
    engine.process_key(&press_ctrl(Keysym::KEY_P));
    assert_eq!(cursor_of(&engine), 3);
    engine.process_key(&press_ctrl(Keysym::KEY_B));
    assert_eq!(cursor_of(&engine), 2);

    // Bare ←/→ navigate too (vertical drops back to caret editing here).
    engine.process_key(&press_key(Keysym::LEFT));
    assert_eq!(cursor_of(&engine), 1);
    engine.process_key(&press_key(Keysym::RIGHT));
    assert_eq!(cursor_of(&engine), 2);
    assert!(matches!(engine.state(), InputState::Conversion { .. }));

    // ↑ from the second row returns to the top.
    engine.process_key(&press_key(Keysym::UP));
    assert_eq!(cursor_of(&engine), 0);

    // Ctrl+E / Ctrl+A jump to the row's last / first cell.
    engine.process_key(&press_ctrl(Keysym::KEY_E));
    assert_eq!(cursor_of(&engine), 1);
    engine.process_key(&press_ctrl(Keysym::KEY_A));
    assert_eq!(cursor_of(&engine), 0);
    assert!(matches!(engine.state(), InputState::Conversion { .. }));
}

#[test]
fn test_grid_conversion_page_size_and_wire_columns() {
    // A real conversion (no model: fallback + rewriter candidates) pages at
    // columns × rows, and the engine reports the columns for the frontend.
    let mut engine = grid_engine();
    engine.process_key(&press('a'));
    engine.process_key(&press('i'));
    engine.process_key(&press_key(Keysym::SPACE));

    let candidates = engine.state().candidates().unwrap();
    assert_eq!(candidates.page_size(), 4);
    assert_eq!(engine.candidate_grid_columns(), Some(2));
    assert_eq!(engine.candidate_page_size(), 4);
}

#[test]
fn test_vertical_keeps_ctrl_f_as_caret_edit() {
    // In the default vertical layout Ctrl+F still drops back to composing
    // (caret move), and ↓ stays next-candidate.
    let mut engine = InputMethodEngine::new();
    engine.process_key(&press('a'));
    engine.process_key(&press('i'));
    engine.process_key(&press_key(Keysym::SPACE));
    assert!(matches!(engine.state(), InputState::Conversion { .. }));

    engine.process_key(&press_key(Keysym::DOWN));
    assert_eq!(cursor_of(&engine), 1);

    engine.process_key(&press_ctrl(Keysym::KEY_F));
    assert!(matches!(engine.state(), InputState::Composing { .. }));
}
