//! Candidate list management
//!
//! Handles the list of conversion candidates with pagination support.

/// Source of a conversion candidate — which subsystem produced it.
/// Presentation ([`label`](Self::label), [`is_deletable`](Self::is_deletable))
/// is derived from this on read, never stored, so it can't fall out of sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateSource {
    /// User dictionary lookup
    UserDictionary,
    /// Learning cache (user history)
    Learning,
    /// Model inference result
    Model,
    /// System dictionary lookup (also covers reading→symbol lookups via
    /// mozc's symbol.tsv — they're treated as just another dictionary).
    Dictionary,
    /// Rewriter-generated variant (half-width katakana, symbol)
    Rewriter,
    /// Date/time candidate rendered from the clock (`[date]` phrases)
    Date,
    /// Hiragana/katakana fallback
    Fallback,
}

impl CandidateSource {
    /// Aux-text label telling the user which subsystem produced the
    /// candidate. Empty for sources that aren't worth calling out (Fallback).
    pub fn label(&self) -> &'static str {
        match self {
            CandidateSource::UserDictionary => "\u{1F464} \u{30E6}\u{30FC}\u{30B6}\u{30FC}", // 👤 ユーザー
            CandidateSource::Learning => "\u{1F4DD} \u{5B66}\u{7FD2}", // 📝 学習
            CandidateSource::Model => "\u{1F916} AI",                  // 🤖 AI
            CandidateSource::Dictionary => "\u{1F4DA} \u{8F9E}\u{66F8}", // 📚 辞書
            CandidateSource::Rewriter => "\u{1F504} \u{5909}\u{63DB}", // 🔄 変換
            CandidateSource::Date => "\u{1F4C5} \u{65E5}\u{4ED8}",     // 📅 日付
            CandidateSource::Fallback => "",
        }
    }

    /// Short glyph for tight spots like the Ctrl+R filter header in the
    /// conversion aux text. Empty for Fallback, which has no view of its
    /// own — its kana ride at the tail of the rewriter view.
    pub fn emoji(&self) -> &'static str {
        match self {
            CandidateSource::UserDictionary => "\u{1F464}", // 👤
            CandidateSource::Learning => "\u{1F4DD}",       // 📝
            CandidateSource::Model => "\u{1F916}",          // 🤖
            CandidateSource::Dictionary => "\u{1F4DA}",     // 📚
            CandidateSource::Rewriter => "\u{1F504}",       // 🔄
            CandidateSource::Date => "\u{1F4C5}",           // 📅
            CandidateSource::Fallback => "",
        }
    }

    /// Whether candidates from this source can be removed from the learning
    /// history with Ctrl+Backspace / Ctrl+Delete.
    pub fn is_deletable(&self) -> bool {
        matches!(self, CandidateSource::Learning)
    }

    /// Whether committing a candidate from this source is recorded in the
    /// learning cache. Date/time candidates are rendered from the clock, so
    /// a recorded one would resurface later as a stale date.
    pub fn is_learnable(&self) -> bool {
        !matches!(self, CandidateSource::Date)
    }
}

/// A single conversion candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// The converted text
    pub text: String,
    /// The original reading (hiragana)
    pub reading: Option<String>,
    /// Which subsystem produced this candidate, if known.
    pub source: Option<CandidateSource>,
    /// Per-candidate description shown as the right-side comment on the
    /// candidate (mozc-style). Only set when the candidate itself has a
    /// meaningful description — symbol descriptions like `三点リーダ`,
    /// rewriter descriptions like `[全]英大文字`. Source labels are
    /// intentionally excluded so they don't duplicate the aux text.
    pub description: Option<String>,
}

impl Candidate {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            reading: None,
            source: None,
            description: None,
        }
    }

    pub fn with_reading(text: impl Into<String>, reading: impl Into<String>) -> Self {
        Self {
            reading: Some(reading.into()),
            ..Self::new(text)
        }
    }

    /// Aux-text source label (`🤖 AI`, `📚 辞書`, ...) derived from `source`;
    /// `None` when the source is unknown or has no label.
    pub fn source_label(&self) -> Option<&'static str> {
        self.source.map(|s| s.label()).filter(|l| !l.is_empty())
    }

    /// Whether this candidate can be removed from the learning history with
    /// Ctrl+Backspace / Ctrl+Delete. See [`CandidateSource::is_deletable`].
    pub fn is_deletable(&self) -> bool {
        self.source.is_some_and(|s| s.is_deletable())
    }
}

/// A list of candidates with pagination and selection support
#[derive(Debug, Clone)]
pub struct CandidateList {
    /// All candidates
    candidates: Vec<Candidate>,
    /// Currently selected candidate index
    cursor: usize,
    /// Number of candidates per page
    page_size: usize,
}

impl CandidateList {
    /// Default page size for candidate display
    pub const DEFAULT_PAGE_SIZE: usize = 9;

    /// Create a new candidate list
    pub fn new(candidates: Vec<Candidate>) -> Self {
        Self {
            candidates,
            cursor: 0,
            page_size: Self::DEFAULT_PAGE_SIZE,
        }
    }

    /// Create a candidate list with an explicit page size (the grid layout
    /// shows rows × columns per page). Clamped to at least 1 so pagination
    /// arithmetic never divides by zero.
    pub fn with_page_size(candidates: Vec<Candidate>, page_size: usize) -> Self {
        Self {
            page_size: page_size.max(1),
            ..Self::new(candidates)
        }
    }

    /// Create a candidate list from strings (test fixture).
    #[cfg(test)]
    pub fn from_strings(strings: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::new(strings.into_iter().map(Candidate::new).collect())
    }

    /// Get all candidates
    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }

    /// Get the number of candidates
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Get the current cursor position
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Get the page size
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Get the current page number (0-indexed)
    pub fn current_page(&self) -> usize {
        self.cursor.checked_div(self.page_size).unwrap_or(0)
    }

    /// Get the total number of pages
    pub fn total_pages(&self) -> usize {
        if self.page_size == 0 || self.candidates.is_empty() {
            0
        } else {
            self.candidates.len().div_ceil(self.page_size)
        }
    }

    /// Get the start index of the current page
    pub fn page_start(&self) -> usize {
        self.current_page() * self.page_size
    }

    /// Get the candidates for the current page
    pub fn page_candidates(&self) -> &[Candidate] {
        let start = self.page_start();
        let end = (start + self.page_size).min(self.candidates.len());
        &self.candidates[start..end]
    }

    /// Get the cursor position within the current page (0-indexed)
    pub fn page_cursor(&self) -> usize {
        self.cursor - self.page_start()
    }

    /// Get the currently selected candidate
    pub fn selected(&self) -> Option<&Candidate> {
        self.candidates.get(self.cursor)
    }

    /// Get the currently selected text
    pub fn selected_text(&self) -> Option<&str> {
        self.selected().map(|c| c.text.as_str())
    }

    /// Move to the next candidate
    pub fn move_next(&mut self) -> bool {
        if self.cursor + 1 < self.candidates.len() {
            self.cursor += 1;
            true
        } else if !self.candidates.is_empty() {
            // Wrap to beginning
            self.cursor = 0;
            true
        } else {
            false
        }
    }

    /// Move to the previous candidate
    pub fn move_prev(&mut self) -> bool {
        if self.cursor > 0 {
            self.cursor -= 1;
            true
        } else if !self.candidates.is_empty() {
            // Wrap to end
            self.cursor = self.candidates.len() - 1;
            true
        } else {
            false
        }
    }

    /// Grid layout: move one row down — the candidate `columns` further
    /// along, which is the cell below in a row-major grid. Crossing the last
    /// row of a page lands on the next page's first row in the same column
    /// (`page_size` is a multiple of `columns`, so the visual column is
    /// `cursor % columns` on every page); past the last candidate it wraps
    /// to the top of the same column.
    pub fn move_down(&mut self, columns: usize) -> bool {
        if self.candidates.is_empty() || columns == 0 {
            return false;
        }
        if self.cursor + columns < self.candidates.len() {
            self.cursor += columns;
        } else {
            self.cursor %= columns;
        }
        true
    }

    /// Grid layout: move one row up, wrapping to the bottom of the same
    /// column — the mirror of [`Self::move_down`].
    pub fn move_up(&mut self, columns: usize) -> bool {
        if self.candidates.is_empty() || columns == 0 {
            return false;
        }
        if self.cursor >= columns {
            self.cursor -= columns;
        } else {
            let column = self.cursor % columns;
            self.cursor = column + ((self.candidates.len() - 1 - column) / columns) * columns;
        }
        true
    }

    /// Grid layout: jump to the first cell of the cursor's row (Ctrl+A).
    /// Rows never span pages (`page_size` is a multiple of `columns`).
    pub fn move_row_start(&mut self, columns: usize) -> bool {
        if self.candidates.is_empty() || columns == 0 {
            return false;
        }
        self.cursor -= self.cursor % columns;
        true
    }

    /// Grid layout: jump to the last occupied cell of the cursor's row
    /// (Ctrl+E) — the row's final column, or the last candidate when the
    /// final row is partial.
    pub fn move_row_end(&mut self, columns: usize) -> bool {
        if self.candidates.is_empty() || columns == 0 {
            return false;
        }
        let row_start = self.cursor - self.cursor % columns;
        self.cursor = (row_start + columns - 1).min(self.candidates.len() - 1);
        true
    }

    /// Move to the next page
    pub fn next_page(&mut self) -> bool {
        if self.candidates.is_empty() {
            return false;
        }

        let next_page_start = self.page_start() + self.page_size;
        if next_page_start < self.candidates.len() {
            self.cursor = next_page_start;
            true
        } else {
            // Wrap to first page
            self.cursor = 0;
            true
        }
    }

    /// Move to the previous page
    pub fn prev_page(&mut self) -> bool {
        if self.candidates.is_empty() {
            return false;
        }

        let current_page = self.current_page();
        if current_page > 0 {
            self.cursor = (current_page - 1) * self.page_size;
            true
        } else {
            // Wrap to last page
            let last_page = self.total_pages().saturating_sub(1);
            self.cursor = last_page * self.page_size;
            true
        }
    }

    /// Select a candidate by index within the current page (1-9)
    pub fn select_on_page(&mut self, page_index: usize) -> Option<&Candidate> {
        if page_index == 0 || page_index > self.page_size {
            return None;
        }

        let absolute_index = self.page_start() + page_index - 1;
        if absolute_index < self.candidates.len() {
            self.cursor = absolute_index;
            self.selected()
        } else {
            None
        }
    }

    /// Move the cursor to `cursor`, clamped into the list (0 when empty).
    pub fn set_cursor(&mut self, cursor: usize) {
        self.cursor = cursor.min(self.candidates.len().saturating_sub(1));
    }
}

impl Default for CandidateList {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_list_basic() {
        let candidates = CandidateList::from_strings(["今日", "京", "恭"]);
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates.selected_text(), Some("今日"));
    }

    #[test]
    fn test_candidate_list_navigation() {
        let mut candidates = CandidateList::from_strings(["a", "b", "c"]);

        assert!(candidates.move_next());
        assert_eq!(candidates.selected_text(), Some("b"));

        assert!(candidates.move_next());
        assert_eq!(candidates.selected_text(), Some("c"));

        // Wrap around
        assert!(candidates.move_next());
        assert_eq!(candidates.selected_text(), Some("a"));

        // Wrap back
        assert!(candidates.move_prev());
        assert_eq!(candidates.selected_text(), Some("c"));
    }

    #[test]
    fn test_candidate_list_pagination() {
        // Default page_size is 9, so 20 items = 3 pages (9+9+2)
        let items: Vec<_> = (1..=20).map(|i| format!("item{}", i)).collect();
        let mut candidates = CandidateList::from_strings(items);

        assert_eq!(candidates.total_pages(), 3);
        assert_eq!(candidates.current_page(), 0);
        assert_eq!(candidates.page_candidates().len(), 9);

        candidates.next_page();
        assert_eq!(candidates.current_page(), 1);
        assert_eq!(candidates.page_start(), 9);

        candidates.next_page();
        assert_eq!(candidates.current_page(), 2);
        assert_eq!(candidates.page_candidates().len(), 2);

        // Wrap to first page
        candidates.next_page();
        assert_eq!(candidates.current_page(), 0);
    }

    /// 2×3-per-page grid list (page_size 6, columns 2) over `n` items.
    fn grid_list(n: usize) -> CandidateList {
        let items = (1..=n)
            .map(|i| Candidate::new(format!("item{}", i)))
            .collect();
        CandidateList::with_page_size(items, 6)
    }

    #[test]
    fn test_with_page_size_pagination() {
        let mut candidates = grid_list(8);
        assert_eq!(candidates.page_size(), 6);
        assert_eq!(candidates.total_pages(), 2);
        assert_eq!(candidates.page_candidates().len(), 6);
        candidates.next_page();
        assert_eq!(candidates.page_candidates().len(), 2);
        // Zero is clamped so pagination arithmetic stays defined.
        assert_eq!(CandidateList::with_page_size(Vec::new(), 0).page_size(), 1);
    }

    #[test]
    fn test_grid_move_down_and_up() {
        // 2 columns: item1 item2 / item3 item4 / item5 item6 | item7 item8
        let mut candidates = grid_list(8);

        assert!(candidates.move_down(2));
        assert_eq!(candidates.selected_text(), Some("item3"));

        // Crossing the page's last row lands on the next page, same column.
        candidates.set_cursor(4); // item5, bottom row of page 0
        assert!(candidates.move_down(2));
        assert_eq!(candidates.selected_text(), Some("item7"));
        assert_eq!(candidates.current_page(), 1);

        // Past the last candidate: wrap to the top of the same column.
        assert!(candidates.move_down(2));
        assert_eq!(candidates.selected_text(), Some("item1"));

        // move_up mirrors it: wrap to the bottom of the same column.
        assert!(candidates.move_up(2));
        assert_eq!(candidates.selected_text(), Some("item7"));
        assert!(candidates.move_up(2));
        assert_eq!(candidates.selected_text(), Some("item5"));

        // A column whose bottom row is missing wraps to its lowest cell:
        // column 1's last occupied row holds item8.
        candidates.set_cursor(1); // item2
        assert!(candidates.move_up(2));
        assert_eq!(candidates.selected_text(), Some("item8"));
    }

    #[test]
    fn test_grid_move_row_start_and_end() {
        // 2 columns: item1 item2 / item3 item4 / item5 item6 | item7 item8
        let mut candidates = grid_list(8);

        candidates.set_cursor(3); // item4
        assert!(candidates.move_row_start(2));
        assert_eq!(candidates.selected_text(), Some("item3"));
        assert!(candidates.move_row_end(2));
        assert_eq!(candidates.selected_text(), Some("item4"));

        // A partial final row ends at the last candidate.
        let mut candidates = grid_list(7);
        candidates.set_cursor(6); // item7, alone on its row
        assert!(candidates.move_row_end(2));
        assert_eq!(candidates.selected_text(), Some("item7"));
        assert!(candidates.move_row_start(2));
        assert_eq!(candidates.selected_text(), Some("item7"));
    }

    #[test]
    fn test_candidate_list_select_on_page() {
        let items: Vec<_> = (1..=20).map(|i| format!("item{}", i)).collect();
        let mut candidates = CandidateList::from_strings(items);

        // Select item 3 on first page
        candidates.select_on_page(3);
        assert_eq!(candidates.selected_text(), Some("item3"));

        // Move to second page and select item 2
        candidates.next_page();
        candidates.select_on_page(2);
        assert_eq!(candidates.selected_text(), Some("item11")); // 9 + 2 = 11
    }
}
