import Cocoa

/// Custom candidate window (borderless non-activating NSPanel).
///
/// The engine pre-paginates: `show` receives only the visible page plus
/// page metadata, so this controller just renders it — as a numbered
/// vertical list, or as a row-major grid when the engine sends
/// `grid_columns`. An optional aux line (reading hint / model info from
/// the engine) is shown as a footer.
///
/// Rendering is incremental: a cursor-only change restyles the affected
/// cells in place (and moves the grid highlight), an aux-only change
/// rewrites the footer label — walking the candidates never tears the
/// view tree down, which is what used to flicker. The stack is pinned to
/// the panel's top only and the panel is sized to the measured content,
/// so a sizing mismatch can never stretch gaps open inside the window.
class CandidateWindowController {
    // Visual scale of the panel. Candidate rows use a larger type size
    // than the footers (page indicator / aux line), matching the system
    // Japanese IME's proportions.
    private static let candidateFontSize: CGFloat = 18
    private static let footerFontSize: CGFloat = 13
    private static let minPanelWidth: CGFloat = 160
    // Grid metrics. The cell padding is what the selection highlight
    // covers around the text — the highlight hugs the cell's own text,
    // never the column's full width, so a short candidate in a wide
    // column doesn't drag trailing blank into its selection.
    private static let gridColumnSpacing: CGFloat = 16
    private static let gridRowSpacing: CGFloat = 4
    private static let gridCellPaddingX: CGFloat = 5
    private static let gridCellPaddingY: CGFloat = 1

    private let panel: NSPanel
    private let stackView: NSStackView
    private var rowViews: [NSView] = []
    /// One label per candidate on the page, in candidate order — restyled
    /// in place when only the selection moves.
    private var cellLabels: [NSTextField] = []
    /// Grid only: the highlight view behind the selected cell.
    private var gridHighlight: NSView?
    private var auxLabel: NSTextField?
    private var auxText: String?

    private struct PageState: Equatable {
        let candidates: [CandidateItem]
        let cursor: Int
        let page: Int
        let totalPages: Int
        /// Grid layout column count from the engine; nil = vertical list.
        let gridColumns: Int?

        var isGrid: Bool { (gridColumns ?? 1) > 1 }
    }
    private var pageState: PageState?
    /// What the on-screen views were built from; nil after hide.
    private var renderedState: PageState?
    private var renderedAux: String?

    init() {
        panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 200, height: 100),
            styleMask: [.nonactivatingPanel, .borderless],
            backing: .buffered,
            defer: true
        )
        panel.level = .popUpMenu
        panel.hidesOnDeactivate = false
        panel.isOpaque = false
        panel.backgroundColor = NSColor.windowBackgroundColor
        panel.ignoresMouseEvents = true

        stackView = NSStackView()
        stackView.orientation = .vertical
        stackView.alignment = .leading
        stackView.spacing = 4
        stackView.edgeInsets = NSEdgeInsets(top: 8, left: 12, bottom: 8, right: 12)
        stackView.translatesAutoresizingMaskIntoConstraints = false

        panel.contentView?.addSubview(stackView)
        if let contentView = panel.contentView {
            // Top/leading/trailing only: the stack keeps its own fitting
            // height. Pinning the bottom too would stretch the content
            // whenever the panel is taller than the stack's own layout,
            // opening blank gaps between the rows and the footers.
            NSLayoutConstraint.activate([
                stackView.topAnchor.constraint(equalTo: contentView.topAnchor),
                stackView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
                stackView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            ])
        }
    }

    var isVisible: Bool { panel.isVisible }

    /// `cursorRect: nil` reuses the rect from the previous `show` — the
    /// caller can skip its (synchronous, per-keystroke) client IPC while
    /// the panel is already on screen, since the composition anchor
    /// doesn't move mid-composition.
    func show(
        candidates: [CandidateItem], cursor: Int, page: Int, totalPages: Int,
        gridColumns: Int?, cursorRect: NSRect?
    ) {
        pageState = PageState(
            candidates: candidates, cursor: cursor, page: page, totalPages: totalPages,
            gridColumns: gridColumns)
        render(cursorRect: cursorRect)
    }

    /// Update the aux footer; re-renders in place if the window is visible.
    /// Pass `deferRender: true` when a `show`/`hide` follows in the same
    /// action batch, so the panel is rendered once per batch instead of
    /// once for the aux change and again for the candidates.
    func setAux(_ text: String?, deferRender: Bool = false) {
        auxText = text
        if !deferRender, panel.isVisible, pageState != nil {
            render(cursorRect: nil)
        }
    }

    func hide() {
        pageState = nil
        renderedState = nil
        renderedAux = nil
        panel.orderOut(nil)
    }

    private func render(cursorRect: NSRect?) {
        guard let state = pageState else {
            hide()
            return
        }
        // An empty list still renders while the aux footer has text: a
        // source-filtered view narrowed to an empty source must show its
        // 「候補なし」 footer, not a silently vanishing panel.
        if state.candidates.isEmpty && (auxText ?? "").isEmpty {
            hide()
            return
        }

        // Incremental path: same candidates, only the cursor and/or aux
        // text changed. Restyle in place instead of rebuilding, so
        // navigation doesn't flicker.
        if let rendered = renderedState, panel.isVisible,
            rendered.candidates == state.candidates,
            rendered.gridColumns == state.gridColumns,
            rendered.page == state.page,
            rendered.totalPages == state.totalPages,
            (renderedAux ?? "").isEmpty == (auxText ?? "").isEmpty
        {
            if rendered.cursor != state.cursor {
                restyleCell(at: rendered.cursor, in: state)
                restyleCell(at: state.cursor, in: state)
                if state.isGrid {
                    moveGridHighlight(to: state.cursor)
                }
            }
            if renderedAux != auxText, let label = auxLabel, let aux = auxText {
                label.stringValue = aux
                // A longer aux line may need a wider panel.
                positionPanel(cursorRect: cursorRect)
            }
            renderedState = state
            renderedAux = auxText
            return
        }

        clearRows()
        if state.isGrid, let columns = state.gridColumns, !state.candidates.isEmpty {
            addCandidateGrid(state.candidates, columns: columns, cursor: state.cursor)
        } else {
            for (index, candidate) in state.candidates.enumerated() {
                addCandidateRow(candidate, number: index + 1, selected: index == state.cursor)
            }
        }
        if state.totalPages > 1 {
            addFooterLabel("[\(state.page + 1)/\(state.totalPages)]")
        }
        if let aux = auxText, !aux.isEmpty {
            auxLabel = addFooterLabel(aux)
        }
        renderedState = state
        renderedAux = auxText

        positionPanel(cursorRect: cursorRect)
    }

    private func clearRows() {
        for view in rowViews {
            stackView.removeArrangedSubview(view)
            view.removeFromSuperview()
        }
        rowViews.removeAll()
        cellLabels.removeAll()
        gridHighlight = nil
        auxLabel = nil
    }

    private func addCandidateRow(_ candidate: CandidateItem, number: Int, selected: Bool) {
        let label = NSTextField(labelWithString: "")
        label.translatesAutoresizingMaskIntoConstraints = false
        applyStyle(label, candidate: candidate, number: number, selected: selected, inGrid: false)
        cellLabels.append(label)
        stackView.addArrangedSubview(label)
        rowViews.append(label)
    }

    /// Grid layout: the page's candidates laid out row-major at `columns`
    /// cells per row, with explicitly measured frames — the controller
    /// computes column widths itself (NSGridView's fitting size inside the
    /// stack proved unreliable, opening huge blank areas) and hands the
    /// grid its exact size. Cells carry no number: the grid holds more
    /// candidates than digit selection can reach, so a partial numbering
    /// would mislead.
    private func addCandidateGrid(_ candidates: [CandidateItem], columns: Int, cursor: Int) {
        let grid = CandidateGridView()
        grid.translatesAutoresizingMaskIntoConstraints = false

        // The highlight sits behind the labels and is moved — not
        // rebuilt — when the cursor changes.
        let highlight = NSView()
        highlight.wantsLayer = true
        highlight.layer?.cornerRadius = 4
        grid.addSubview(highlight)
        gridHighlight = highlight

        for (index, candidate) in candidates.enumerated() {
            let label = NSTextField(labelWithString: "")
            applyStyle(label, candidate: candidate, number: nil, selected: index == cursor,
                inGrid: true)
            grid.addSubview(label)
            cellLabels.append(label)
        }
        layoutGrid(grid, columns: columns)
        moveGridHighlight(to: cursor)

        stackView.addArrangedSubview(grid)
        rowViews.append(grid)
    }

    /// Measure every cell and place the labels: column width = the widest
    /// cell in that column, uniform row height. The grid's intrinsic size
    /// is set from the result, which is what the stack (and the panel)
    /// size themselves from.
    private func layoutGrid(_ grid: CandidateGridView, columns: Int) {
        let sizes = cellLabels.map { $0.fittingSize }
        var columnWidths = [CGFloat](repeating: 0, count: columns)
        for (index, size) in sizes.enumerated() {
            let column = index % columns
            columnWidths[column] = max(columnWidths[column], size.width)
        }
        let textHeight = sizes.map(\.height).max() ?? 0
        let rowHeight = textHeight + Self.gridCellPaddingY * 2
        let rows = (cellLabels.count + columns - 1) / columns

        var columnX = [CGFloat](repeating: 0, count: columns)
        var x: CGFloat = 0
        for column in 0..<columns {
            columnX[column] = x
            x += Self.gridCellPaddingX * 2 + columnWidths[column] + Self.gridColumnSpacing
        }
        let totalWidth = x - Self.gridColumnSpacing

        for (index, label) in cellLabels.enumerated() {
            let column = index % columns
            let row = index / columns
            label.frame = NSRect(
                x: columnX[column] + Self.gridCellPaddingX,
                y: CGFloat(row) * (rowHeight + Self.gridRowSpacing) + Self.gridCellPaddingY,
                width: sizes[index].width,
                height: textHeight
            )
        }
        grid.contentSize = NSSize(
            width: totalWidth,
            height: CGFloat(rows) * rowHeight + CGFloat(max(rows - 1, 0)) * Self.gridRowSpacing
        )
    }

    /// Put the grid highlight behind cell `index`: the label's frame plus
    /// the cell padding, i.e. exactly the text the user is selecting.
    private func moveGridHighlight(to index: Int) {
        guard let highlight = gridHighlight, index < cellLabels.count else { return }
        highlight.layer?.backgroundColor = NSColor.selectedContentBackgroundColor.cgColor
        highlight.frame = cellLabels[index].frame.insetBy(
            dx: -Self.gridCellPaddingX, dy: -Self.gridCellPaddingY)
    }

    private func restyleCell(at index: Int, in state: PageState) {
        guard index < cellLabels.count, index < state.candidates.count else { return }
        applyStyle(
            cellLabels[index], candidate: state.candidates[index],
            number: state.isGrid ? nil : index + 1,
            selected: index == state.cursor, inGrid: state.isGrid)
    }

    /// Style one cell for its (un)selected state: `number` prefixes the
    /// vertical list's rows, nil for grid cells. Restyling in place is
    /// what lets a cursor move skip the full rebuild.
    private func applyStyle(
        _ label: NSTextField, candidate: CandidateItem, number: Int?, selected: Bool,
        inGrid: Bool
    ) {
        let prefix = number.map { "\($0). " } ?? ""
        let text = NSMutableAttributedString(
            string: "\(prefix)\(candidate.text)",
            attributes: [
                .font: NSFont.systemFont(ofSize: Self.candidateFontSize),
                .foregroundColor: selected ? NSColor.white : NSColor.labelColor,
            ]
        )
        if let description = candidate.description {
            text.append(
                NSAttributedString(
                    string: "  \(description)",
                    attributes: [
                        .font: NSFont.systemFont(ofSize: Self.footerFontSize),
                        .foregroundColor: selected
                            ? NSColor.white.withAlphaComponent(0.8)
                            : NSColor.secondaryLabelColor,
                    ]
                ))
        }
        label.attributedStringValue = text
        // The vertical list highlights by painting the label itself; grid
        // cells sit over the separate highlight view instead.
        if selected && !inGrid {
            label.backgroundColor = NSColor.selectedContentBackgroundColor
            label.drawsBackground = true
        } else {
            label.backgroundColor = .clear
            label.drawsBackground = false
        }
    }

    @discardableResult
    private func addFooterLabel(_ text: String) -> NSTextField {
        let label = NSTextField(labelWithString: text)
        label.font = NSFont.systemFont(ofSize: Self.footerFontSize)
        label.textColor = NSColor.secondaryLabelColor
        label.translatesAutoresizingMaskIntoConstraints = false
        stackView.addArrangedSubview(label)
        rowViews.append(label)
        return label
    }

    private var lastCursorRect: NSRect = .zero

    private func positionPanel(cursorRect: NSRect?) {
        if let rect = cursorRect {
            lastCursorRect = rect
        }
        let cursorRect = lastCursorRect

        stackView.layoutSubtreeIfNeeded()
        let contentSize = stackView.fittingSize
        let panelWidth = max(contentSize.width, Self.minPanelWidth)
        let panelHeight = contentSize.height

        guard cursorRect != .zero else {
            setPanelFrame(NSRect(x: 100, y: 100, width: panelWidth, height: panelHeight))
            return
        }

        // Flip above the cursor when the panel would fall off the bottom of
        // the screen.
        let showAbove: Bool
        if let screen = NSScreen.main {
            showAbove = cursorRect.origin.y - panelHeight < screen.visibleFrame.origin.y
        } else {
            showAbove = false
        }

        let originY: CGFloat
        if showAbove {
            originY = cursorRect.origin.y + cursorRect.size.height
        } else {
            originY = cursorRect.origin.y - panelHeight
        }

        setPanelFrame(
            NSRect(x: cursorRect.origin.x, y: originY, width: panelWidth, height: panelHeight))
    }

    /// Apply the frame only when it actually changed — an unchanged frame
    /// redisplayed every keystroke is visible as flicker.
    private func setPanelFrame(_ frame: NSRect) {
        if panel.frame != frame {
            panel.setFrame(frame, display: true)
        }
        panel.orderFront(nil)
    }
}

/// Flipped container for grid cells laid out with explicit frames. Its
/// intrinsic size is set by the controller's own measurement, so Auto
/// Layout can neither stretch nor misreport the grid.
private final class CandidateGridView: NSView {
    var contentSize: NSSize = .zero {
        didSet { invalidateIntrinsicContentSize() }
    }
    override var isFlipped: Bool { true }
    override var intrinsicContentSize: NSSize { contentSize }
}
