import Cocoa

/// A candidate row that paints its own selection background across its full
/// bounds. Rows are given a uniform width (the widest row on the page), so a
/// selected row's highlight is a clean bar rather than one that hugs the text.
private final class RowView: NSView {
    var fillColor: NSColor? {
        didSet { needsDisplay = true }
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard let fillColor else { return }
        fillColor.setFill()
        bounds.fill()
    }
}

/// Custom candidate window (borderless non-activating NSPanel).
///
/// The engine pre-paginates: `show` receives only the visible page plus
/// page metadata, so this controller just renders rows. An optional aux
/// line (reading hint / model info from the engine) is shown as a footer.
///
/// Each row places the number, candidate text, and (optional) description at
/// fixed x offsets computed from the widest number and widest candidate on the
/// page. Because the offsets are the same on every row, the candidate and
/// description columns line up regardless of number width — `1.` and `2.`
/// render at different widths in the proportional system font, which would
/// otherwise make the candidate text start at a different x per row.
class CandidateWindowController {
    // Visual scale of the panel. Candidate rows use a larger type size
    // than the footers (page indicator / aux line), matching the system
    // Japanese IME's proportions.
    private static let candidateFontSize: CGFloat = 18
    private static let footerFontSize: CGFloat = 13
    private static let minPanelWidth: CGFloat = 160
    // Horizontal gaps between the aligned columns (points).
    private static let numberGap: CGFloat = 6
    private static let descriptionGap: CGFloat = 16
    // Uniform height for every candidate row.
    private static let rowHeight: CGFloat = 25

    private let panel: NSPanel
    private let stackView: NSStackView
    private var rowViews: [NSView] = []
    private var auxText: String?

    private struct PageState {
        let candidates: [CandidateItem]
        let cursor: Int
        let page: Int
        let totalPages: Int
    }
    private var pageState: PageState?

    /// Precomputed column geometry shared by every row on a page. All values
    /// are x offsets (points) from the row's leading edge.
    private struct ColumnLayout {
        let candidateX: CGFloat
        let descriptionX: CGFloat?
        let rowWidth: CGFloat
    }

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
            NSLayoutConstraint.activate([
                stackView.topAnchor.constraint(equalTo: contentView.topAnchor),
                stackView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
                stackView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
                stackView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
            ])
        }
    }

    var isVisible: Bool { panel.isVisible }

    /// `cursorRect: nil` reuses the rect from the previous `show` — the
    /// caller can skip its (synchronous, per-keystroke) client IPC while
    /// the panel is already on screen, since the composition anchor
    /// doesn't move mid-composition.
    func show(
        candidates: [CandidateItem], cursor: Int, page: Int, totalPages: Int, cursorRect: NSRect?
    ) {
        pageState = PageState(
            candidates: candidates, cursor: cursor, page: page, totalPages: totalPages)
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
        panel.orderOut(nil)
    }

    private func render(cursorRect: NSRect?) {
        clearRows()
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

        let layout = columnLayout(for: state.candidates)
        for (index, candidate) in state.candidates.enumerated() {
            addCandidateRow(
                candidate, number: index + 1, selected: index == state.cursor, layout: layout)
        }
        if state.totalPages > 1 {
            addFooterLabel("[\(state.page + 1)/\(state.totalPages)]")
        }
        if let aux = auxText, !aux.isEmpty {
            addFooterLabel(aux)
        }

        positionPanel(cursorRect: cursorRect)
    }

    private func clearRows() {
        for view in rowViews {
            stackView.removeArrangedSubview(view)
            view.removeFromSuperview()
        }
        rowViews.removeAll()
    }

    /// Measure the page's columns. The candidate column starts past the widest
    /// `N.`, and the description column (when any candidate has one) starts past
    /// the widest candidate text. These offsets are shared by every row, which
    /// is what keeps the columns aligned.
    private func columnLayout(for candidates: [CandidateItem]) -> ColumnLayout {
        let candidateFont = NSFont.systemFont(ofSize: Self.candidateFontSize)
        let descriptionFont = NSFont.systemFont(ofSize: Self.footerFontSize)

        func width(_ string: String, _ font: NSFont) -> CGFloat {
            ceil((string as NSString).size(withAttributes: [.font: font]).width)
        }

        let numberWidth = (1...candidates.count).map { width("\($0).", candidateFont) }.max() ?? 0
        let candidateWidth = candidates.map { width($0.text, candidateFont) }.max() ?? 0
        let descriptions = candidates.compactMap { $0.description }
        let descriptionWidth = descriptions.map { width($0, descriptionFont) }.max() ?? 0

        let candidateX = numberWidth + Self.numberGap
        if descriptions.isEmpty {
            return ColumnLayout(
                candidateX: candidateX, descriptionX: nil,
                rowWidth: candidateX + candidateWidth)
        }
        let descriptionX = candidateX + candidateWidth + Self.descriptionGap
        return ColumnLayout(
            candidateX: candidateX, descriptionX: descriptionX,
            rowWidth: descriptionX + descriptionWidth)
    }

    private func addCandidateRow(
        _ candidate: CandidateItem, number: Int, selected: Bool, layout: ColumnLayout
    ) {
        let row = RowView()
        row.translatesAutoresizingMaskIntoConstraints = false
        row.fillColor = selected ? NSColor.selectedContentBackgroundColor : nil

        let textColor: NSColor = selected ? .white : .labelColor
        let numberLabel = makeLabel(
            "\(number).", size: Self.candidateFontSize, color: textColor)
        let candidateLabel = makeLabel(
            candidate.text, size: Self.candidateFontSize, color: textColor)
        row.addSubview(numberLabel)
        row.addSubview(candidateLabel)

        var constraints: [NSLayoutConstraint] = [
            row.heightAnchor.constraint(equalToConstant: Self.rowHeight),
            row.widthAnchor.constraint(equalToConstant: layout.rowWidth),
            numberLabel.leadingAnchor.constraint(equalTo: row.leadingAnchor),
            numberLabel.centerYAnchor.constraint(equalTo: row.centerYAnchor),
            candidateLabel.leadingAnchor.constraint(
                equalTo: row.leadingAnchor, constant: layout.candidateX),
            candidateLabel.centerYAnchor.constraint(equalTo: row.centerYAnchor),
        ]

        if let descriptionX = layout.descriptionX, let description = candidate.description {
            let descColor: NSColor =
                selected ? NSColor.white.withAlphaComponent(0.8) : .secondaryLabelColor
            let descriptionLabel = makeLabel(
                description, size: Self.footerFontSize, color: descColor)
            row.addSubview(descriptionLabel)
            constraints.append(
                descriptionLabel.leadingAnchor.constraint(
                    equalTo: row.leadingAnchor, constant: descriptionX))
            constraints.append(
                descriptionLabel.centerYAnchor.constraint(equalTo: row.centerYAnchor))
        }

        NSLayoutConstraint.activate(constraints)
        stackView.addArrangedSubview(row)
        rowViews.append(row)
    }

    /// A single-line, non-wrapping label pinned to its intrinsic width.
    private func makeLabel(_ string: String, size: CGFloat, color: NSColor) -> NSTextField {
        let label = NSTextField(labelWithString: string)
        label.font = NSFont.systemFont(ofSize: size)
        label.textColor = color
        label.maximumNumberOfLines = 1
        label.lineBreakMode = .byClipping
        label.drawsBackground = false
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }

    private func addFooterLabel(_ text: String) {
        let label = NSTextField(labelWithString: text)
        label.font = NSFont.systemFont(ofSize: Self.footerFontSize)
        label.textColor = NSColor.secondaryLabelColor
        label.translatesAutoresizingMaskIntoConstraints = false
        stackView.addArrangedSubview(label)
        rowViews.append(label)
    }

    private var lastCursorRect: NSRect = .zero

    private func positionPanel(cursorRect: NSRect?) {
        if let rect = cursorRect {
            lastCursorRect = rect
        }
        let cursorRect = lastCursorRect

        stackView.layoutSubtreeIfNeeded()
        let contentSize = stackView.fittingSize
        let panelWidth = max(contentSize.width + 16, Self.minPanelWidth)
        let panelHeight = contentSize.height + 8

        guard cursorRect != .zero else {
            panel.setFrame(
                NSRect(x: 100, y: 100, width: panelWidth, height: panelHeight), display: true)
            panel.orderFront(nil)
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

        panel.setFrame(
            NSRect(x: cursorRect.origin.x, y: originY, width: panelWidth, height: panelHeight),
            display: true)
        panel.orderFront(nil)
    }
}
