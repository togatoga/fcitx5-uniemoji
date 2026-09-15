import Foundation

// Swift-side mirror of karukan-im/core/src/server/protocol.rs.
// All positions (caret, attribute start/end) are in Unicode scalar values;
// convert to UTF-16 offsets before passing to IMK APIs.

struct KeyResult: Decodable {
    let consumed: Bool
    let actions: [EngineAction]
    let conversionMs: UInt64?
    let processKeyMs: UInt64?
}

struct PreeditAttr: Decodable {
    let start: Int
    let end: Int
    let style: String
}

struct CandidateItem: Decodable, Equatable {
    let text: String
    let description: String?
}

struct InitResult: Decodable {
    let protocolVersion: Int
    let modelName: String
}

enum EngineAction: Decodable {
    case updatePreedit(text: String, caret: Int, attributes: [PreeditAttr])
    /// `gridColumns` is present when the engine's candidate layout is the
    /// grid (`[display] candidate_layout`): render the page row-major at
    /// that many columns. Absent = classic vertical list.
    case showCandidates(
        candidates: [CandidateItem], cursor: Int, page: Int, totalPages: Int, gridColumns: Int?)
    case hideCandidates
    case commit(text: String)
    case updateAux(text: String)
    case hideAux

    private enum CodingKeys: String, CodingKey {
        case type, text, caret, attributes, candidates, cursor, page, totalPages, gridColumns
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "update_preedit":
            self = .updatePreedit(
                text: try container.decode(String.self, forKey: .text),
                caret: try container.decode(Int.self, forKey: .caret),
                attributes: try container.decodeIfPresent([PreeditAttr].self, forKey: .attributes)
                    ?? []
            )
        case "show_candidates":
            self = .showCandidates(
                candidates: try container.decode([CandidateItem].self, forKey: .candidates),
                cursor: try container.decode(Int.self, forKey: .cursor),
                page: try container.decode(Int.self, forKey: .page),
                totalPages: try container.decode(Int.self, forKey: .totalPages),
                gridColumns: try container.decodeIfPresent(Int.self, forKey: .gridColumns)
            )
        case "hide_candidates":
            self = .hideCandidates
        case "commit":
            self = .commit(text: try container.decode(String.self, forKey: .text))
        case "update_aux":
            self = .updateAux(text: try container.decode(String.self, forKey: .text))
        case "hide_aux":
            self = .hideAux
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type, in: container, debugDescription: "unknown action type: \(type)")
        }
    }
}

/// JSONDecoder configured for the engine protocol (snake_case keys).
func makeProtocolDecoder() -> JSONDecoder {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return decoder
}
