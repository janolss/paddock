import Foundation

public struct ModelDownload: Decodable, Identifiable, Sendable {
  public let id: String
  public let model: String
  public let display: String
  public let artifacts: [String]?
  public let weights: String?
  public let downloaded: UInt64
  public let total: UInt64
  public let createdMs: UInt64
  public let status: Status
  public let phase: String?
  public let stage: String?
  public let cancelling: Bool?
  public struct Status: Decodable, Sendable {
    public let state: String
    public let message: String?
  }
  public var active: Bool { status.state == "running" }
  public var complete: Bool { status.state == "done" }
  public var resumable: Bool { ["cancelled", "error"].contains(status.state) }
  public var progress: Double { total == 0 ? 0 : min(1, Double(downloaded) / Double(total)) }
  public var title: String {
    if active && cancelling == true { return "Pausing…" }
    switch status.state {
    case "running":
      return stage == "verifying" || downloaded >= total ? "Verifying…" : "Downloading"
    case "done": return "Ready to start"
    case "cancelled": return "Paused"
    case "error": return "Needs attention"
    default: return "Status unavailable"
    }
  }
}

public struct DownloadPlan: Decodable, Sendable, Identifiable {
  public let model: String
  public let artifact: String
  public let display: String
  public let selection: [String]
  public let total: UInt64
  public let remaining: UInt64
  public let diskNeed: UInt64?
  public let free: UInt64?
  public let fileCount: Int
  public let pieces: [Piece]
  public var id: String { model + ":" + artifact }
  public var fits: Bool {
    free.map { (diskNeed ?? remaining) <= $0.saturatingSubtract(1 << 30) } ?? false
  }
  public struct Piece: Decodable, Sendable, Identifiable {
    public let id: String
    public let label: String
    public let size: UInt64
    public var displayLabel: String {
      label.replacingOccurrences(of: " (Metal preview)", with: "")
        .replacingOccurrences(of: " (macOS preview)", with: "")
    }
  }
}

extension UInt64 {
  fileprivate func saturatingSubtract(_ other: UInt64) -> UInt64 {
    self >= other ? self - other : 0
  }
}

public struct DownloadReply: Decodable, Sendable {
  public let jobs: [ModelDownload]
  public let plan: DownloadPlan?
  public let job: String?
  public let deleted: DeletedDownload?
  public struct DeletedDownload: Decodable, Sendable {
    public let model: String
    public let artifact: String
    public let removedFiles: Int
    public let freedBytes: UInt64
    public let keptSharedFiles: Int
  }
}

public enum DownloadCommand: Sendable, Encodable {
  case list
  case plan(model: String, artifact: String)
  case pull(DownloadPlan)
  case pause(id: String)
  case resume(id: String)
  case delete(model: String, artifact: String)
  private enum CodingKeys: String, CodingKey { case kind, model, artifact, selection, id }
  public func encode(to encoder: any Encoder) throws {
    var values = encoder.container(keyedBy: CodingKeys.self)
    switch self {
    case .list: try values.encode("list", forKey: .kind)
    case .plan(let model, let artifact):
      try values.encode("plan", forKey: .kind)
      try values.encode(model, forKey: .model)
      try values.encode(artifact, forKey: .artifact)
    case .pull(let plan):
      try values.encode("pull", forKey: .kind)
      try values.encode(plan.model, forKey: .model)
      try values.encode(plan.artifact, forKey: .artifact)
      try values.encode(plan.selection, forKey: .selection)
    case .pause(let id):
      try values.encode("pause", forKey: .kind)
      try values.encode(id, forKey: .id)
    case .resume(let id):
      try values.encode("resume", forKey: .kind)
      try values.encode(id, forKey: .id)
    case .delete(let model, let artifact):
      try values.encode("delete", forKey: .kind)
      try values.encode(model, forKey: .model)
      try values.encode(artifact, forKey: .artifact)
    }
  }
}
