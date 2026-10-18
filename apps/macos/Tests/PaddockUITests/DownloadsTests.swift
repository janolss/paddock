import AppKit
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native model downloads") @MainActor
struct DownloadsTests {
  @Test func speedExcludesResumeBytesAndResetsAfterStallsAndRetries() {
    var sample = DownloadActivity()
    sample.record(bytes: 656 * 1024 * 1024, at: 10)
    #expect(sample.bytesPerSecond == 0 && !sample.waiting)
    sample.record(bytes: 657 * 1024 * 1024, at: 11)
    #expect(sample.bytesPerSecond == 1024 * 1024)
    sample.record(bytes: 657 * 1024 * 1024, at: 16)
    #expect(sample.waiting && sample.bytesPerSecond == 0)
    sample.record(bytes: 658 * 1024 * 1024, at: 17)
    #expect(!sample.waiting && sample.bytesPerSecond > 0)
    sample.record(bytes: 656 * 1024 * 1024, at: 18)
    #expect(!sample.waiting && sample.bytesPerSecond == 0)
  }

  static func reply(
    state: String = "running", bytes: UInt64 = 10, id: String = "a", created: UInt64 = 1
  ) throws -> DownloadReply {
    try ManagerWire.decode(
      DownloadReply.self,
      from: JSONSerialization.data(withJSONObject: [
        "jobs": [
          [
            "id": id, "model": "fixture", "display": "Synthetic MLX model",
            "artifacts": ["mlx", "vision"],
            "downloaded": bytes, "total": 100, "created_ms": created, "status": ["state": state],
            "phase": "model-00001-of-00004.safetensors",
          ]
        ]
      ]))
  }
  @Test func completedBytesAreNotCompletedVerification() throws {
    let verifying = try #require(Self.reply(bytes: 100).jobs.first)
    #expect(verifying.title == "Verifying…")
    #expect(verifying.active && !verifying.complete && !verifying.resumable)
    let failed = try #require(Self.reply(state: "error", bytes: 100).jobs.first)
    #expect(failed.resumable && !failed.complete)
    #expect(try Self.reply(bytes: 1000).jobs.first?.progress == 1)
  }

  @Test func staleDownloadCannotStartADifferentModel() async throws {
    let snapshot = try endpointFixture()
    let view = StartModelView(
      snapshot: snapshot, model: "missing-downloaded-model", artifact: "mlx-4bit"
    ) { _ in false }
    #expect(view.validation == "Select a downloaded Metal weights artifact.")
  }

  @Test func metadataRefreshOccursOnceAfterVerificationAndFailuresRetainRows() async throws {
    let client = DownloadFixture(reply: try Self.reply())
    let model = DownloadsModel(client: client)
    var completions = 0
    model.onCompletion = { completions += 1 }
    await model.refresh()
    #expect(model.active && completions == 0)
    await client.set(try Self.reply(state: "done", bytes: 100))
    await model.refresh()
    await model.refresh()
    #expect(completions == 1 && model.jobs.first?.complete == true)
    await client.fail()
    await model.refresh()
    #expect(model.jobs.first?.complete == true && model.error != nil)
    model.stop()
  }

  @Test func resumeShowsOnlyTheLatestAttemptAndUnknownStatesRemainHonest() async throws {
    let first = try Self.reply(state: "cancelled").jobs
    let next = try Self.reply(id: "b", created: 2).jobs
    let data = try JSONSerialization.data(withJSONObject: ["jobs": []])
    let fixture = DownloadFixture(reply: try ManagerWire.decode(DownloadReply.self, from: data))
    // Encode fixture rows using the same wire shape; no public mutable model state.
    let combined = try JSONSerialization.data(withJSONObject: [
      "jobs": (first + next).map {
        [
          "id": $0.id, "model": $0.model, "display": $0.display, "artifacts": $0.artifacts!,
          "downloaded": $0.downloaded, "total": $0.total, "created_ms": $0.createdMs,
          "status": ["state": $0.status.state],
        ] as [String: Any]
      }
    ])
    await fixture.set(try ManagerWire.decode(DownloadReply.self, from: combined))
    let model = DownloadsModel(client: fixture)
    await model.refresh()
    #expect(model.visible.map(\.id) == ["b"])
    #expect(try Self.reply(state: "future").jobs.first?.title == "Status unavailable")
    model.stop()
  }

  @Test func listFitsLightAndDarkAtNarrowWindowWidth() async throws {
    _ = NSApplication.shared
    let fixture = DownloadFixture(reply: try Self.reply(state: "cancelled"))
    let workspace = WorkspaceModel(client: fixture)
    await workspace.start()
    defer { Task { await workspace.shutdown() } }
    await workspace.downloads.refresh()
    let snapshot = try #require(workspace.snapshot)
    for dark in [false, true] {
      let host = NSHostingController(
        rootView: DownloadsView(
          workspace: workspace, snapshot: snapshot, onStart: { _, _ in },
          deleting: .constant(nil)
        ).environment(\.colorScheme, dark ? .dark : .light))
      let size = host.sizeThatFits(in: CGSize(width: 640, height: 650))
      #expect(size.width <= 640)
      host.view.frame = NSRect(x: 0, y: 0, width: 640, height: 650)
      host.view.layoutSubtreeIfNeeded()
      // Intrinsic fittingSize asks how wide unwrapped text wants to be, not
      // whether this flexible page accepts the proposed narrow viewport.
      #expect(host.sizeThatFits(in: CGSize(width: 640, height: 650)).width <= 640)
    }
  }

  @Test func emptyAndUnavailableStatesFillTheRemainingViewport() {
    _ = NSApplication.shared
    for dark in [false, true] {
      for unavailable in [false, true] {
        let host = NSHostingController(
          rootView: DownloadsEmptyState(unavailable: unavailable)
            .environment(\.colorScheme, dark ? .dark : .light))
        for size in [CGSize(width: 640, height: 520), CGSize(width: 1100, height: 740)] {
          host.view.frame = NSRect(origin: .zero, size: size)
          host.view.layoutSubtreeIfNeeded()
          let fit = host.sizeThatFits(in: size)
          #expect(
            abs(fit.width - size.width) < 1 && abs(fit.height - size.height) < 1,
            "The placeholder must fill, and center within, its proposed viewport: \(fit)")
        }
      }
    }
  }

  @Test func visibleEmptyGroupIsCenteredOnTheWholePageNotBelowItsHeader() async throws {
    let reply = try ManagerWire.decode(DownloadReply.self, from: Data(#"{"jobs":[]}"#.utf8))
    let fixture = DownloadFixture(reply: reply, emptyCatalog: true)
    let workspace = WorkspaceModel(client: fixture)
    await workspace.start()
    defer { Task { await workspace.shutdown() } }
    await workspace.downloads.refresh()
    let snapshot = try #require(workspace.snapshot)
    for size in [CGSize(width: 640, height: 650), CGSize(width: 1100, height: 800)] {
      let renderer = ImageRenderer(
        content: DownloadsView(
          workspace: workspace, snapshot: snapshot, onStart: { _, _ in },
          deleting: .constant(nil)
        ).frame(width: size.width, height: size.height)
          .environment(\.colorScheme, .dark))
      renderer.scale = 1
      let bitmap = NSBitmapImageRep(cgImage: try #require(renderer.cgImage))
      var minX = bitmap.pixelsWide
      var minY = bitmap.pixelsHigh
      var maxX = 0
      var maxY = 0
      // Exclude the fixed heading. Measure the actual visible ink, not the
      // expanding layout frame (which also passed with the previous offset).
      for y in 160..<bitmap.pixelsHigh {
        for x in 24..<(bitmap.pixelsWide - 24) {
          guard let color = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB),
            color.alphaComponent > 0.5,
            max(color.redComponent, color.greenComponent, color.blueComponent) > 0.35
          else { continue }
          minX = min(minX, x)
          maxX = max(maxX, x)
          minY = min(minY, y)
          maxY = max(maxY, y)
        }
      }
      #expect(maxX > minX && maxY > minY, "The placeholder must actually render")
      #expect(abs(Double(minX + maxX) / 2 - size.width / 2) < 6)
      #expect(
        abs(Double(minY + maxY) / 2 - size.height / 2) < 6,
        "The icon/text group is offset: y=\(minY)...\(maxY), pane height=\(size.height)")
    }
  }
}

private actor DownloadFixture: ManagerLoading {
  var reply: DownloadReply
  var fails = false
  let emptyCatalog: Bool
  init(reply: DownloadReply, emptyCatalog: Bool = false) {
    self.reply = reply
    self.emptyCatalog = emptyCatalog
  }
  func set(_ reply: DownloadReply) {
    self.reply = reply
    fails = false
  }
  func fail() { fails = true }
  func snapshot() async throws -> ManagerSnapshot {
    let base = try endpointFixture()
    if !emptyCatalog { return base }
    return try ManagerSnapshot(
      identity: base.identity, readiness: base.readiness,
      catalog: ManagerWire.decode(ModelCatalog.self, from: Data(#"{"schema":3,"models":[]}"#.utf8)),
      runners: [], servers: [])
  }
  func downloads(_ command: DownloadCommand) async throws -> DownloadReply {
    if fails { throw ManagerError.core("Synthetic connection failure") }
    return reply
  }
}
