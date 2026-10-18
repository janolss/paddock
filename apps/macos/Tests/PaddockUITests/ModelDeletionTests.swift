import Foundation
import PaddockClient
import Testing

@testable import PaddockUI

@Suite("Delete downloaded model weights", .serialized) @MainActor
struct ModelDeletionTests {
  @Test func onlyInstalledMetalWeightsProduceADeletionReview() throws {
    let snapshot = try endpointFixture()
    let model = try #require(snapshot.catalog.models.first)
    let mlx = try #require(model.artifacts.first { $0.id == "mlx-4bit" })
    #expect(
      ModelDeletionReview(
        model: model, artifact: mlx, backend: "metal", snapshot: snapshot, latestJob: nil) != nil)
    let cuda = try #require(model.artifacts.first { $0.id == "cuda" })
    #expect(
      ModelDeletionReview(
        model: model, artifact: cuda, backend: "metal", snapshot: snapshot, latestJob: nil) == nil)
    let missing = try ManagerWire.decode(
      CatalogArtifact.self,
      from: Data(
        #"{"id":"mlx-4bit","kind":"weights","format":"safetensors","label":"MLX","installed":false,"total_size":1,"backend_supported":true}"#
          .utf8))
    #expect(
      ModelDeletionReview(
        model: model, artifact: missing, backend: "metal", snapshot: snapshot, latestJob: nil)
        == nil)
  }

  @Test func keepConfigurationDeletesFilesAndLeavesStoppedEndpoint() async throws {
    let client = DeletionFixture()
    let workspace = WorkspaceModel(client: client)
    await workspace.start()
    let review = try #require(deletionReview(in: workspace))
    #expect(review.configurationPorts == [12346])
    #expect(workspace.canDeleteDownloaded(review))
    #expect(await workspace.deleteDownloaded(review, removeConfigurations: false))
    #expect(await client.deleted.map(\.model) == ["qwen3.8-27b"])
    #expect(await client.deleted.map(\.artifact) == ["mlx-4bit"])
    #expect(await client.removedPorts.isEmpty)
    #expect(workspace.snapshot?.servers?.map(\.port) == [12346])
    #expect(
      workspace.snapshot?.catalog.models.first?.artifacts.first { $0.id == "mlx-4bit" }?.installed
        == false)
    #expect(!workspace.canDeleteDownloaded(review))
    await workspace.shutdown()
  }

  @Test func removeConfigurationDeletesFilesAndEndpoint() async throws {
    let client = DeletionFixture()
    let workspace = WorkspaceModel(client: client)
    await workspace.start()
    let review = try #require(deletionReview(in: workspace))
    #expect(await workspace.deleteDownloaded(review, removeConfigurations: true))
    #expect(await client.deleted.map(\.model) == ["qwen3.8-27b"])
    #expect(await client.deleted.map(\.artifact) == ["mlx-4bit"])
    #expect(await client.removedPorts == [12346])
    #expect(workspace.snapshot?.servers?.isEmpty == true)
    await workspace.shutdown()
  }

  @Test func runningOrBusyEndpointsBlockDelete() async throws {
    let client = DeletionFixture(running: true)
    let workspace = WorkspaceModel(client: client)
    await workspace.start()
    let review = try #require(deletionReview(in: workspace))
    #expect(!workspace.canDeleteDownloaded(review))
    #expect(await workspace.deleteDownloaded(review, removeConfigurations: false) == false)
    #expect(await client.deleted.isEmpty)
    await workspace.shutdown()
  }

  @Test func dirtyMatchingEditorBlocksDeleteWithoutTouchingFiles() async throws {
    let client = DeletionFixture()
    let workspace = WorkspaceModel(client: client)
    await workspace.start()
    let row = try #require(
      workspace.snapshot.flatMap { EndpointRow.rows(snapshot: $0, latestJob: nil).first })
    workspace.openEndpoint(row)
    workspace.endpointEditor?.context = "8192"
    let review = try #require(deletionReview(in: workspace))
    #expect(!workspace.canDeleteDownloaded(review))
    #expect(await workspace.deleteDownloaded(review, removeConfigurations: false) == false)
    #expect(await client.deleted.isEmpty)
    await workspace.shutdown()
  }
}

@MainActor
private func deletionReview(in workspace: WorkspaceModel) -> ModelDeletionReview? {
  guard let snapshot = workspace.snapshot,
    let model = snapshot.catalog.models.first,
    let artifact = model.artifacts.first(where: { $0.id == "mlx-4bit" })
  else { return nil }
  return ModelDeletionReview(
    model: model, artifact: artifact, backend: snapshot.readiness.backend,
    snapshot: snapshot, latestJob: workspace.latestJob)
}

private struct DeletedSelection: Sendable {
  let model: String
  let artifact: String
}

private actor DeletionFixture: ManagerLoading {
  private(set) var deleted: [DeletedSelection] = []
  private(set) var removedPorts: [UInt16] = []
  private var installed = true
  private var revision = "reviewed"
  private let running: Bool
  private var jobs: [ManagementJob] = []
  init(running: Bool = false) { self.running = running }

  func snapshot() async throws -> ManagerSnapshot {
    let base = try endpointFixture()
    let catalog = try ManagerWire.decode(
      ModelCatalog.self,
      from: Data(
        """
        {"schema":3,"models":[{"id":"qwen3.8-27b","display":"Qwen 3.8 27B","vendor":"Alibaba",
        "capability":["chat"],"installed":\(installed),"total_size":15000000000,
        "artifacts":[{"id":"mlx-4bit","kind":"weights","format":"safetensors","label":"MLX 4-bit",
        "installed":\(installed),"total_size":15000000000,"backend_supported":true,
        "runtime":{"capability":["chat"]}}]}]}
        """.utf8))
    return ManagerSnapshot(
      identity: base.identity, readiness: base.readiness, catalog: catalog, runners: [],
      servers: removedPorts.contains(12346)
        ? []
        : [
          try ManagerWire.decode(
            ConfiguredEndpoint.self,
            from: JSONSerialization.data(withJSONObject: [
              "port": 12346, "display": "Qwen 3.8 27B", "model": "qwen3.8-27b",
              "artifact": "mlx-4bit", "revision": revision, "running": running, "local_only": true,
              "settings": [
                "host": "127.0.0.1", "max_ctx": 4096, "max_batch": 1, "device": "metal",
                "has_api_key": true, "vision": false, "forensics": false,
              ],
            ]))
        ], jobs: jobs)
  }

  func downloads(_ command: DownloadCommand) async throws -> DownloadReply {
    guard case .delete(let model, let artifact) = command, installed, !running else {
      throw ManagerError.core("Delete refused")
    }
    deleted.append(DeletedSelection(model: model, artifact: artifact))
    installed = false
    return try ManagerWire.decode(
      DownloadReply.self,
      from: Data(
        #"{"jobs":[],"deleted":{"model":"qwen3.8-27b","artifact":"mlx-4bit","removed_files":2,"freed_bytes":15000000000,"kept_shared_files":0}}"#
          .utf8))
  }

  func submit(_ command: ModelCommand) async throws -> ManagementJob {
    guard case .remove(let port, let revision) = command, revision == self.revision, !running else {
      throw ManagerError.core("Configuration changed")
    }
    let receipt = try ManagerWire.decode(
      ManagementJob.self,
      from: JSONSerialization.data(withJSONObject: [
        "id": 1, "port": Int(port), "action": "remove", "state": "succeeded",
        "message": "Configuration removed",
      ]))
    removedPorts.append(port)
    jobs = [receipt]
    return receipt
  }
}
