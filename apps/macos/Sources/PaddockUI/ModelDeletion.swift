import PaddockClient
import SwiftUI

/// Catalog selection reviewed before deleting downloaded weights. Configurations
/// are listed separately so the confirmation can keep or remove them.
struct ModelDeletionReview: Equatable {
  let model: String
  let artifact: String
  let title: String
  let bytes: UInt64
  let configurationPorts: [UInt16]

  init?(
    model: CatalogModel, artifact: CatalogArtifact, backend: String?,
    snapshot: ManagerSnapshot?, latestJob: ManagementJob?
  ) {
    guard artifact.installed, artifact.supports(backend: backend),
      artifact.kind == "weights"
    else { return nil }
    self.model = model.id
    self.artifact = artifact.id
    title = model.display
    bytes = artifact.totalSize
    configurationPorts =
      snapshot.map {
        EndpointRow.rows(snapshot: $0, latestJob: latestJob).compactMap { row in
          guard row.configured?.model == model.id, row.configured?.artifact == artifact.id
          else { return nil }
          return row.port
        }.sorted()
      } ?? []
  }
}

extension WorkspaceModel {
  func canDeleteDownloaded(_ review: ModelDeletionReview) -> Bool {
    guard canSubmit, !downloads.busy, let snapshot else { return false }
    guard let entry = snapshot.catalog.models.first(where: { $0.id == review.model }),
      let artifact = entry.artifacts.first(where: { $0.id == review.artifact }),
      artifact.installed
    else { return false }
    let rows = EndpointRow.rows(snapshot: snapshot, latestJob: latestJob)
    if rows.contains(where: {
      $0.configured?.model == review.model && $0.configured?.artifact == review.artifact
        && ($0.runner != nil || $0.configured?.running == true || $0.job?.isActive == true)
    }) {
      return false
    }
    if downloads.jobs.contains(where: {
      $0.active && $0.model == review.model
        && ($0.artifacts?.contains(review.artifact) == true || $0.weights == review.artifact)
    }) {
      return false
    }
    if let editor = endpointEditor, editor.dirty,
      editor.modelID == review.model, editor.artifactID == review.artifact
    {
      return false
    }
    return ModelDeletionReview(
      model: entry, artifact: artifact, backend: snapshot.readiness.backend,
      snapshot: snapshot, latestJob: latestJob) == review
  }

  @discardableResult
  func deleteDownloaded(_ review: ModelDeletionReview, removeConfigurations: Bool) async -> Bool {
    guard canDeleteDownloaded(review) else {
      desktopError =
        "The download changed, is in use, or has unsaved endpoint edits. Review it before deleting."
      return false
    }
    let removals = removableConfigurations(for: review)
    guard await downloads.perform(.delete(model: review.model, artifact: review.artifact)) else {
      desktopError = downloads.error ?? "The downloaded weights could not be deleted."
      return false
    }
    await refresh()
    if removeConfigurations {
      for target in removals {
        guard canRemoveEndpoint(target) else {
          desktopError =
            "Downloads were deleted, but a configuration changed before it could be removed. Review Instances."
          return false
        }
        guard await removeEndpoint(target) else { return false }
      }
    }
    return true
  }

  private func removableConfigurations(for review: ModelDeletionReview) -> [EndpointRemovalReview] {
    guard let snapshot else { return [] }
    return EndpointRow.rows(snapshot: snapshot, latestJob: latestJob).compactMap { row in
      guard row.configured?.model == review.model, row.configured?.artifact == review.artifact
      else { return nil }
      return EndpointRemovalReview(row)
    }
  }
}

private struct ModelDeletionConfirmation: ViewModifier {
  @Bindable var workspace: WorkspaceModel
  @Binding var review: ModelDeletionReview?

  func body(content: Content) -> some View {
    content.confirmationDialog(
      "Delete downloaded weights for \(review?.title ?? "model")?",
      isPresented: Binding(get: { review != nil }, set: { if !$0 { review = nil } }),
      titleVisibility: .visible, presenting: review
    ) { target in
      if target.configurationPorts.isEmpty {
        Button("Delete downloads", role: .destructive) {
          review = nil
          Task { await workspace.deleteDownloaded(target, removeConfigurations: false) }
        }.disabled(!workspace.canDeleteDownloaded(target))
          .accessibilityIdentifier("confirm-delete-downloads-\(target.model)-\(target.artifact)")
      } else {
        Button("Delete downloads, keep configuration", role: .destructive) {
          review = nil
          Task { await workspace.deleteDownloaded(target, removeConfigurations: false) }
        }.disabled(!workspace.canDeleteDownloaded(target))
          .accessibilityIdentifier("confirm-delete-keep-config-\(target.model)-\(target.artifact)")
        Button("Delete downloads and remove configuration", role: .destructive) {
          review = nil
          Task { await workspace.deleteDownloaded(target, removeConfigurations: true) }
        }.disabled(!workspace.canDeleteDownloaded(target))
          .accessibilityIdentifier("confirm-delete-remove-config-\(target.model)-\(target.artifact)")
      }
      Button("Cancel", role: .cancel) { review = nil }
    } message: { target in
      if target.configurationPorts.isEmpty {
        Text(
          "\(DisplayFormat.bytes(target.bytes)). Deletes model files for this weights option. Conversations are kept. This cannot be undone."
        )
      } else {
        Text(
          "\(DisplayFormat.bytes(target.bytes)). Configured on port \(target.configurationPorts.map(String.init).joined(separator: ", ")). Conversations are kept. Keeping a configuration leaves a stopped instance that needs these weights downloaded again before it can start."
        )
      }
    }
  }
}

extension View {
  func modelDeletionConfirmation(
    workspace: WorkspaceModel, review: Binding<ModelDeletionReview?>
  ) -> some View {
    modifier(ModelDeletionConfirmation(workspace: workspace, review: review))
  }
}
