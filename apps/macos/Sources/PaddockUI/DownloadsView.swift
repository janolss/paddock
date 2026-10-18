import PaddockClient
import SwiftUI

struct DownloadsView: View {
  @Bindable var workspace: WorkspaceModel
  let snapshot: ManagerSnapshot
  let onStart: (String, String) -> Void
  @Binding var deleting: ModelDeletionReview?

  private var downloads: DownloadsModel { workspace.downloads }
  private var transfers: [ModelDownload] {
    downloads.visible.filter { !$0.complete }
  }
  private var installed: [InstalledDownload] {
    let backend = snapshot.readiness.backend
    let transferring = Set(
      transfers.compactMap { job -> String? in
        guard let weights = job.weights else { return nil }
        return job.model + ":" + weights
      })
    return snapshot.catalog.models.flatMap { model in
      model.artifacts.compactMap { artifact -> InstalledDownload? in
        guard artifact.kind == "weights", artifact.installed, artifact.supports(backend: backend)
        else { return nil }
        guard !transferring.contains(model.id + ":" + artifact.id) else { return nil }
        return InstalledDownload(model: model, artifact: artifact)
      }
    }.sorted {
      let names = $0.model.display.localizedStandardCompare($1.model.display)
      if names != .orderedSame { return names == .orderedAscending }
      return $0.artifact.id < $1.artifact.id
    }
  }
  private var isEmpty: Bool { transfers.isEmpty && installed.isEmpty }

  var body: some View {
    VStack(spacing: 0) {
      VStack(alignment: .leading, spacing: 20) {
        PageHeading(title: "Downloads") {}
        if let error = downloads.error {
          HStack(alignment: .top) {
            Label(error, systemImage: "exclamationmark.circle").textSelection(.enabled)
            Spacer()
            Button("Try again") { Task { await downloads.refresh() } }
              .buttonStyle(FlatButtonStyle()).disabled(downloads.busy)
          }.font(.system(size: 12)).foregroundStyle(PaddockStyle.caution)
        }
      }.padding(32)
      if isEmpty {
        Spacer(minLength: 0)
      } else {
        PaddockScrollView {
          LazyVStack(spacing: 0) {
            ForEach(transfers) { job in
              transferRow(job)
              WorkspaceRule()
            }
            ForEach(installed) { row in
              installedRow(row)
              WorkspaceRule()
            }
          }.padding(.horizontal, 32).padding(.bottom, 32)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
      }
    }.frame(maxWidth: .infinity, maxHeight: .infinity)
      .overlay {
        if isEmpty {
          if !downloads.loaded && downloads.error == nil {
            ProgressView("Loading downloads…")
              .frame(maxWidth: .infinity, maxHeight: .infinity).allowsHitTesting(false)
          } else {
            DownloadsEmptyState(unavailable: !downloads.loaded && downloads.error != nil)
              .allowsHitTesting(false)
          }
        }
      }
      .background(PaddockStyle.canvas).task { await downloads.refresh() }
      .accessibilityIdentifier("downloads-list")
  }

  private func transferRow(_ job: ModelDownload) -> some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack(alignment: .top, spacing: 16) {
        VStack(alignment: .leading, spacing: 6) {
          Text(verbatim: job.display).font(.system(size: 14, weight: .medium))
          Text(job.title).font(.system(size: 12)).foregroundStyle(.secondary)
        }
        Spacer()
        if job.active {
          Button("Pause", systemImage: "pause") {
            Task { await downloads.perform(.pause(id: job.id)) }
          }
          .buttonStyle(FlatButtonStyle()).disabled(downloads.busy || job.cancelling == true)
          .accessibilityIdentifier("download-pause-\(job.id)")
        } else if job.resumable {
          Button(job.status.state == "error" ? "Retry" : "Resume", systemImage: "arrow.clockwise") {
            Task { await downloads.perform(.resume(id: job.id)) }
          }.buttonStyle(FlatButtonStyle()).disabled(downloads.busy)
            .accessibilityIdentifier("download-resume-\(job.id)")
        }
      }
      ProgressView(value: job.progress).tint(PaddockStyle.accent)
        .accessibilityLabel("\(job.display) download progress")
      HStack {
        Text("\(DisplayFormat.bytes(job.downloaded)) of \(DisplayFormat.bytes(job.total))")
        Spacer()
        Text(job.progress, format: .percent.precision(.fractionLength(0)))
      }.font(.system(size: 11)).monospacedDigit().foregroundStyle(.secondary)
      if let message = job.status.message {
        Text(verbatim: message).font(.system(size: 12)).foregroundStyle(PaddockStyle.caution)
          .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
      } else if job.active, let phase = job.phase {
        if let transfer = downloads.transferStatus(for: job) {
          Text(transfer).font(.system(size: 11)).monospacedDigit().foregroundStyle(.secondary)
            .accessibilityIdentifier("download-speed-\(job.id)")
        }
        Text(verbatim: phase).font(.system(size: 11)).foregroundStyle(.secondary)
          .lineLimit(2).truncationMode(.middle).help(phase)
      }
    }.padding(.vertical, 20).accessibilityElement(children: .contain)
  }

  private func installedRow(_ row: InstalledDownload) -> some View {
    let review = ModelDeletionReview(
      model: row.model, artifact: row.artifact, backend: snapshot.readiness.backend,
      snapshot: snapshot, latestJob: workspace.latestJob)
    return HStack(alignment: .top, spacing: 16) {
      VStack(alignment: .leading, spacing: 6) {
        Text(verbatim: row.model.display).font(.system(size: 14, weight: .medium))
        Text(
          "\(row.artifact.displayLabel) · \(DisplayFormat.bytes(row.artifact.totalSize)) · Ready to start"
        )
        .font(.system(size: 12)).foregroundStyle(.secondary)
      }
      Spacer(minLength: 8)
      if let review {
        Button("Delete downloads…", systemImage: "trash", role: .destructive) {
          deleting = review
        }
        .labelStyle(.iconOnly).buttonStyle(QuietButtonStyle())
        .frame(width: 32, height: 32)
        .disabled(!workspace.canDeleteDownloaded(review))
        .accessibilityLabel("Delete downloaded weights for \(row.model.display)")
        .accessibilityIdentifier("download-delete-\(row.model.id)-\(row.artifact.id)")
        .help("Delete downloaded model files")
      }
      Button("Configure instance", systemImage: "plus") {
        onStart(row.model.id, row.artifact.id)
      }
      .buttonStyle(FlatButtonStyle(primary: true)).disabled(!workspace.canSubmit)
      .accessibilityIdentifier("download-start-\(row.model.id)-\(row.artifact.id)")
    }.padding(.vertical, 20).accessibilityElement(children: .contain)
  }
}

private struct InstalledDownload: Identifiable {
  let model: CatalogModel
  let artifact: CatalogArtifact
  var id: String { model.id + ":" + artifact.id }
}

/// Center the visible icon/text group as a single unit. ContentUnavailableView
/// owns additional internal spacing; an explicit stack makes our geometry exact.
struct DownloadsEmptyState: View {
  var unavailable = false
  var body: some View {
    VStack(spacing: 14) {
      Image(systemName: unavailable ? "exclamationmark.circle" : "arrow.down.circle")
        .font(.system(size: 30, weight: .light)).foregroundStyle(.secondary)
      Text(unavailable ? "Downloads unavailable" : "No downloads yet")
        .font(.system(size: 18, weight: .medium)).foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
    }.padding(.horizontal, 32)
      .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
      .accessibilityIdentifier("downloads-empty")
  }
}
