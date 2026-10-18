import Foundation
import PaddockClient
import Testing

@testable import PaddockUI

@Suite("App-owned management lifetime", .timeLimit(.minutes(1))) @MainActor
struct WorkspaceTests {
  @Test func appearanceFollowsSystemUnlessExplicitlyChosen() {
    #expect(WorkspaceAppearance.system.colorScheme == nil)
    #expect(WorkspaceAppearance.light.colorScheme == .light)
    #expect(WorkspaceAppearance.dark.colorScheme == .dark)
  }

  @Test func automaticStartupIsIdempotentAndShutdownClosesCore() async throws {
    let loader = ControlledLoader()
    let model = WorkspaceModel(client: loader)
    let operation = Task { await model.start() }
    await loader.waitForCall(1)
    #expect(model.state == .loading)
    await model.start()
    #expect(await loader.callCount == 1)
    await loader.resolve(1, with: .success(try fixture()))
    await operation.value
    #expect(model.state == .ready)
    #expect(model.refreshedAt != nil)
    await model.shutdown()
    #expect(model.state == .stopped)
    #expect(model.snapshot == nil)
    #expect(model.refreshedAt == nil)
    #expect(await loader.closeCount == 1)
    await model.refresh()
    #expect(await loader.callCount == 1)
  }

  @Test func lateOldRefreshCannotOverwriteNewState() async throws {
    let loader = ControlledLoader()
    let model = WorkspaceModel(client: loader)
    let first = Task { await model.refresh() }
    await loader.waitForCall(1)
    let second = Task { await model.refresh() }
    await loader.waitForCall(2)
    await loader.resolve(2, with: .success(try fixture(version: "new")))
    await second.value
    await loader.resolve(1, with: .success(try fixture(version: "old")))
    await first.value
    #expect(model.snapshot?.identity.version == "new")
    #expect(model.state == .ready)
  }

  @Test func shutdownWinsOverUncooperativeCore() async throws {
    let loader = ControlledLoader()
    let model = WorkspaceModel(client: loader)
    let operation = Task { await model.start() }
    await loader.waitForCall(1)
    await model.shutdown()
    await loader.resolve(1, with: .success(try fixture()))
    await operation.value
    #expect(model.snapshot == nil)
    #expect(model.state == .stopped)
  }

  @Test func failedRefreshRetainsSnapshotButMarksItStale() async throws {
    let loader = ControlledLoader()
    let model = WorkspaceModel(client: loader)
    let initial = Task { await model.start() }
    await loader.waitForCall(1)
    await loader.resolve(1, with: .success(try fixture()))
    await initial.value
    let date = model.refreshedAt
    let refresh = Task { await model.refresh() }
    await loader.waitForCall(2)
    await loader.resolve(2, with: .failure(ManagerError.core("Test failure")))
    await refresh.value
    #expect(model.snapshot != nil)
    #expect(model.refreshedAt == date)
    #expect(model.state == .failed("Test failure"))
  }

  @Test func backgroundRefreshStaysSilentWhileSnapshotIsKept() async throws {
    let loader = ControlledLoader()
    let model = WorkspaceModel(client: loader)
    let initial = Task { await model.start() }
    await loader.waitForCall(1)
    await loader.resolve(1, with: .success(try fixture()))
    await initial.value
    #expect(model.state == .ready)
    #expect(model.canSubmit)
    let refresh = Task { await model.refresh() }
    await loader.waitForCall(2)
    // In-flight reconcile must not flip to .loading — that briefly disables
    // canSubmit and flashes FlatButtonStyle opacity on every poll.
    #expect(model.state == .ready)
    #expect(model.canSubmit)
    #expect(model.snapshot != nil)
    await loader.resolve(2, with: .success(try fixture(version: "next")))
    await refresh.value
    #expect(model.state == .ready)
    #expect(model.snapshot?.identity.version == "next")
  }

  @Test func cancellationDoesNotPublishLateResults() async throws {
    let loader = ControlledLoader()
    let model = WorkspaceModel(client: loader)
    let operation = Task { await model.start() }
    await loader.waitForCall(1)
    operation.cancel()
    await loader.resolve(1, with: .success(try fixture()))
    await operation.value
    #expect(model.state == .failed("Refresh cancelled."))
    #expect(model.snapshot == nil)
  }

  @Test func missingMeasurementsNeverDisplayZero() {
    #expect(DisplayFormat.bytes(nil) == "Not reported")
    #expect(DisplayFormat.bytes(UInt64.max) == "Not reported")
    #expect(DisplayFormat.bytes(1024).contains("KB"))
    #expect(DisplayFormat.defaultBundle(0) == "No default")
  }
}

// Deliberately ignores cancellation so the UI's generation guard is tested,
// not just the behaviour of a cooperative core implementation.
private actor ControlledLoader: ManagerLoading {
  private var continuations: [Int: CheckedContinuation<ManagerSnapshot, any Error>] = [:]
  private var waiters: [(Int, CheckedContinuation<Void, Never>)] = []
  private(set) var callCount = 0
  private(set) var closeCount = 0

  func close() async { closeCount += 1 }

  func snapshot() async throws
    -> ManagerSnapshot
  {
    callCount += 1
    let call = callCount
    return try await withCheckedThrowingContinuation { continuation in
      continuations[call] = continuation
      let ready = waiters.filter { $0.0 <= callCount }
      waiters.removeAll { $0.0 <= callCount }
      for (_, waiter) in ready { waiter.resume() }
    }
  }

  func waitForCall(_ call: Int) async {
    if callCount >= call { return }
    await withCheckedContinuation { waiters.append((call, $0)) }
  }

  func resolve(_ call: Int, with result: Result<ManagerSnapshot, any Error>) {
    continuations.removeValue(forKey: call)?.resume(with: result)
  }
}

func fixture(version: String = "0.1.5") throws -> ManagerSnapshot {
  ManagerSnapshot(
    identity: try ManagerWire.decode(
      ManagerIdentity.self, from: Data("{\"role\":\"manager\",\"version\":\"\(version)\"}".utf8)),
    readiness: try ManagerWire.decode(
      Readiness.self,
      from: Data(#"{"backend":"metal","state":"untested","os":"macos","card":"Apple M5 Max"}"#.utf8)
    ),
    catalog: try ManagerWire.decode(
      ModelCatalog.self, from: Data(#"{"schema":3,"models":[]}"#.utf8)),
    runners: []
  )
}
