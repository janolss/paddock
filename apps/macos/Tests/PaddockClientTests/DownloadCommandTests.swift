import Foundation
import Testing

@testable import PaddockClient

@Suite("Download wire contract")
struct DownloadCommandTests {
  @Test func commandsCarryOnlyReviewedCatalogIDs() throws {
    let plan = try ManagerWire.decode(
      DownloadPlan.self,
      from: Data(
        """
        {"model":"qwen","artifact":"mlx","display":"Qwen","selection":["mlx","vision"],
         "total":1024,"remaining":512,"free":2147483648,"file_count":4,"pieces":[]}
        """.utf8))
    let encoded = try #require(
      JSONSerialization.jsonObject(with: JSONEncoder().encode(DownloadCommand.pull(plan)))
        as? [String: Any])
    #expect(Set(encoded.keys) == ["kind", "model", "artifact", "selection"])
    #expect(encoded["selection"] as? [String] == ["mlx", "vision"])
    #expect(plan.fits)
    let delete = try #require(
      JSONSerialization.jsonObject(
        with: JSONEncoder().encode(DownloadCommand.delete(model: "qwen", artifact: "mlx")))
        as? [String: Any])
    #expect(Set(delete.keys) == ["kind", "model", "artifact"])
    #expect(delete["kind"] as? String == "delete")
    let noSpace = try ManagerWire.decode(
      DownloadPlan.self,
      from: Data(
        """
        {"model":"qwen","artifact":"mlx","display":"Qwen","selection":["mlx"],
         "total":1024,"remaining":512,"free":511,"file_count":1,"pieces":[]}
        """.utf8))
    #expect(!noSpace.fits)
  }
}
