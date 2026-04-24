import XCTest
import Foundation
@testable import Transcribe

final class ShardedSafetensorsTests: XCTestCase {

    private func tmpDir(_ name: String) -> URL {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("ml_transcribe_shard_\(name)_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            atPath: path, withIntermediateDirectories: true
        )
        return URL(fileURLWithPath: path)
    }

    /// Write two shards + an HF-style index.json describing them.
    /// Returns the directory URL; caller is responsible for cleanup.
    private func makeFixture() throws -> URL {
        let dir = tmpDir("fixture")

        // Shard 1: encoder weights.
        let shard1Name = "model-00001-of-00002.safetensors"
        let shard1Path = dir.appendingPathComponent(shard1Name).path
        try SafetensorsWeights.save([
            ("encoder.layer0.weight", try Tensor.from(data: [1, 2, 3, 4], shape: [2, 2])),
            ("encoder.layer0.bias",   try Tensor.from(data: [10, 20],    shape: [2])),
        ], to: shard1Path)

        // Shard 2: decoder weights (non-overlapping names).
        let shard2Name = "model-00002-of-00002.safetensors"
        let shard2Path = dir.appendingPathComponent(shard2Name).path
        try SafetensorsWeights.save([
            ("decoder.layer0.weight", try Tensor.from(data: [5, 6, 7, 8, 9, 10], shape: [3, 2])),
            ("decoder.layer0.bias",   try Tensor.from(data: [100, 200, 300],     shape: [3])),
        ], to: shard2Path)

        // Write the HF-style index.
        let indexJSON: [String: Any] = [
            "metadata": ["total_size": 1_000_000],   // ignored by our parser
            "weight_map": [
                "encoder.layer0.weight": shard1Name,
                "encoder.layer0.bias":   shard1Name,
                "decoder.layer0.weight": shard2Name,
                "decoder.layer0.bias":   shard2Name,
            ],
        ]
        let data = try JSONSerialization.data(
            withJSONObject: indexJSON,
            options: .prettyPrinted
        )
        let indexURL = dir.appendingPathComponent("model.safetensors.index.json")
        try data.write(to: indexURL)

        return dir
    }

    // MARK: - Tests

    func testShardedLoadSeesAllTensors() throws {
        let dir = try makeFixture()
        defer { try? FileManager.default.removeItem(at: dir) }

        let weights = try SafetensorsWeights(shardedDirectory: dir)
        XCTAssertEqual(weights.count, 4)
        XCTAssertEqual(weights.keys, [
            "decoder.layer0.bias",
            "decoder.layer0.weight",
            "encoder.layer0.bias",
            "encoder.layer0.weight",
        ])
    }

    func testShardedLookupReturnsCorrectValues() throws {
        let dir = try makeFixture()
        defer { try? FileManager.default.removeItem(at: dir) }

        let weights = try SafetensorsWeights(shardedDirectory: dir)

        // From shard 1:
        XCTAssertEqual(weights["encoder.layer0.weight"]!.shape, [2, 2])
        XCTAssertEqual(weights["encoder.layer0.weight"]!.toArray(), [1, 2, 3, 4])
        XCTAssertEqual(weights["encoder.layer0.bias"]!.shape, [2])
        XCTAssertEqual(weights["encoder.layer0.bias"]!.toArray(), [10, 20])

        // From shard 2:
        XCTAssertEqual(weights["decoder.layer0.weight"]!.shape, [3, 2])
        XCTAssertEqual(weights["decoder.layer0.weight"]!.toArray(), [5, 6, 7, 8, 9, 10])
        XCTAssertEqual(weights["decoder.layer0.bias"]!.shape, [3])
        XCTAssertEqual(weights["decoder.layer0.bias"]!.toArray(), [100, 200, 300])

        // Not present:
        XCTAssertNil(weights["does.not.exist"])
    }

    func testShardedTensorsOutliveHandleViaARC() throws {
        // Same ownership contract as the single-file path: a borrowed
        // Tensor view must keep all shard handles alive via its parent
        // reference. Dropping the SafetensorsWeights explicitly while
        // holding a borrowed tensor must not free the underlying ids.
        let dir = try makeFixture()
        defer { try? FileManager.default.removeItem(at: dir) }

        var weights: SafetensorsWeights? =
            try SafetensorsWeights(shardedDirectory: dir)
        let borrowed = weights!["decoder.layer0.weight"]!
        weights = nil  // release our explicit ref

        // ARC should keep `weights` alive via `borrowed.owner`.
        XCTAssertEqual(borrowed.toArray(), [5, 6, 7, 8, 9, 10])
    }

    func testMissingIndexThrows() {
        let dir = tmpDir("no_index")
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try SafetensorsWeights(shardedDirectory: dir)) {
            guard case SafetensorsError.indexMalformed = $0 else {
                return XCTFail("expected .indexMalformed, got \($0)")
            }
        }
    }

    func testMissingShardThrows() throws {
        let dir = tmpDir("missing_shard")
        defer { try? FileManager.default.removeItem(at: dir) }

        // index.json points at a file that doesn't exist.
        let indexJSON: [String: Any] = [
            "weight_map": [
                "layer.weight": "does-not-exist.safetensors",
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: indexJSON, options: [])
        let indexURL = dir.appendingPathComponent("model.safetensors.index.json")
        try data.write(to: indexURL)

        XCTAssertThrowsError(try SafetensorsWeights(shardedDirectory: dir)) {
            guard case SafetensorsError.missingShard = $0 else {
                return XCTFail("expected .missingShard, got \($0)")
            }
        }
    }

    /// Regression: loading one shard via the single-file path and then
    /// the same file via the sharded path (in different tests) must
    /// not interfere. Runs a single-file load alongside a sharded load.
    func testSingleFileAndShardedCoexist() throws {
        let dir = try makeFixture()
        defer { try? FileManager.default.removeItem(at: dir) }

        let singleFilePath = dir.appendingPathComponent("model-00001-of-00002.safetensors").path
        let single = try SafetensorsWeights(path: singleFilePath)
        XCTAssertEqual(single.count, 2)

        let sharded = try SafetensorsWeights(shardedDirectory: dir)
        XCTAssertEqual(sharded.count, 4)

        // Both can be queried independently.
        XCTAssertEqual(single["encoder.layer0.bias"]!.toArray(), [10, 20])
        XCTAssertEqual(sharded["decoder.layer0.bias"]!.toArray(), [100, 200, 300])
    }
}
