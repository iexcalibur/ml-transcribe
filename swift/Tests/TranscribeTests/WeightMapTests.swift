import XCTest
import Foundation
@testable import Transcribe

final class WeightMapTests: XCTestCase {

    private func tmp(_ name: String) -> String {
        (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("ml_transcribe_wm_\(name)_\(UUID().uuidString).safetensors")
    }

    // MARK: - Basic behavior

    /// With no rename rules and no transposes, WeightMap is a
    /// transparent pass-through.
    func testNoRenamesPassesThrough() throws {
        let path = tmp("passthrough")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let a = try Tensor.from(data: [1, 2, 3, 4], shape: [2, 2])
        let b = try Tensor.from(data: [10, 20], shape: [2])
        try SafetensorsWeights.save(
            [("a", a), ("b", b)], to: path
        )

        let raw = try SafetensorsWeights(path: path)
        let map = WeightMap(source: raw)

        XCTAssertEqual(map.count, 2)
        XCTAssertEqual(map.keys, ["a", "b"])
        XCTAssertEqual(map["a"]?.toArray(), [1, 2, 3, 4])
        XCTAssertEqual(map["b"]?.toArray(), [10, 20])
        XCTAssertNil(map["missing"])
    }

    /// With a rename, looking up the our-name returns the tensor at
    /// the source-name; the source-name itself is no longer visible.
    func testExactRename() throws {
        let path = tmp("rename")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let t = try Tensor.from(data: [1, 2, 3, 4], shape: [2, 2])
        try SafetensorsWeights.save([("model.embed_tokens.weight", t)], to: path)

        let map = WeightMap(
            source: try SafetensorsWeights(path: path),
            renames: ["embedding": "model.embed_tokens.weight"]
        )

        XCTAssertEqual(map.keys, ["embedding"])
        XCTAssertEqual(map["embedding"]?.toArray(), [1, 2, 3, 4])
        // Source name is no longer exposed.
        XCTAssertNil(map["model.embed_tokens.weight"])
    }

    /// Renamed tensor + unrenamed tensor coexist — `keys` shows both.
    func testKeysMixesRenamesAndPassThrough() throws {
        let path = tmp("mix_keys")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let a = try Tensor.from(data: [1, 2], shape: [2])
        let b = try Tensor.from(data: [3, 4], shape: [2])
        try SafetensorsWeights.save([("old_name", a), ("keep_me", b)], to: path)

        let map = WeightMap(
            source: try SafetensorsWeights(path: path),
            renames: ["new_name": "old_name"]
        )

        XCTAssertEqual(map.keys, ["keep_me", "new_name"])
        XCTAssertEqual(map["new_name"]?.toArray(), [1, 2])
        XCTAssertEqual(map["keep_me"]?.toArray(), [3, 4])
    }

    // MARK: - Transpose

    /// A transpose-tagged key: shape is `[rows, cols]` → `[cols, rows]`
    /// and data is reordered to the new strides.
    func testTransposeSwapsShape() throws {
        let path = tmp("transpose")
        defer { try? FileManager.default.removeItem(atPath: path) }

        // Save as [2, 3]: row-major = [1,2,3,4,5,6].
        let t = try Tensor.from(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
        try SafetensorsWeights.save([("w", t)], to: path)

        let map = WeightMap(
            source: try SafetensorsWeights(path: path),
            transpose: ["w"]
        )

        let out = map["w"]!
        // Expected [3, 2] with [1, 4, 2, 5, 3, 6] (each original column
        // becomes a row).
        XCTAssertEqual(out.shape, [3, 2])
        XCTAssertEqual(out.toArray(), [1, 4, 2, 5, 3, 6])
    }

    /// Rename + transpose can both apply to the same key.
    func testRenameAndTransposeCombine() throws {
        let path = tmp("rename_transpose")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let t = try Tensor.from(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
        try SafetensorsWeights.save([("pytorch.style", t)], to: path)

        let map = WeightMap(
            source: try SafetensorsWeights(path: path),
            renames: ["our.style": "pytorch.style"],
            transpose: ["our.style"]
        )

        XCTAssertEqual(map["our.style"]?.shape, [3, 2])
        XCTAssertEqual(map["our.style"]?.toArray(), [1, 4, 2, 5, 3, 6])
    }

    /// Transposed tensors are cached (eager) — the returned Tensor
    /// instance is the SAME across two subscript calls.
    func testTransposedTensorIsCached() throws {
        let path = tmp("cached")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let t = try Tensor.from(data: [1, 2, 3, 4], shape: [2, 2])
        try SafetensorsWeights.save([("w", t)], to: path)

        let map = WeightMap(
            source: try SafetensorsWeights(path: path),
            transpose: ["w"]
        )
        let a = map["w"]!
        let b = map["w"]!
        XCTAssertTrue(a === b, "cached Tensor should be reused across lookups")
    }

    // MARK: - Integration with DecoderLM

    /// End-to-end: save weights with HF-style names, wrap with a
    /// WeightMap that renames to our convention, pass to DecoderLM
    /// and run the identity-echo test. If names AND subscript routing
    /// weren't both right, this would fail.
    func testDecoderLMLoadsViaWeightMapWithHFNames() throws {
        let path = tmp("hf_named_weights")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let config = DecoderLM.Config.gptStyle(
            vocabSize: 8, modelDim: 8, numHeads: 2, ffnDim: 16,
            numLayers: 1, maxSeqLen: 16
        )

        // Build identity-echo fixtures, but under HF-style names.
        let D = config.modelDim
        let F = config.ffnDim
        let V = config.vocabSize
        precondition(V == D)

        func zeros(_ n: Int) -> [Float] { Array(repeating: 0, count: n) }
        func ones(_ n: Int)  -> [Float] { Array(repeating: 1, count: n) }
        func eye(_ n: Int) -> [Float] {
            var m = [Float](repeating: 0, count: n * n)
            for i in 0..<n { m[i * n + i] = 1 }
            return m
        }

        let pairs: [(String, [Float], [Int])] = [
            // HF-style global keys
            ("hf.embedding",     eye(D),   [V, D]),
            ("hf.final_norm.w",  ones(D),  [D]),
            ("hf.final_norm.b",  zeros(D), [D]),
            ("hf.lm_head",       eye(D),   [D, V]),
            // HF-style layer 0 keys
            ("hf.L0.norm1.w",    ones(D),  [D]),
            ("hf.L0.norm1.b",    zeros(D), [D]),
            ("hf.L0.attn.q",     zeros(D * D), [D, D]),
            ("hf.L0.attn.q.b",   zeros(D),     [1, 1, D]),
            ("hf.L0.attn.k",     zeros(D * D), [D, D]),
            ("hf.L0.attn.k.b",   zeros(D),     [1, 1, D]),
            ("hf.L0.attn.v",     zeros(D * D), [D, D]),
            ("hf.L0.attn.v.b",   zeros(D),     [1, 1, D]),
            ("hf.L0.attn.o",     zeros(D * D), [D, D]),
            ("hf.L0.attn.o.b",   zeros(D),     [1, 1, D]),
            ("hf.L0.norm2.w",    ones(D),  [D]),
            ("hf.L0.norm2.b",    zeros(D), [D]),
            ("hf.L0.ffn.up",     zeros(D * F), [D, F]),
            ("hf.L0.ffn.up.b",   zeros(F),     [1, 1, F]),
            ("hf.L0.ffn.down",   zeros(F * D), [F, D]),
            ("hf.L0.ffn.down.b", zeros(D),     [1, 1, D]),
        ]
        let tensors: [(String, Tensor)] = try pairs.map { name, data, shape in
            (name, try Tensor.from(data: data, shape: shape))
        }
        try SafetensorsWeights.save(tensors, to: path)

        // Build the rename map: our-convention name → HF-on-disk name.
        let renames: [String: String] = [
            "embedding":         "hf.embedding",
            "final_norm.weight": "hf.final_norm.w",
            "final_norm.bias":   "hf.final_norm.b",
            "lm_head":           "hf.lm_head",
            "layers.0.norm1.weight":        "hf.L0.norm1.w",
            "layers.0.norm1.bias":          "hf.L0.norm1.b",
            "layers.0.attn.q_proj.weight":  "hf.L0.attn.q",
            "layers.0.attn.q_proj.bias":    "hf.L0.attn.q.b",
            "layers.0.attn.k_proj.weight":  "hf.L0.attn.k",
            "layers.0.attn.k_proj.bias":    "hf.L0.attn.k.b",
            "layers.0.attn.v_proj.weight":  "hf.L0.attn.v",
            "layers.0.attn.v_proj.bias":    "hf.L0.attn.v.b",
            "layers.0.attn.o_proj.weight":  "hf.L0.attn.o",
            "layers.0.attn.o_proj.bias":    "hf.L0.attn.o.b",
            "layers.0.norm2.weight":        "hf.L0.norm2.w",
            "layers.0.norm2.bias":          "hf.L0.norm2.b",
            "layers.0.ffn.up_proj.weight":  "hf.L0.ffn.up",
            "layers.0.ffn.up_proj.bias":    "hf.L0.ffn.up.b",
            "layers.0.ffn.down_proj.weight":"hf.L0.ffn.down",
            "layers.0.ffn.down_proj.bias":  "hf.L0.ffn.down.b",
        ]

        let raw = try SafetensorsWeights(path: path)
        let weights = WeightMap(source: raw, renames: renames)

        let lm = try DecoderLM(weights: weights, config: config)
        for t in 0..<config.vocabSize {
            lm.reset()
            XCTAssertEqual(
                try lm.step(tokenId: t), t,
                "DecoderLM should still echo via a WeightMap — token \(t)")
        }
    }

    // MARK: - Convenience builders

    func testLlamaRenameBuilderCoversAllLayers() {
        let renames = WeightMap.huggingFaceLlamaRenames(numLayers: 3)
        XCTAssertEqual(renames["embedding"], "model.embed_tokens.weight")
        XCTAssertEqual(renames["final_norm.weight"], "model.norm.weight")
        XCTAssertEqual(renames["lm_head"], "lm_head.weight")
        for i in 0..<3 {
            XCTAssertEqual(
                renames["layers.\(i).norm1.weight"],
                "model.layers.\(i).input_layernorm.weight")
            XCTAssertEqual(
                renames["layers.\(i).ffn.gate_proj.weight"],
                "model.layers.\(i).mlp.gate_proj.weight")
        }
    }

    func testLlamaTransposeSetCoversExpectedKeys() {
        let s = WeightMap.huggingFaceLlamaTransposeSet(numLayers: 2)
        // Global
        XCTAssertTrue(s.contains("lm_head"))
        // Per-layer — just probe a few.
        XCTAssertTrue(s.contains("layers.0.attn.q_proj.weight"))
        XCTAssertTrue(s.contains("layers.1.ffn.down_proj.weight"))
        // Norm weights are NOT transposed (they're 1-D).
        XCTAssertFalse(s.contains("layers.0.norm1.weight"))
        XCTAssertFalse(s.contains("final_norm.weight"))
    }
}
