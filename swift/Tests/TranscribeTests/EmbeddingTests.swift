import XCTest
import Foundation
@testable import Transcribe

final class EmbeddingTests: XCTestCase {

    /// Helper to build a small embedding table where row i = [i, i+0.5, ...]
    /// with `dim` values. Makes it easy to verify which row was returned.
    private func makeTable(vocab: Int, dim: Int) throws -> Tensor {
        var data = [Float](); data.reserveCapacity(vocab * dim)
        for i in 0..<vocab {
            for j in 0..<dim {
                data.append(Float(i) + Float(j) * 0.01)
            }
        }
        return try Tensor.from(data: data, shape: [vocab, dim])
    }

    // MARK: - Correctness

    func testSingleSequenceLookup() throws {
        // Table rows are distinct by integer part, so we can just check
        // the integer part of each output slice matches the requested id.
        let vocab = 10, dim = 4
        let table = try makeTable(vocab: vocab, dim: dim)

        // Tokens [3, 0, 7] → shape [1, 3, 4].
        let out = table.embed(tokens: [3, 0, 7])
        XCTAssertEqual(out.shape, [1, 3, 4])
        let arr = out.toArray()

        // First output row should be row 3 of the table: [3.0, 3.01, 3.02, 3.03].
        for j in 0..<dim {
            XCTAssertEqual(arr[j], Float(3) + Float(j) * 0.01, accuracy: 1e-6)
        }
        // Second: row 0 = [0, 0.01, 0.02, 0.03].
        for j in 0..<dim {
            XCTAssertEqual(arr[dim + j], Float(0) + Float(j) * 0.01, accuracy: 1e-6)
        }
        // Third: row 7.
        for j in 0..<dim {
            XCTAssertEqual(arr[2 * dim + j], Float(7) + Float(j) * 0.01, accuracy: 1e-6)
        }
    }

    func testBatchedLookup() throws {
        let vocab = 5, dim = 2
        let table = try makeTable(vocab: vocab, dim: dim)

        // Two sequences of length 3: batch=2, seq=3.
        let out = table.embed(tokens: [
            [4, 2, 0],
            [1, 1, 3],
        ])
        XCTAssertEqual(out.shape, [2, 3, 2])
        let arr = out.toArray()

        // Row 4: [4, 4.01]; row 2: [2, 2.01]; row 0: [0, 0.01];
        // row 1: [1, 1.01] (twice); row 3: [3, 3.01].
        let expected: [Float] = [
            4, 4.01, 2, 2.01, 0, 0.01,
            1, 1.01, 1, 1.01, 3, 3.01,
        ]
        for (a, b) in zip(arr, expected) {
            XCTAssertEqual(a, b, accuracy: 1e-5)
        }
    }

    func testRepeatedTokensReturnSameRow() throws {
        let table = try makeTable(vocab: 3, dim: 4)
        let out = table.embed(tokens: [1, 1, 1, 1])
        let arr = out.toArray()
        // All four positions should be row 1 = [1, 1.01, 1.02, 1.03].
        for t in 0..<4 {
            for j in 0..<4 {
                XCTAssertEqual(arr[t * 4 + j], 1 + Float(j) * 0.01, accuracy: 1e-6)
            }
        }
    }

    // MARK: - Composition with TransformerLayer

    /// Run: token_ids -> embeddings -> transformer layer -> output.
    /// Proves the full decoder-only forward path composes end to end.
    func testEmbedThenTransformerLayer() throws {
        // Build a transformer layer with zero projection weights so the
        // residuals pass through. Since the layer is effectively the
        // identity, final output should equal the embeddings (which
        // equal specific table rows).
        let D = 4, H = 2, F = 8
        let vocab = 16
        let config = TransformerLayer.Config(modelDim: D, numHeads: H, ffnDim: F)

        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("ml_transcribe_embed_xfmr.safetensors")
        defer { try? FileManager.default.removeItem(atPath: path) }

        // Reuse the zero-weights builder's logic inline.
        func zeros(_ n: Int) -> [Float] { Array(repeating: 0, count: n) }
        func ones(_ n: Int) -> [Float]  { Array(repeating: 1, count: n) }
        let fixtures: [(String, [Float], [Int])] = [
            ("ln1.gamma", ones(D),  [D]),
            ("ln1.beta",  zeros(D), [D]),
            ("attn.q_proj.weight", zeros(D * D), [D, D]),
            ("attn.q_proj.bias",   zeros(D),     [1, 1, D]),
            ("attn.k_proj.weight", zeros(D * D), [D, D]),
            ("attn.k_proj.bias",   zeros(D),     [1, 1, D]),
            ("attn.v_proj.weight", zeros(D * D), [D, D]),
            ("attn.v_proj.bias",   zeros(D),     [1, 1, D]),
            ("attn.o_proj.weight", zeros(D * D), [D, D]),
            ("attn.o_proj.bias",   zeros(D),     [1, 1, D]),
            ("ln2.gamma", ones(D),  [D]),
            ("ln2.beta",  zeros(D), [D]),
            ("ffn.up_proj.weight",   zeros(D * F), [D, F]),
            ("ffn.up_proj.bias",     zeros(F),     [1, 1, F]),
            ("ffn.down_proj.weight", zeros(F * D), [F, D]),
            ("ffn.down_proj.bias",   zeros(D),     [1, 1, D]),
        ]
        let tensors: [(String, Tensor)] = try fixtures.map { name, data, shape in
            (name, try Tensor.from(data: data, shape: shape))
        }
        try SafetensorsWeights.save(tensors, to: path)

        let weights = try SafetensorsWeights(path: path)
        let layer = try TransformerLayer(weights: weights, config: config)

        let table = try makeTable(vocab: vocab, dim: D)

        // Embedding: [1, 5, D]. Tokens: [0, 1, 2, 3, 4].
        let x = table.embed(tokens: [0, 1, 2, 3, 4])
        XCTAssertEqual(x.shape, [1, 5, D])

        let y = layer.forward(x)
        XCTAssertEqual(y.shape, [1, 5, D])

        // With zero weights, the layer should be (approximately) identity.
        let xArr = x.toArray()
        let yArr = y.toArray()
        for (a, b) in zip(xArr, yArr) {
            XCTAssertEqual(a, b, accuracy: 1e-4)
        }
    }

    // MARK: - Determinism

    func testEmbedIsDeterministic() throws {
        let table = try makeTable(vocab: 8, dim: 3)
        let first = table.embed(tokens: [7, 3, 1, 0, 5]).toArray()
        for _ in 0..<3 {
            XCTAssertEqual(table.embed(tokens: [7, 3, 1, 0, 5]).toArray(), first)
        }
    }
}
