import XCTest
import Foundation
@testable import Transcribe

final class TransformerLayerTests: XCTestCase {

    private func tmp(_ name: String) -> String {
        (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("ml_transcribe_xfmr_\(name).safetensors")
    }

    private func writeFixture(
        _ pairs: [(String, [Float], [Int])],
        to path: String
    ) throws {
        let tensors: [(String, Tensor)] = try pairs.map { (name, data, shape) in
            (name, try Tensor.from(data: data, shape: shape))
        }
        try SafetensorsWeights.save(tensors, to: path)
    }

    /// Build a valid weight set: all projection weights/biases are zero,
    /// LayerNorm gammas are 1 and betas are 0. With zero projections,
    /// both sub-layers contribute zero and the residuals pass through
    /// unchanged, so `forward(x) == x` exactly.
    private func zeroWeights(config: TransformerLayer.Config) -> [(String, [Float], [Int])] {
        let D = config.modelDim
        let F = config.ffnDim

        func zeros(_ n: Int) -> [Float] { Array(repeating: 0, count: n) }
        func ones(_ n: Int) -> [Float]  { Array(repeating: 1, count: n) }

        return [
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
    }

    /// Randomly initialized weights with a seeded RNG — deterministic
    /// across runs but non-trivial, so the forward pass actually does
    /// something. Uses a small std so LayerNorm's `x - mean` stays
    /// well-scaled and avoids blow-up during the first forward.
    private func randomWeights(
        config: TransformerLayer.Config,
        seed: UInt64
    ) -> [(String, [Float], [Int])] {
        let D = config.modelDim
        let F = config.ffnDim
        var rng = SeededRNG(seed: seed)
        func randn(_ n: Int, scale: Float) -> [Float] {
            (0..<n).map { _ in Float.random(in: -scale...scale, using: &rng) }
        }
        return [
            ("ln1.gamma", Array(repeating: Float(1), count: D), [D]),
            ("ln1.beta",  Array(repeating: Float(0), count: D), [D]),

            ("attn.q_proj.weight", randn(D * D, scale: 0.1), [D, D]),
            ("attn.q_proj.bias",   randn(D,     scale: 0.0), [1, 1, D]),
            ("attn.k_proj.weight", randn(D * D, scale: 0.1), [D, D]),
            ("attn.k_proj.bias",   randn(D,     scale: 0.0), [1, 1, D]),
            ("attn.v_proj.weight", randn(D * D, scale: 0.1), [D, D]),
            ("attn.v_proj.bias",   randn(D,     scale: 0.0), [1, 1, D]),
            ("attn.o_proj.weight", randn(D * D, scale: 0.1), [D, D]),
            ("attn.o_proj.bias",   randn(D,     scale: 0.0), [1, 1, D]),

            ("ln2.gamma", Array(repeating: Float(1), count: D), [D]),
            ("ln2.beta",  Array(repeating: Float(0), count: D), [D]),

            ("ffn.up_proj.weight",   randn(D * F, scale: 0.1), [D, F]),
            ("ffn.up_proj.bias",     randn(F,     scale: 0.0), [1, 1, F]),
            ("ffn.down_proj.weight", randn(F * D, scale: 0.1), [F, D]),
            ("ffn.down_proj.bias",   randn(D,     scale: 0.0), [1, 1, D]),
        ]
    }

    // MARK: - Tests

    /// With all projection weights zero, BOTH sub-layers produce zero.
    /// The two residual adds pass the input through unchanged — a clean
    /// test that the residual connections are wired correctly.
    func testZeroWeightsResidualIsIdentity() throws {
        let path = tmp("residual_identity")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let config = TransformerLayer.Config(modelDim: 4, numHeads: 2, ffnDim: 8)
        try writeFixture(zeroWeights(config: config), to: path)

        let weights = try SafetensorsWeights(path: path)
        let layer = try TransformerLayer(weights: weights, config: config)

        let x = try Tensor.from(
            data: [ 1, -2,  3, -4,
                    5, -6,  7, -8,
                    9, -10, 11, -12],
            shape: [1, 3, 4]
        )
        let y = layer.forward(x)

        XCTAssertEqual(y.shape, x.shape)
        // Allow a tiny floating-point drift (LayerNorm does actual math
        // even though the downstream projections zero it out).
        let yArr = y.toArray()
        let xArr = x.toArray()
        for (a, b) in zip(xArr, yArr) {
            XCTAssertEqual(a, b, accuracy: 1e-4)
        }
    }

    /// With real non-zero weights, the output must still be the right
    /// shape and entirely finite, but it should generally differ from
    /// the input (otherwise both sub-layers are dead).
    func testRandomWeightsProduceFiniteOutput() throws {
        let path = tmp("random_weights")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let config = TransformerLayer.Config(modelDim: 8, numHeads: 2, ffnDim: 16)
        try writeFixture(randomWeights(config: config, seed: 7), to: path)

        let weights = try SafetensorsWeights(path: path)
        let layer = try TransformerLayer(weights: weights, config: config)

        var rng = SeededRNG(seed: 99)
        let B = 2, S = 5
        let input: [Float] = (0..<(B * S * config.modelDim)).map { _ in
            Float.random(in: -1...1, using: &rng)
        }
        let x = try Tensor.from(data: input, shape: [B, S, config.modelDim])
        let y = layer.forward(x)

        XCTAssertEqual(y.shape, [B, S, config.modelDim])
        let yArr = y.toArray()
        XCTAssertEqual(yArr.count, B * S * config.modelDim)
        for v in yArr {
            XCTAssertTrue(v.isFinite, "non-finite output: \(v)")
        }

        // At least ONE element should differ noticeably — otherwise
        // we've accidentally reduced to the identity again.
        let changed = zip(input, yArr).contains { abs($0 - $1) > 1e-3 }
        XCTAssertTrue(changed, "forward pass was effectively identity")
    }

    /// Forward is deterministic: same input → same output, every time.
    func testForwardIsDeterministic() throws {
        let path = tmp("determinism")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let config = TransformerLayer.Config(modelDim: 4, numHeads: 2, ffnDim: 8)
        try writeFixture(randomWeights(config: config, seed: 42), to: path)

        let weights = try SafetensorsWeights(path: path)
        let layer = try TransformerLayer(weights: weights, config: config)

        let x = try Tensor.from(
            data: [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
            shape: [1, 2, 4]
        )
        let first = layer.forward(x).toArray()
        for _ in 0..<3 {
            let again = layer.forward(x).toArray()
            XCTAssertEqual(first, again)
        }
    }

    /// If a weight has the wrong shape, construction throws a helpful
    /// error rather than panicking later inside an op.
    func testShapeMismatchSurfacesLoadError() throws {
        let path = tmp("bad_shape")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let config = TransformerLayer.Config(modelDim: 4, numHeads: 2, ffnDim: 8)
        var fixtures = zeroWeights(config: config)
        // Corrupt one shape: q_proj.weight should be [4, 4], not [2, 8].
        for i in fixtures.indices where fixtures[i].0 == "attn.q_proj.weight" {
            fixtures[i] = ("attn.q_proj.weight",
                           Array(repeating: 0, count: 16), [2, 8])
        }
        try writeFixture(fixtures, to: path)

        let weights = try SafetensorsWeights(path: path)
        XCTAssertThrowsError(try TransformerLayer(weights: weights, config: config)) {
            guard let err = $0 as? TransformerLayer.LoadError else {
                return XCTFail("expected TransformerLayer.LoadError, got \($0)")
            }
            if case .unexpectedShape(let name, _, _) = err {
                XCTAssertEqual(name, "attn.q_proj.weight")
            } else {
                XCTFail("expected .unexpectedShape, got \(err)")
            }
        }
    }
}
