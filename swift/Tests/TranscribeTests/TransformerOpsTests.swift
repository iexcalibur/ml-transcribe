import XCTest
@testable import Transcribe

/// Correctness tests for the transformer-flavored ops (GELU, sigmoid,
/// SiLU, LayerNorm, scaled dot-product attention).
final class TransformerOpsTests: XCTestCase {

    // MARK: - Elementwise activations

    func testSigmoidKnownValues() throws {
        // sigmoid(0) = 0.5, sigmoid(+inf) -> 1, sigmoid(-inf) -> 0.
        // Large magnitudes saturate quickly.
        let x = try Tensor.from(data: [-100, -1, 0, 1, 100], shape: [5])
        let y = x.sigmoid().toArray()
        XCTAssertEqual(y[0], 0.0, accuracy: 1e-6)
        XCTAssertEqual(y[1], 0.26894143, accuracy: 1e-5)
        XCTAssertEqual(y[2], 0.5, accuracy: 1e-6)
        XCTAssertEqual(y[3], 0.7310586, accuracy: 1e-5)
        XCTAssertEqual(y[4], 1.0, accuracy: 1e-6)
    }

    func testGeluKnownValues() throws {
        // GELU(0) = 0, GELU(large positive) ≈ x, GELU(large negative) ≈ 0.
        // Exact GELU uses erf; framework likely uses tanh approximation.
        // Either way: at x = 0 it's 0; at x = 1 it's ~0.8413; at x = -1
        // it's ~-0.1587. Allow a loose tolerance to cover approximation.
        let x = try Tensor.from(data: [-2, -1, 0, 1, 2], shape: [5])
        let y = x.gelu().toArray()
        XCTAssertEqual(y[2], 0.0, accuracy: 1e-5)                   // exact
        XCTAssertEqual(y[3], 0.8413, accuracy: 1e-2)                // GELU(1)
        XCTAssertEqual(y[1], -0.1587, accuracy: 1e-2)               // GELU(-1)
        XCTAssertEqual(y[4], 1.9545, accuracy: 1e-2)                // GELU(2)
        XCTAssertEqual(y[0], -0.0455, accuracy: 1e-2)               // GELU(-2)
    }

    func testSiLUComposition() throws {
        // SiLU(x) = x * sigmoid(x). Verify against a hand-computed
        // set of values.
        let x = try Tensor.from(data: [-2, -1, 0, 1, 2], shape: [5])
        let y = x.silu().toArray()
        XCTAssertEqual(y[0], -2 * 0.11920, accuracy: 1e-4)
        XCTAssertEqual(y[1], -1 * 0.26894, accuracy: 1e-4)
        XCTAssertEqual(y[2], 0.0, accuracy: 1e-6)
        XCTAssertEqual(y[3], 1 * 0.73106, accuracy: 1e-4)
        XCTAssertEqual(y[4], 2 * 0.88080, accuracy: 1e-4)
    }

    // MARK: - LayerNorm

    func testLayerNormNormalizes() throws {
        // For gamma=1, beta=0, LayerNorm should produce mean 0 and
        // variance ≈ 1 along the last dim.
        let x = try Tensor.from(data: [1, 2, 3, 4], shape: [1, 4])
        let gamma = try Tensor.from(data: [1, 1, 1, 1], shape: [4])
        let beta = try Tensor.from(data: [0, 0, 0, 0], shape: [4])
        let y = x.layerNorm(gamma: gamma, beta: beta).toArray()

        // Mean across the last dim should be (very close to) 0.
        let mean = y.reduce(0, +) / Float(y.count)
        XCTAssertEqual(mean, 0.0, accuracy: 1e-5)

        // Variance (unbiased denom = n, matching LayerNorm's internal
        // calc) should be ~1.
        let variance = y.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(y.count)
        XCTAssertEqual(variance, 1.0, accuracy: 1e-3)
    }

    func testLayerNormGammaBetaApplied() throws {
        // LayerNorm with gamma=2, beta=10 should shift/scale after
        // normalizing: mean -> 10, variance -> 4.
        let x = try Tensor.from(data: [1, 2, 3, 4], shape: [1, 4])
        let gamma = try Tensor.from(data: [2, 2, 2, 2], shape: [4])
        let beta = try Tensor.from(data: [10, 10, 10, 10], shape: [4])
        let y = x.layerNorm(gamma: gamma, beta: beta).toArray()

        let mean = y.reduce(0, +) / Float(y.count)
        let variance = y.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(y.count)
        XCTAssertEqual(mean, 10.0, accuracy: 1e-4)
        XCTAssertEqual(variance, 4.0, accuracy: 1e-2)
    }

    // MARK: - Scaled dot-product attention

    func testAttentionUniformValuesAverageEvenly() throws {
        // Q = K = [[1], [1]], V = [[10], [20]].
        // Attention weights row 0: softmax([1, 1] * scale) = [0.5, 0.5]
        // Attention weights row 1: same
        // out[0] = 0.5*10 + 0.5*20 = 15
        // out[1] = 15
        // Shape: [BH=1, S=2, D=1]
        let q = try Tensor.from(data: [1, 1], shape: [1, 2, 1])
        let k = try Tensor.from(data: [1, 1], shape: [1, 2, 1])
        let v = try Tensor.from(data: [10, 20], shape: [1, 2, 1])
        let out = q.attention(key: k, value: v, scale: 1.0, causal: false).toArray()
        XCTAssertEqual(out[0], 15.0, accuracy: 1e-4)
        XCTAssertEqual(out[1], 15.0, accuracy: 1e-4)
    }

    func testAttentionCausalMasksFuture() throws {
        // Same Q, K, V but causal=true.
        // Row 0 can only see col 0: softmax([1]) * V[0] = 1 * 10 = 10
        // Row 1 sees both: 0.5*10 + 0.5*20 = 15
        let q = try Tensor.from(data: [1, 1], shape: [1, 2, 1])
        let k = try Tensor.from(data: [1, 1], shape: [1, 2, 1])
        let v = try Tensor.from(data: [10, 20], shape: [1, 2, 1])
        let out = q.attention(key: k, value: v, scale: 1.0, causal: true).toArray()
        XCTAssertEqual(out[0], 10.0, accuracy: 1e-4)
        XCTAssertEqual(out[1], 15.0, accuracy: 1e-4)
    }

    func testAttentionSelectsMatchingKey() throws {
        // Q = [[1, 0]] queries for "match the first feature".
        // K = [[1, 0], [0, 1]]:  key 0 matches, key 1 doesn't.
        // V = [[42, 0], [7, 0]]: value for key 0 is 42, for key 1 is 7.
        //
        // With a large scale, softmax is peaked on the matching key,
        // so the output should be very close to V[0] = [42, 0].
        let q = try Tensor.from(data: [1, 0], shape: [1, 1, 2])
        let k = try Tensor.from(data: [1, 0, 0, 1], shape: [1, 2, 2])
        let v = try Tensor.from(data: [42, 0, 7, 0], shape: [1, 2, 2])
        let out = q.attention(key: k, value: v, scale: 100.0, causal: false).toArray()

        // Output is [1, 1, 2]; should be ≈ [42, 0] up to softmax saturation.
        XCTAssertEqual(out[0], 42.0, accuracy: 1e-3)
        XCTAssertEqual(out[1], 0.0,  accuracy: 1e-3)
    }

    // MARK: - Composed: one transformer FFN block

    /// Exercises the pattern `LayerNorm -> Linear -> GELU -> Linear`,
    /// i.e. the feed-forward sub-block of a transformer layer.
    func testTransformerFFNBlockComposes() throws {
        // Pre-norm, hidden dim 4, expansion dim 8.
        // Fixed (not-random) weights so the result is deterministic
        // and the point of the test is "it runs without blowing up".
        let x = try Tensor.from(data: [0.5, -0.5, 1.0, -1.0], shape: [1, 4])
        let gamma = try Tensor.from(data: [1, 1, 1, 1], shape: [4])
        let beta = try Tensor.from(data: [0, 0, 0, 0], shape: [4])

        // Two linear layers with identity-ish projections.
        let w1 = try Tensor.from(
            data: Array(repeating: Float(0.1), count: 4 * 8),
            shape: [4, 8]
        )
        let b1 = try Tensor.from(data: Array(repeating: Float(0), count: 8), shape: [1, 8])
        let w2 = try Tensor.from(
            data: Array(repeating: Float(0.1), count: 8 * 4),
            shape: [8, 4]
        )
        let b2 = try Tensor.from(data: Array(repeating: Float(0), count: 4), shape: [1, 4])

        let out = x
            .layerNorm(gamma: gamma, beta: beta)
            .matmul(w1).add(b1)
            .gelu()
            .matmul(w2).add(b2)

        XCTAssertEqual(out.shape, [1, 4])
        // All values should be finite (not NaN or infinity).
        let arr = out.toArray()
        for v in arr {
            XCTAssertTrue(v.isFinite, "non-finite output: \(v)")
        }
    }
}
