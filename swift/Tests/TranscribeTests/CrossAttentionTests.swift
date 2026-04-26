import XCTest
import Foundation
@testable import Transcribe

final class CrossAttentionTests: XCTestCase {

    // MARK: - Mathematical correctness

    /// All-zero K → uniform attention → output is the mean of V along
    /// the K-axis. (softmax([0,0,...]) = [1/n, 1/n, ...].)
    func testUniformKeysAverageValues() throws {
        // BH=1, S_q=1, S_kv=4, D=2.
        let q = try Tensor.from(data: [1, 0], shape: [1, 1, 2])
        let k = try Tensor.from(
            data: Array(repeating: Float(0), count: 8), shape: [1, 4, 2]
        )
        let v = try Tensor.from(
            data: [10, 0,
                   20, 0,
                   30, 0,
                   40, 0],
            shape: [1, 4, 2]
        )
        let out = q.crossAttention(key: k, value: v, scale: 1.0).toArray()
        // Mean of V[:,0] = (10+20+30+40)/4 = 25; V[:,1] all zero.
        XCTAssertEqual(out[0], 25.0, accuracy: 1e-4)
        XCTAssertEqual(out[1], 0.0, accuracy: 1e-6)
    }

    /// Sharp Q-K match → output ≈ matched V row.
    /// Q=[1,0,0]; K rows are basis vectors; one row matches Q with
    /// large scale, so softmax peaks on it.
    func testSharpQKMatchSelectsValue() throws {
        let q = try Tensor.from(data: [1, 0, 0], shape: [1, 1, 3])
        let k = try Tensor.from(
            data: [1, 0, 0,
                   0, 1, 0,
                   0, 0, 1],
            shape: [1, 3, 3]
        )
        let v = try Tensor.from(
            data: [99, 11, 11,
                   1, 99, 11,
                   1, 11, 99],
            shape: [1, 3, 3]
        )
        // High scale → very peaked softmax on the matching row (row 0).
        let out = q.crossAttention(key: k, value: v, scale: 100.0).toArray()
        XCTAssertEqual(out[0], 99.0, accuracy: 0.5)
        XCTAssertEqual(out[1], 11.0, accuracy: 0.5)
        XCTAssertEqual(out[2], 11.0, accuracy: 0.5)
    }

    // MARK: - Shape behavior

    /// Q has S_q=2, K/V have S_kv=5: output should be [BH, 2, D]
    /// (Q's seq_len, NOT K's).
    func testOutputSeqLenMatchesQuery() throws {
        let q = try Tensor.from(data: [1, 0,  0, 1], shape: [1, 2, 2])
        let k = try Tensor.from(
            data: Array(repeating: Float(0.5), count: 5 * 2),
            shape: [1, 5, 2]
        )
        let v = try Tensor.from(
            data: Array(repeating: Float(7), count: 5 * 2),
            shape: [1, 5, 2]
        )
        let out = q.crossAttention(key: k, value: v, scale: 1.0)
        XCTAssertEqual(out.shape, [1, 2, 2])
    }

    /// Multi-batched-heads (BH=4) handled per-head independently:
    /// each batch slice should compute its own attention.
    func testMultiBatchHeadsHandledIndependently() throws {
        let bh = 4, sQ = 1, sKv = 2, d = 2
        // Each "head" gets the same Q=[1,0] but different V.
        // Heads' V[0] differ → outputs should differ.
        var qData: [Float] = []
        var kData: [Float] = []
        var vData: [Float] = []
        for b in 0..<bh {
            qData += [1, 0]
            kData += [1, 0,  0, 1]      // identical across heads
            vData += [Float(b * 10), 0,  Float(b + 100), 0]
        }
        let q = try Tensor.from(data: qData, shape: [bh, sQ, d])
        let k = try Tensor.from(data: kData, shape: [bh, sKv, d])
        let v = try Tensor.from(data: vData, shape: [bh, sKv, d])

        // Sharp peaking on K row 0 (matches Q=[1,0]).
        let out = q.crossAttention(key: k, value: v, scale: 100.0).toArray()
        for b in 0..<bh {
            XCTAssertEqual(out[b * d], Float(b * 10), accuracy: 0.5,
                "head \(b) should pick its own V[0]")
        }
    }

    // MARK: - Composition (mini decoder layer with cross-attention)

    /// A complete cross-attention sub-layer: project Q from a decoder
    /// hidden state, project K and V from a (fixed) encoder output,
    /// run cross-attention, project the result. This is the standard
    /// encoder-decoder pattern.
    func testCrossAttentionSublayerComposes() throws {
        let D = 8, B = 1, sQ = 2, sKv = 6
        // Pretend: the decoder's hidden h, the encoder's output e.
        let h = try Tensor.from(
            data: (0..<(B * sQ * D)).map { Float($0) * 0.01 },
            shape: [B, sQ, D]
        )
        let e = try Tensor.from(
            data: (0..<(B * sKv * D)).map { Float($0) * 0.1 },
            shape: [B, sKv, D]
        )

        // Random-like projections (just to get non-trivial outputs).
        var rng = SeededRNG(seed: 99)
        func randW() throws -> Tensor {
            let n = D * D
            return try Tensor.from(
                data: (0..<n).map { _ in Float.random(in: -0.1...0.1, using: &rng) },
                shape: [D, D]
            )
        }
        let wQ = try randW(), wK = try randW(), wV = try randW(), wO = try randW()

        // Compute Q from h, K and V from e.
        let q = h.matmul(wQ)    // [B, sQ, D]
        let k = e.matmul(wK)    // [B, sKv, D]
        let v = e.matmul(wV)    // [B, sKv, D]

        let attn = q.crossAttention(key: k, value: v, scale: 1.0 / sqrtf(Float(D)))
        XCTAssertEqual(attn.shape, [B, sQ, D])

        let out = attn.matmul(wO)
        XCTAssertEqual(out.shape, [B, sQ, D])
        for x in out.toArray() {
            XCTAssertTrue(x.isFinite, "non-finite cross-attention output")
        }
    }
}
