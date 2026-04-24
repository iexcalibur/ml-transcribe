import XCTest
@testable import Transcribe

/// Correctness tests for layout ops (reshape, permute, transpose,
/// contiguous), culminating in a full multi-head attention flow.
final class LayoutOpsTests: XCTestCase {

    // MARK: - Reshape

    func testReshapeBasic() throws {
        // [6] -> [2, 3] -> [6] preserves values and order (row-major).
        let a = try Tensor.from(data: [1, 2, 3, 4, 5, 6], shape: [6])
        let b = a.reshape([2, 3])
        XCTAssertEqual(b.shape, [2, 3])
        XCTAssertEqual(b.toArray(), [1, 2, 3, 4, 5, 6])

        let c = b.reshape([6])
        XCTAssertEqual(c.shape, [6])
        XCTAssertEqual(c.toArray(), [1, 2, 3, 4, 5, 6])
    }

    func testReshapeToThreeDim() throws {
        // [8] -> [2, 2, 2].
        let a = try Tensor.from(data: [1, 2, 3, 4, 5, 6, 7, 8], shape: [8])
        let b = a.reshape([2, 2, 2])
        XCTAssertEqual(b.shape, [2, 2, 2])
        // Row-major: same linear order.
        XCTAssertEqual(b.toArray(), [1, 2, 3, 4, 5, 6, 7, 8])
    }

    // MARK: - Permute / transpose

    func testTranspose2D() throws {
        // [2, 3]:                    [3, 2] (after transpose(0,1) + contiguous):
        //   1 2 3                      1 4
        //   4 5 6                      2 5
        //                              3 6
        let a = try Tensor.from(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
        let t = a.transpose(0, 1).contiguous()
        XCTAssertEqual(t.shape, [3, 2])
        XCTAssertEqual(t.toArray(), [1, 4, 2, 5, 3, 6])
    }

    func testPermuteThreeDim() throws {
        // [2, 2, 3] with values 1..12, permute [0, 2, 1] swaps last 2 dims.
        // Input (row-major):
        //   block 0:  [[1, 2, 3], [4, 5, 6]]
        //   block 1:  [[7, 8, 9], [10, 11, 12]]
        // After permute([0, 2, 1]) -> shape [2, 3, 2]:
        //   block 0:  [[1, 4], [2, 5], [3, 6]]
        //   block 1:  [[7, 10], [8, 11], [9, 12]]
        let a = try Tensor.from(data: Array(1...12).map(Float.init),
                                shape: [2, 2, 3])
        let p = a.permute([0, 2, 1]).contiguous()
        XCTAssertEqual(p.shape, [2, 3, 2])
        XCTAssertEqual(p.toArray(),
            [1, 4, 2, 5, 3, 6,
             7, 10, 8, 11, 9, 12])
    }

    func testDoubleTransposeIsIdentity() throws {
        // transpose(i, j) twice returns the original tensor.
        let a = try Tensor.from(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
        let t = a.transpose(0, 1).transpose(0, 1).contiguous()
        XCTAssertEqual(t.shape, [2, 3])
        XCTAssertEqual(t.toArray(), [1, 2, 3, 4, 5, 6])
    }

    // MARK: - Contiguous

    func testContiguousIsNoOpOnFreshTensor() throws {
        // A freshly-constructed tensor is already contiguous; calling
        // contiguous() produces a value-equal tensor.
        let a = try Tensor.from(data: [1, 2, 3, 4], shape: [2, 2])
        let c = a.contiguous()
        XCTAssertEqual(c.shape, [2, 2])
        XCTAssertEqual(c.toArray(), [1, 2, 3, 4])
    }

    // MARK: - Multi-head attention flow

    /// The canonical reshape pattern for multi-head attention:
    ///
    ///   x   : [B, S, H*D]       ← projected queries (or K, V)
    ///   -> reshape  [B, S, H, D]
    ///   -> permute  [0, 2, 1, 3]  → [B, H, S, D]
    ///   -> reshape  [B*H, S, D]   ← ready for attention(...)
    ///   ... attention ...
    ///   -> reshape  [B, H, S, D]  ← un-flatten heads
    ///   -> permute  [0, 2, 1, 3]  → [B, S, H, D]
    ///   -> reshape  [B, S, H*D]   ← back to original layout
    ///
    /// The round-trip (reshape → permute → permute → reshape) must
    /// produce the exact original tensor.
    func testMultiHeadReshapeRoundTrip() throws {
        // B=1, S=2, H=2, D=2 → H*D = 4. 8 values total.
        let B = 1, S = 2, H = 2, D = 2
        let x = try Tensor.from(
            data: [1, 2, 3, 4,
                   5, 6, 7, 8],
            shape: [B, S, H * D]
        )
        XCTAssertEqual(x.shape, [1, 2, 4])

        // Split heads.
        let split = x
            .reshape([B, S, H, D])
            .permute([0, 2, 1, 3])        // [B, H, S, D]
            .contiguous()
            .reshape([B * H, S, D])        // [2, 2, 2]
        XCTAssertEqual(split.shape, [2, 2, 2])

        // Rejoin heads — exact inverse.
        let rejoined = split
            .reshape([B, H, S, D])
            .permute([0, 2, 1, 3])         // [B, S, H, D]
            .contiguous()
            .reshape([B, S, H * D])
        XCTAssertEqual(rejoined.shape, [1, 2, 4])
        XCTAssertEqual(rejoined.toArray(),
            [1, 2, 3, 4,
             5, 6, 7, 8])
    }

    /// Compose the full multi-head self-attention path:
    ///
    ///   x projected to q, k, v already in [B, S, H*D];
    ///   split heads  →  attention per head  →  merge heads.
    ///
    /// We verify the OUTPUT SHAPE matches the input shape and no
    /// element is NaN/inf. Exact values are hard to hand-check for
    /// H>1, so correctness of the math is covered by single-head
    /// attention tests (in TransformerOpsTests) and the reshape
    /// round-trip test above.
    func testMultiHeadAttentionComposes() throws {
        let B = 1, S = 3, H = 2, D = 2  // model dim = 4
        let modelDim = H * D

        // Fixed-but-varied values so attention has something to do.
        let q = try Tensor.from(
            data: [1, 0, 0, 1,
                   0, 1, 1, 0,
                   1, 1, 0, 0],
            shape: [B, S, modelDim]
        )
        // k = q for simplicity (self-attention).
        let k = try Tensor.from(
            data: [1, 0, 0, 1,
                   0, 1, 1, 0,
                   1, 1, 0, 0],
            shape: [B, S, modelDim]
        )
        let v = try Tensor.from(
            data: [10, 20, 30, 40,
                   50, 60, 70, 80,
                   90, 100, 110, 120],
            shape: [B, S, modelDim]
        )

        let scale = 1.0 / sqrt(Float(D))

        func splitHeads(_ t: Tensor) -> Tensor {
            return t
                .reshape([B, S, H, D])
                .permute([0, 2, 1, 3])
                .contiguous()
                .reshape([B * H, S, D])
        }
        func mergeHeads(_ t: Tensor) -> Tensor {
            return t
                .reshape([B, H, S, D])
                .permute([0, 2, 1, 3])
                .contiguous()
                .reshape([B, S, modelDim])
        }

        let out = mergeHeads(
            splitHeads(q).attention(
                key: splitHeads(k),
                value: splitHeads(v),
                scale: scale,
                causal: false
            )
        )

        XCTAssertEqual(out.shape, [B, S, modelDim])
        let arr = out.toArray()
        XCTAssertEqual(arr.count, B * S * modelDim)
        for v in arr {
            XCTAssertTrue(v.isFinite, "non-finite output: \(v)")
        }
    }
}
