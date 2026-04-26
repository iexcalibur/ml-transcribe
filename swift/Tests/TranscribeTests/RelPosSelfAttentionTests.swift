import XCTest
import Foundation
@testable import Transcribe

/// Tests for `Tensor.relShift`, the sinusoidal pos-emb generator,
/// and the full `RelPosSelfAttention` forward pass.
final class RelPosSelfAttentionTests: XCTestCase {

    // MARK: - rel_shift gather correctness

    /// For T=3, NeMo's `rel_shift` produces the gather
    ///
    ///     out[0, 0, i, k] = x[0, 0, i, (T-1) - i + k] = x[0, 0, i, 2 - i + k]
    ///
    /// With encoded values `x[0, 0, i, j] = 10*i + j`, that produces
    /// the row-by-row pattern:
    ///   i=0: cols [2, 3, 4]
    ///   i=1: cols [1, 2, 3]
    ///   i=2: cols [0, 1, 2]
    ///
    /// (Equivalent to the pad-then-reshape trick in NeMo's source.)
    func testRelShiftSmallHandComputed() throws {
        let T = 3
        let posLen = 2 * T - 1     // 5
        var data = [Float]()
        for i in 0..<T {
            for j in 0..<posLen {
                data.append(Float(10 * i + j))
            }
        }
        let x = try Tensor.from(data: data, shape: [1, 1, T, posLen])
        let out = x.relShift().toArray()

        let expected: [Float] = [
            /* i=0 */  2,  3,  4,
            /* i=1 */ 11, 12, 13,
            /* i=2 */ 20, 21, 22,
        ]
        XCTAssertEqual(out, expected)
    }

    /// Verify the shape contract: input `[B, H, T, 2T-1]` → `[B, H, T, T]`.
    func testRelShiftShapeOutput() throws {
        let B = 2, H = 4, T = 5
        let posLen = 2 * T - 1
        let data = [Float](repeating: 1.0, count: B * H * T * posLen)
        let x = try Tensor.from(data: data, shape: [B, H, T, posLen])
        let out = x.relShift()
        XCTAssertEqual(out.shape, [B, H, T, T])
    }

    /// rel_shift treats each (batch, head) slice independently.
    /// Two heads with different inputs should produce different outputs.
    func testRelShiftPerHeadIndependence() throws {
        let T = 3, posLen = 2 * T - 1
        // Head 0 = all 1s; head 1 = all 7s. Output should be all 1s for
        // head 0 and all 7s for head 1.
        var data = [Float]()
        data += [Float](repeating: 1, count: T * posLen)
        data += [Float](repeating: 7, count: T * posLen)
        let x = try Tensor.from(data: data, shape: [1, 2, T, posLen])
        let out = x.relShift().toArray()
        for k in 0..<(T * T) {
            XCTAssertEqual(out[k], 1)
        }
        for k in (T * T)..<(2 * T * T) {
            XCTAssertEqual(out[k], 7)
        }
    }

    // MARK: - Sinusoidal pos_emb layout

    /// The middle row (index T-1) corresponds to relative offset 0.
    /// At offset 0, sin(0) = 0 and cos(0) = 1, so the row should be
    /// `[0, 1, 0, 1, ..., 0, 1]` interleaved.
    func testSinusoidalPosEmbZeroOffsetRow() throws {
        let T = 4
        let D = 8
        let pe = RelPosSelfAttention.makeSinusoidalPosEmb(seqLen: T, dModel: D)
        XCTAssertEqual(pe.shape, [1, 2 * T - 1, D])

        let arr = pe.toArray()
        let centerRowOffset = (T - 1) * D
        for k in 0..<(D / 2) {
            XCTAssertEqual(arr[centerRowOffset + 2 * k],     0.0,
                accuracy: 1e-6, "sin(0) at center row, dim \(2*k) should be 0")
            XCTAssertEqual(arr[centerRowOffset + 2 * k + 1], 1.0,
                accuracy: 1e-6, "cos(0) at center row, dim \(2*k+1) should be 1")
        }
    }

    /// Symmetry: row at offset +p has sin(+p) and cos(+p), while row
    /// at offset -p has sin(-p) and cos(-p). Since cos is even
    /// (cos(-p) = cos(p)) and sin is odd (sin(-p) = -sin(p)), the
    /// even-indexed dims should flip sign and the odd-indexed dims
    /// should match between the +p and -p rows.
    func testSinusoidalPosEmbEvenOddSymmetry() throws {
        let T = 5
        let D = 8
        let pe = RelPosSelfAttention.makeSinusoidalPosEmb(seqLen: T, dModel: D)
        let arr = pe.toArray()
        // Offset +p sits at row (T-1) - p; offset -p at row (T-1) + p.
        for p in 1..<T {
            let posRow = (T - 1) - p
            let negRow = (T - 1) + p
            for k in 0..<(D / 2) {
                let pSin = arr[posRow * D + 2 * k]
                let nSin = arr[negRow * D + 2 * k]
                let pCos = arr[posRow * D + 2 * k + 1]
                let nCos = arr[negRow * D + 2 * k + 1]
                XCTAssertEqual(nSin, -pSin, accuracy: 1e-5,
                    "sin should be odd: p=\(p), k=\(k)")
                XCTAssertEqual(nCos, pCos, accuracy: 1e-5,
                    "cos should be even: p=\(p), k=\(k)")
            }
        }
    }

    // MARK: - Equivalence to standard self-attention

    /// When `posBiasU = posBiasV = 0` AND `wPos = 0`, the position
    /// attention matrix is identically zero, so rel-pos attention
    /// reduces to standard non-causal self-attention. Verify by
    /// running both and comparing outputs element-wise.
    func testReducesToStandardWhenPosWeightsAreZero() throws {
        let D = 16, H = 4, T = 6
        let Dh = D / H

        // Random-ish but reproducible weights for Q/K/V/Out.
        var rng = SeededRNG(seed: 42)
        func randMatrix(_ rows: Int, _ cols: Int) throws -> Tensor {
            try Tensor.from(
                data: (0..<(rows * cols)).map { _ in
                    Float.random(in: -0.5...0.5, using: &rng)
                },
                shape: [rows, cols]
            )
        }
        func zeros(_ count: Int) throws -> Tensor {
            try Tensor.from(
                data: [Float](repeating: 0, count: count),
                shape: [count]
            )
        }
        let wQ = try randMatrix(D, D)
        let bQ = try zeros(D)
        let wK = try randMatrix(D, D)
        let bK = try zeros(D)
        let wV = try randMatrix(D, D)
        let bV = try zeros(D)
        let wOut = try randMatrix(D, D)
        let bOut = try zeros(D)
        // Position-related weights all zero — collapses BD to zero.
        let wPos = try Tensor.from(
            data: [Float](repeating: 0, count: D * D),
            shape: [D, D]
        )
        let posBiasU = try Tensor.from(
            data: [Float](repeating: 0, count: H * Dh),
            shape: [H, Dh]
        )
        let posBiasV = try Tensor.from(
            data: [Float](repeating: 0, count: H * Dh),
            shape: [H, Dh]
        )
        let attn = RelPosSelfAttention(
            dModel: D, numHeads: H,
            wQ: wQ, bQ: bQ, wK: wK, bK: bK, wV: wV, bV: bV,
            wPos: wPos, wOut: wOut, bOut: bOut,
            posBiasU: posBiasU, posBiasV: posBiasV
        )

        // Random input.
        let x = try Tensor.from(
            data: (0..<(T * D)).map { _ in
                Float.random(in: -1.0...1.0, using: &rng)
            },
            shape: [1, T, D]
        )
        let posEmb = RelPosSelfAttention.makeSinusoidalPosEmb(
            seqLen: T, dModel: D
        )

        // Rel-pos attention forward.
        let outRel = attn.forward(x, posEmb: posEmb).toArray()

        // Reference: project Q, K, V then run standard non-causal
        // attention with the same scale, then apply the output proj.
        let bQ3 = bQ.reshape([1, 1, D])
        let bK3 = bK.reshape([1, 1, D])
        let bV3 = bV.reshape([1, 1, D])
        let q = x.matmul(wQ).add(bQ3)
        let k = x.matmul(wK).add(bK3)
        let v = x.matmul(wV).add(bV3)
        // Split heads.
        func split(_ t: Tensor) -> Tensor {
            t.reshape([1, T, H, Dh])
             .permute([0, 2, 1, 3]).contiguous()
             .reshape([H, T, Dh])
        }
        let attended = split(q).attention(
            key: split(k), value: split(v),
            scale: 1.0 / sqrtf(Float(Dh)),
            causal: false
        )
        let merged = attended
            .reshape([1, H, T, Dh])
            .permute([0, 2, 1, 3]).contiguous()
            .reshape([1, T, D])
        let bOut3 = bOut.reshape([1, 1, D])
        let outStd = merged.matmul(wOut).add(bOut3).toArray()

        XCTAssertEqual(outRel.count, outStd.count)
        for i in 0..<outRel.count {
            XCTAssertEqual(outRel[i], outStd[i], accuracy: 1e-3,
                "outRel[\(i)] = \(outRel[i]) vs outStd[\(i)] = \(outStd[i])")
        }
    }

    // MARK: - Forward pass shape & finiteness

    /// Smoke test on a real-sized config (D=128, H=4, T=24): make sure
    /// the forward pass produces the expected shape and finite values
    /// with random weights.
    func testForwardShapeAndFinite() throws {
        let D = 128, H = 4, T = 24
        var rng = SeededRNG(seed: 7)
        func rnd(_ n: Int) throws -> Tensor {
            try Tensor.from(
                data: (0..<n).map { _ in Float.random(in: -0.05...0.05, using: &rng) },
                shape: [n]
            )
        }
        func rndMat(_ r: Int, _ c: Int) throws -> Tensor {
            try Tensor.from(
                data: (0..<(r * c)).map { _ in
                    Float.random(in: -0.05...0.05, using: &rng)
                },
                shape: [r, c]
            )
        }
        let attn = RelPosSelfAttention(
            dModel: D, numHeads: H,
            wQ: try rndMat(D, D), bQ: try rnd(D),
            wK: try rndMat(D, D), bK: try rnd(D),
            wV: try rndMat(D, D), bV: try rnd(D),
            wPos: try rndMat(D, D),
            wOut: try rndMat(D, D), bOut: try rnd(D),
            posBiasU: try Tensor.from(
                data: (0..<(H * (D / H))).map { _ in
                    Float.random(in: -0.05...0.05, using: &rng) },
                shape: [H, D / H]
            ),
            posBiasV: try Tensor.from(
                data: (0..<(H * (D / H))).map { _ in
                    Float.random(in: -0.05...0.05, using: &rng) },
                shape: [H, D / H]
            )
        )
        let x = try Tensor.from(
            data: (0..<(T * D)).map { _ in
                Float.random(in: -1.0...1.0, using: &rng)
            },
            shape: [1, T, D]
        )
        let posEmb = RelPosSelfAttention.makeSinusoidalPosEmb(
            seqLen: T, dModel: D
        )
        let out = attn.forward(x, posEmb: posEmb)
        XCTAssertEqual(out.shape, [1, T, D])
        for v in out.toArray() {
            XCTAssertTrue(v.isFinite, "non-finite output value")
        }
    }
}
