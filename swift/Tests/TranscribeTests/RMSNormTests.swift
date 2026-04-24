import XCTest
import Foundation
@testable import Transcribe

final class RMSNormTests: XCTestCase {

    // MARK: - Mathematical properties

    /// With gamma = 1, RMSNorm should produce a row whose mean-square
    /// is ~1.0 (by definition: y = x / sqrt(mean(x²) + eps), ignoring eps).
    func testRMSNormProducesUnitMeanSquare() throws {
        let x = try Tensor.from(data: [1, 2, 3, 4], shape: [4])
        let gamma = try Tensor.from(data: [1, 1, 1, 1], shape: [4])
        let y = x.rmsNorm(gamma: gamma).toArray()

        let meanSq = y.reduce(0) { $0 + $1 * $1 } / Float(y.count)
        XCTAssertEqual(meanSq, 1.0, accuracy: 1e-4)
    }

    /// Hand-computed expected values for x = [1, 2, 3, 4]:
    ///   mean_sq = (1+4+9+16)/4 = 7.5
    ///   rstd    = 1/sqrt(7.5) ≈ 0.36514837
    ///   y       = [1, 2, 3, 4] * 0.36515 = [0.3651, 0.7303, 1.0954, 1.4606]
    func testRMSNormKnownValues() throws {
        let x = try Tensor.from(data: [1, 2, 3, 4], shape: [4])
        let gamma = try Tensor.from(data: [1, 1, 1, 1], shape: [4])
        let y = x.rmsNorm(gamma: gamma).toArray()
        let expected: [Float] = [0.36514837, 0.73029673, 1.09544512, 1.46059348]
        for (a, b) in zip(y, expected) {
            XCTAssertEqual(a, b, accuracy: 1e-4)
        }
    }

    /// RMSNorm is scale-invariant (up to gamma): multiplying x by a
    /// positive constant k produces the SAME output (because the
    /// mean-square scales by k², and dividing by sqrt(k²·m²) = k·m
    /// cancels the k in the numerator).
    func testRMSNormIsScaleInvariantInInput() throws {
        let base: [Float] = [0.5, -1.5, 2.0, 3.25]
        let scaled = base.map { $0 * 7.0 }
        let gamma = try Tensor.from(data: [1, 1, 1, 1], shape: [4])

        let y1 = try Tensor.from(data: base,   shape: [4]).rmsNorm(gamma: gamma).toArray()
        let y2 = try Tensor.from(data: scaled, shape: [4]).rmsNorm(gamma: gamma).toArray()

        for (a, b) in zip(y1, y2) {
            XCTAssertEqual(a, b, accuracy: 1e-4)
        }
    }

    /// Gamma scales each feature independently: setting gamma[j] = k_j
    /// scales output[j] by k_j compared to the gamma=1 case.
    func testRMSNormGammaScales() throws {
        let xArr: [Float] = [1, 2, 3, 4]
        let ones = try Tensor.from(data: [1, 1, 1, 1], shape: [4])
        let multi = try Tensor.from(data: [2, 3, 4, 5], shape: [4])

        let yUnit = try Tensor.from(data: xArr, shape: [4]).rmsNorm(gamma: ones).toArray()
        let yScaled = try Tensor.from(data: xArr, shape: [4]).rmsNorm(gamma: multi).toArray()

        XCTAssertEqual(yScaled[0], yUnit[0] * 2, accuracy: 1e-4)
        XCTAssertEqual(yScaled[1], yUnit[1] * 3, accuracy: 1e-4)
        XCTAssertEqual(yScaled[2], yUnit[2] * 4, accuracy: 1e-4)
        XCTAssertEqual(yScaled[3], yUnit[3] * 5, accuracy: 1e-4)
    }

    // MARK: - Shape / dim handling

    /// 2-D input: each row along the last dim is normalized independently.
    func testRMSNormAppliesPerRowIn2D() throws {
        // Row 0: [1, 2, 3, 4]                   → mean_sq=7.5
        // Row 1: [10, 20, 30, 40] (10× row 0)   → mean_sq=750
        //
        // RMSNorm is scale-invariant in the input, so the two rows
        // must produce the same normalized output.
        let x = try Tensor.from(
            data: [1,  2,  3,  4,
                   10, 20, 30, 40],
            shape: [2, 4]
        )
        let gamma = try Tensor.from(data: [1, 1, 1, 1], shape: [4])
        let y = x.rmsNorm(gamma: gamma).toArray()

        for j in 0..<4 {
            XCTAssertEqual(y[j], y[4 + j], accuracy: 1e-4,
                "rows should match after per-row normalization")
        }
    }

    func testRMSNormPreservesShape() throws {
        for shape in [[4], [2, 4], [1, 3, 4], [2, 1, 4], [1, 1, 4, 4]] {
            let count = shape.reduce(1, *)
            var rng = SeededRNG(seed: 3)
            let data = (0..<count).map { _ in Float.random(in: -2...2, using: &rng) }
            let x = try Tensor.from(data: data, shape: shape)
            let D = shape.last!
            let gamma = try Tensor.from(
                data: Array(repeating: Float(1), count: D), shape: [D]
            )
            let y = x.rmsNorm(gamma: gamma)
            XCTAssertEqual(y.shape, shape)
        }
    }

    /// Zero input → zero output (gamma * 0 / anything = 0).
    /// This ALSO tests that `eps` prevents division-by-zero blowing up.
    func testRMSNormOfZerosIsZero() throws {
        let x = try Tensor.from(data: [0, 0, 0, 0], shape: [4])
        let gamma = try Tensor.from(data: [1, 1, 1, 1], shape: [4])
        let y = x.rmsNorm(gamma: gamma, eps: 1e-6).toArray()
        for e in y {
            XCTAssertEqual(e, 0.0, accuracy: 1e-5)
            XCTAssertTrue(e.isFinite, "eps should prevent NaN/inf on zero input")
        }
    }

    // MARK: - Composition

    /// RMSNorm composes with the rest of the LM stack — no ownership
    /// or shape bugs when chained with embedding, attention, or FFN.
    func testRMSNormComposesWithTransformerOps() throws {
        let D = 8
        let x = try Tensor.from(
            data: (0..<16).map { Float($0) * 0.1 },
            shape: [1, 2, D]
        )
        let gamma = try Tensor.from(
            data: Array(repeating: Float(1), count: D), shape: [D]
        )
        let w = try Tensor.from(
            data: Array(repeating: Float(0.1), count: D * D), shape: [D, D]
        )

        // RMSNorm → matmul → GELU → matmul. Standard LLaMA-style
        // "pre-norm" sub-layer kernel.
        let out = x.rmsNorm(gamma: gamma).matmul(w).gelu().matmul(w)
        XCTAssertEqual(out.shape, [1, 2, D])
        for f in out.toArray() {
            XCTAssertTrue(f.isFinite, "non-finite: \(f)")
        }
    }
}
