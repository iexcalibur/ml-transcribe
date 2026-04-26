import XCTest
@testable import Transcribe

/// Tests for `Tensor.batchNorm1d` — inference-mode batch norm.
/// Formula: y = (x - running_mean) / sqrt(running_var + eps) * gamma + beta
/// Used inside Conformer's ConvModule.
final class BatchNorm1dTests: XCTestCase {

    /// With `gamma=1, beta=0, mean=0, var=1, eps=0`, batchnorm is the
    /// identity function.
    func testIdentityWhenNeutralStats() throws {
        let x = try Tensor.from(
            data: [1, 2, 3, 4, 5, 6],
            shape: [1, 2, 3]            // [N=1, C=2, L=3]
        )
        let mean = try Tensor.from(data: [0, 0], shape: [2])
        let varT = try Tensor.from(data: [1, 1], shape: [2])
        let gamma = try Tensor.from(data: [1, 1], shape: [2])
        let beta = try Tensor.from(data: [0, 0], shape: [2])
        let out = x.batchNorm1d(
            runningMean: mean, runningVar: varT,
            gamma: gamma, beta: beta, eps: 0.0
        ).toArray()
        XCTAssertEqual(out, [1, 2, 3, 4, 5, 6])
    }

    /// Subtracting `running_mean` and dividing by `sqrt(running_var)`
    /// produces zero-mean / unit-variance per channel (within tol).
    func testZeroMeanUnitVariancePerChannel() throws {
        // ch 0: [10, 20, 30] -> mean=20, var=66.67
        // ch 1: [1, 2, 3]    -> mean=2, var=0.667
        let x = try Tensor.from(
            data: [10, 20, 30,
                   1, 2, 3],
            shape: [1, 2, 3]
        )
        let mean = try Tensor.from(data: [20, 2], shape: [2])
        let v0: Float = ((10 - 20) * (10 - 20) + 0 + (30 - 20) * (30 - 20)) / 3
        let v1: Float = ((1 - 2) * (1 - 2) + 0 + (3 - 2) * (3 - 2)) / 3
        let varT = try Tensor.from(data: [v0, v1], shape: [2])
        let gamma = try Tensor.from(data: [1, 1], shape: [2])
        let beta = try Tensor.from(data: [0, 0], shape: [2])
        let out = x.batchNorm1d(
            runningMean: mean, runningVar: varT,
            gamma: gamma, beta: beta, eps: 1e-12
        ).toArray()
        // Per-channel mean should be ~0, std should be ~1.
        let ch0 = Array(out[0..<3])
        let ch1 = Array(out[3..<6])
        XCTAssertEqual(ch0.reduce(0, +), 0, accuracy: 1e-4)
        XCTAssertEqual(ch1.reduce(0, +), 0, accuracy: 1e-4)
        let s0: Float = ch0.map { $0 * $0 }.reduce(0, +) / 3
        let s1: Float = ch1.map { $0 * $0 }.reduce(0, +) / 3
        XCTAssertEqual(s0, 1, accuracy: 1e-3)
        XCTAssertEqual(s1, 1, accuracy: 1e-3)
    }

    /// `gamma` and `beta` apply a per-channel affine transform after
    /// normalization. With `gamma=2, beta=10`, we get 2x scale + 10
    /// offset on every normalized value.
    func testGammaBetaAffineApplied() throws {
        let x = try Tensor.from(data: [0, 0, 0, 0], shape: [1, 1, 4])
        let mean = try Tensor.from(data: [0], shape: [1])
        let varT = try Tensor.from(data: [1], shape: [1])
        let gamma = try Tensor.from(data: [2], shape: [1])
        let beta = try Tensor.from(data: [10], shape: [1])
        let out = x.batchNorm1d(
            runningMean: mean, runningVar: varT,
            gamma: gamma, beta: beta, eps: 0
        ).toArray()
        // (0 - 0) / 1 * 2 + 10 = 10 for all.
        XCTAssertEqual(out, [10, 10, 10, 10])
    }

    /// Multi-channel shapes are preserved end to end.
    func testShapePreserved() throws {
        let x = try Tensor.from(
            data: (0..<(2 * 4 * 8)).map { Float($0) * 0.01 },
            shape: [2, 4, 8]
        )
        let mean = try Tensor.from(data: [0, 0, 0, 0], shape: [4])
        let varT = try Tensor.from(data: [1, 1, 1, 1], shape: [4])
        let gamma = try Tensor.from(data: [1, 1, 1, 1], shape: [4])
        let beta = try Tensor.from(data: [0, 0, 0, 0], shape: [4])
        let out = x.batchNorm1d(
            runningMean: mean, runningVar: varT,
            gamma: gamma, beta: beta
        )
        XCTAssertEqual(out.shape, [2, 4, 8])
    }
}
