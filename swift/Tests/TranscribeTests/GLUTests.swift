import XCTest
@testable import Transcribe

/// Tests for `Tensor.glu` — Gated Linear Unit. Splits input in half
/// along `dim` and returns `firstHalf * sigmoid(secondHalf)`. Output
/// has the `dim` size halved.
final class GLUTests: XCTestCase {

    /// 1-D vector along dim 0: `[a, b, c, d]` → `[a*σ(c), b*σ(d)]`.
    func testGLUOnLastDim1D() throws {
        let x = try Tensor.from(data: [1, 2, 0, 0], shape: [4])
        // First half: [1, 2]; second half: [0, 0]; sigmoid(0) = 0.5.
        // Output: [1*0.5, 2*0.5] = [0.5, 1.0].
        let out = x.glu(dim: 0).toArray()
        XCTAssertEqual(out.count, 2)
        XCTAssertEqual(out[0], 0.5, accuracy: 1e-6)
        XCTAssertEqual(out[1], 1.0, accuracy: 1e-6)
    }

    /// Negative dim resolves to the last axis.
    func testNegativeDimWrapsToLast() throws {
        let x = try Tensor.from(data: [3, 4, 0, 0], shape: [4])
        let out = x.glu(dim: -1).toArray()
        XCTAssertEqual(out.count, 2)
        XCTAssertEqual(out[0], 1.5, accuracy: 1e-6)  // 3 * 0.5
        XCTAssertEqual(out[1], 2.0, accuracy: 1e-6)  // 4 * 0.5
    }

    /// 3-D split along channel dim. Input `[1, 4, 2]`, split dim=1 →
    /// output `[1, 2, 2]`. Used inside Conformer's ConvModule.
    func testGLUOnChannelDim3D() throws {
        // First half (channels 0,1): [1,2; 3,4]
        // Second half (channels 2,3): [10,10; 100,100] (large)
        // sigmoid(10) ≈ 1, sigmoid(100) ≈ 1
        // Output ≈ [1,2; 3,4]
        let x = try Tensor.from(
            data: [1, 2,        // ch 0
                   3, 4,        // ch 1
                   10, 10,      // ch 2 (gate, large)
                   100, 100],   // ch 3 (gate, larger)
            shape: [1, 4, 2]
        )
        let out = x.glu(dim: 1)
        XCTAssertEqual(out.shape, [1, 2, 2])
        let arr = out.toArray()
        XCTAssertEqual(arr[0], 1.0, accuracy: 1e-3)
        XCTAssertEqual(arr[1], 2.0, accuracy: 1e-3)
        XCTAssertEqual(arr[2], 3.0, accuracy: 1e-3)
        XCTAssertEqual(arr[3], 4.0, accuracy: 1e-3)
    }

    /// Large negative gate kills the value (sigmoid → 0).
    func testNegativeGateKillsOutput() throws {
        let x = try Tensor.from(data: [99, 99, -100, -100], shape: [4])
        let out = x.glu(dim: 0).toArray()
        XCTAssertEqual(out[0], 0, accuracy: 1e-3)
        XCTAssertEqual(out[1], 0, accuracy: 1e-3)
    }
}
