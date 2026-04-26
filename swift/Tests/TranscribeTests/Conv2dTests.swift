import XCTest
@testable import Transcribe

/// Tests for `Tensor.conv2d` — the 2-D convolution used by Cohere's
/// `ConvSubsampling` audio frontend. PyTorch convention:
///   input  [N, C_in, H, W]
///   weight [C_out, C_in/groups, kH, kW]
///   output [N, C_out, H_out, W_out]
final class Conv2dTests: XCTestCase {

    // MARK: - Mathematical correctness

    /// 1×1 identity conv with one channel: each output value equals
    /// the matching input value. The simplest sanity check.
    func testIdentity1x1Kernel() throws {
        // input = [[[[1, 2], [3, 4]]]] shape [1,1,2,2]
        // weight = [[[[1]]]] shape [1,1,1,1]
        let x = try Tensor.from(data: [1, 2, 3, 4], shape: [1, 1, 2, 2])
        let w = try Tensor.from(data: [1], shape: [1, 1, 1, 1])
        let out = x.conv2d(weight: w)
        XCTAssertEqual(out.shape, [1, 1, 2, 2])
        XCTAssertEqual(out.toArray(), [1, 2, 3, 4])
    }

    /// 3×3 sum kernel of ones: each output is the sum of its receptive
    /// field. With stride=1, padding=1 the output keeps spatial dims.
    func testSumKernelWithPadding() throws {
        // 3×3 input of all 1s. With a 3×3 kernel of ones and pad=1:
        //   - center output = 9 (full 3×3)
        //   - edge outputs  = 6 (2×3 inside, 1×3 outside zero-padded)
        //   - corner outputs = 4 (2×2 inside)
        let x = try Tensor.from(
            data: Array(repeating: Float(1), count: 9),
            shape: [1, 1, 3, 3]
        )
        let w = try Tensor.from(
            data: Array(repeating: Float(1), count: 9),
            shape: [1, 1, 3, 3]
        )
        let out = x.conv2d(weight: w, stride: 1, padding: 1).toArray()
        // 3×3 result — corners=4, edges=6, center=9.
        XCTAssertEqual(out, [4, 6, 4,
                             6, 9, 6,
                             4, 6, 4])
    }

    /// Stride-2 conv halves both spatial dims (modulo padding).
    /// 4×4 input → 2×2 output with kernel=3, stride=2, padding=1.
    func testStrideHalvesSpatialDims() throws {
        let x = try Tensor.from(
            data: Array(repeating: Float(1), count: 16),
            shape: [1, 1, 4, 4]
        )
        let w = try Tensor.from(
            data: Array(repeating: Float(1), count: 9),
            shape: [1, 1, 3, 3]
        )
        let out = x.conv2d(weight: w, stride: 2, padding: 1)
        XCTAssertEqual(out.shape, [1, 1, 2, 2])
    }

    // MARK: - Groups (depthwise)

    /// Depthwise 2-D conv: `groups == C_in == C_out`. Each input
    /// channel maps to its own output channel through its own kernel.
    func testDepthwise2x2Channels() throws {
        // 2 input channels, each gets a different identity-like kernel.
        // Channel 0 kernel: [[1]] (identity)
        // Channel 1 kernel: [[2]] (scale by 2)
        let x = try Tensor.from(
            data: [10, 20,        // channel 0
                   100, 200],     // channel 1
            shape: [1, 2, 1, 2]
        )
        let w = try Tensor.from(
            data: [1, 2],         // [C_out=2, C_in/groups=1, kH=1, kW=1]
            shape: [2, 1, 1, 1]
        )
        let out = x.conv2d(weight: w, groups: 2).toArray()
        // Out[0] = 10*1, 20*1; Out[1] = 100*2, 200*2.
        XCTAssertEqual(out, [10, 20, 200, 400])
    }

    // MARK: - Cohere ConvSubsampling shape stack

    /// The Cohere encoder stacks 5 conv2d layers:
    ///   Conv2d(1,    256, k=3, s=2, p=1)            # dense
    ///   Conv2d(256,  256, k=3, s=2, p=1, g=256)     # depthwise
    ///   Conv2d(256,  256, k=1)                       # pointwise
    ///   Conv2d(256,  256, k=3, s=2, p=1, g=256)     # depthwise
    ///   Conv2d(256,  256, k=1)                       # pointwise
    /// reducing time by 8× and frequency by 8×. Verify the chain
    /// produces the expected shapes on a small synthetic input
    /// (the values would obviously differ from real weights, but
    /// the shape pipeline is what we're checking).
    func testConvSubsamplingShapeStack() throws {
        let nMels = 128
        let nFrames = 64    // small power-of-2 for clean math
        let x = try Tensor.from(
            data: (0..<(nMels * nFrames)).map { Float($0) * 0.001 },
            shape: [1, 1, nFrames, nMels]
        )
        // layer 1: dense, [1,1,3,3] → [1,256,32,64]
        let w1 = try Tensor.from(
            data: Array(repeating: Float(0.01), count: 256 * 1 * 3 * 3),
            shape: [256, 1, 3, 3]
        )
        let h1 = x.conv2d(weight: w1, stride: 2, padding: 1)
        XCTAssertEqual(h1.shape, [1, 256, 32, 64])

        // layer 2: depthwise, kernel groups=256
        let w2 = try Tensor.from(
            data: Array(repeating: Float(0.01), count: 256 * 1 * 3 * 3),
            shape: [256, 1, 3, 3]
        )
        let h2 = h1.conv2d(weight: w2, stride: 2, padding: 1, groups: 256)
        XCTAssertEqual(h2.shape, [1, 256, 16, 32])

        // layer 3: pointwise (k=1, stride=1, no padding)
        let w3 = try Tensor.from(
            data: Array(repeating: Float(0.01), count: 256 * 256),
            shape: [256, 256, 1, 1]
        )
        let h3 = h2.conv2d(weight: w3)
        XCTAssertEqual(h3.shape, [1, 256, 16, 32])

        // layer 4: depthwise again
        let h4 = h3.conv2d(weight: w2, stride: 2, padding: 1, groups: 256)
        XCTAssertEqual(h4.shape, [1, 256, 8, 16])

        // layer 5: pointwise — final shape should be [1, 256, 8, 16]
        // (8× time, 8× frequency subsampled).
        let h5 = h4.conv2d(weight: w3)
        XCTAssertEqual(h5.shape, [1, 256, 8, 16])
        for v in h5.toArray() {
            XCTAssertTrue(v.isFinite)
        }
    }
}
