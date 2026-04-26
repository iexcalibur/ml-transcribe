import XCTest
@testable import Transcribe

/// Tests for the new `groups` parameter on `Tensor.conv1d`. Depthwise
/// conv (groups == C_in) is the inner loop of Conformer's
/// `ConvModule`; verify it lines up with the by-hand reference.
final class Conv1dGroupsTests: XCTestCase {

    /// `groups=1` (the previous default) must produce identical
    /// results to a hand-written sum-over-channels reference.
    func testGroups1MatchesDenseConv() throws {
        // input shape [1, 2, 4]: 2 channels, 4 time steps.
        let x = try Tensor.from(
            data: [1, 2, 3, 4,    // ch 0
                   5, 6, 7, 8],   // ch 1
            shape: [1, 2, 4]
        )
        // weight [C_out=1, C_in=2, K=2]:
        //   c0 sums ch0 and 2x ch1 over a 2-tap window.
        let w = try Tensor.from(
            data: [1, 1,    // c_out=0, c_in=0, K=2
                   2, 2],   // c_out=0, c_in=1, K=2
            shape: [1, 2, 2]
        )
        let out = x.conv1d(weight: w, stride: 1, padding: 0, groups: 1).toArray()
        // For each output position p (0..2):
        //   sum_{c,k} x[c, p+k] * w[c, k]
        //   p=0: (1+2) + 2*(5+6) = 3 + 22 = 25
        //   p=1: (2+3) + 2*(6+7) = 5 + 26 = 31
        //   p=2: (3+4) + 2*(7+8) = 7 + 30 = 37
        XCTAssertEqual(out, [25, 31, 37])
    }

    /// Depthwise: each input channel maps to one output channel via
    /// its own kernel. With `groups == C_in == C_out`, output channel
    /// i depends ONLY on input channel i.
    func testDepthwiseChannelsAreIndependent() throws {
        // 3 channels × 4 time steps.
        let x = try Tensor.from(
            data: [1,  2,  3,  4,
                   10, 20, 30, 40,
                   100, 200, 300, 400],
            shape: [1, 3, 4]
        )
        // weight [C_out=3, C_in/groups=1, K=2] — each channel has its
        // own 2-tap sum kernel scaled differently.
        let w = try Tensor.from(
            data: [1, 1,   // ch 0: identity sum
                   2, 2,   // ch 1: 2x sum
                   3, 3],  // ch 2: 3x sum
            shape: [3, 1, 2]
        )
        let out = x.conv1d(
            weight: w, stride: 1, padding: 0, groups: 3
        ).toArray()
        // For each channel separately:
        //   ch 0: (1+2)=3, (2+3)=5, (3+4)=7
        //   ch 1: 2*(10+20)=60, 2*(20+30)=100, 2*(30+40)=140
        //   ch 2: 3*(100+200)=900, 3*(200+300)=1500, 3*(300+400)=2100
        XCTAssertEqual(out, [3, 5, 7,
                             60, 100, 140,
                             900, 1500, 2100])
    }

    /// Cohere ConvModule layout: pointwise(d→2d) → glu → depthwise
    /// (groups=d, k=9). The shape contract is what we're after here.
    func testCohereConvModuleShapes() throws {
        let D = 32      // shrink for test speed; production = 1280
        let T = 16
        let kernel = 9
        // Pointwise expansion: d → 2d
        let xPre = try Tensor.from(
            data: (0..<(D * T)).map { Float($0) * 0.01 },
            shape: [1, D, T]
        )
        let wPointwiseExpand = try Tensor.from(
            data: Array(repeating: Float(0.05), count: 2 * D * D * 1),
            shape: [2 * D, D, 1]
        )
        let expanded = xPre.conv1d(weight: wPointwiseExpand)
        XCTAssertEqual(expanded.shape, [1, 2 * D, T])

        // GLU halves the channel count back to D.
        let gated = expanded.glu(dim: 1)
        XCTAssertEqual(gated.shape, [1, D, T])

        // Depthwise conv with kernel=9, padding=(k-1)/2 → keeps T.
        let wDepth = try Tensor.from(
            data: Array(repeating: Float(0.1), count: D * 1 * kernel),
            shape: [D, 1, kernel]
        )
        let depthOut = gated.conv1d(
            weight: wDepth, stride: 1, padding: (kernel - 1) / 2,
            groups: D
        )
        XCTAssertEqual(depthOut.shape, [1, D, T])
        for v in depthOut.toArray() {
            XCTAssertTrue(v.isFinite)
        }
    }
}
