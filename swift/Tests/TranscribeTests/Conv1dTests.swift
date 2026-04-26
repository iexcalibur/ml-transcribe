import XCTest
import Foundation
@testable import Transcribe

final class Conv1dTests: XCTestCase {

    // MARK: - Shape arithmetic

    /// k=3, stride=1, padding=1 preserves the length axis (Whisper's
    /// first conv layer is exactly this shape).
    func testWhisperConv1ShapePreservesLength() throws {
        let n = 1, cIn = 4, l = 16, cOut = 8, k = 3
        let input = try Tensor.from(
            data: Array(repeating: Float(0), count: n * cIn * l),
            shape: [n, cIn, l]
        )
        let weight = try Tensor.from(
            data: Array(repeating: Float(0), count: cOut * cIn * k),
            shape: [cOut, cIn, k]
        )
        let out = input.conv1d(weight: weight, stride: 1, padding: 1)
        XCTAssertEqual(out.shape, [n, cOut, l])
    }

    /// k=3, stride=2, padding=1 halves the length (Whisper's second
    /// conv layer).
    func testWhisperConv2ShapeHalvesLength() throws {
        let n = 1, cIn = 8, l = 200, cOut = 8, k = 3
        let input = try Tensor.from(
            data: Array(repeating: Float(0), count: n * cIn * l),
            shape: [n, cIn, l]
        )
        let weight = try Tensor.from(
            data: Array(repeating: Float(0), count: cOut * cIn * k),
            shape: [cOut, cIn, k]
        )
        let out = input.conv1d(weight: weight, stride: 2, padding: 1)
        // L_out = (200 + 2 - 3) / 2 + 1 = 100.
        XCTAssertEqual(out.shape, [n, cOut, 100])
    }

    /// stride=1, padding=0, k=K shrinks length by K-1 (no padding).
    func testNoPaddingShrinksLength() throws {
        let n = 1, c = 1, l = 10, k = 3
        let input = try Tensor.from(
            data: Array(repeating: Float(0), count: l),
            shape: [n, c, l]
        )
        let weight = try Tensor.from(
            data: Array(repeating: Float(0), count: k),
            shape: [c, c, k]
        )
        let out = input.conv1d(weight: weight, stride: 1, padding: 0)
        // L_out = (10 + 0 - 3) / 1 + 1 = 8.
        XCTAssertEqual(out.shape, [n, c, 8])
    }

    // MARK: - Numerical correctness

    /// Identity-1 kernel: a single channel with kernel = [1] should
    /// produce output equal to input (stride=1, padding=0).
    func testKernel1IdentityCopiesInput() throws {
        let inputData: [Float] = [1, 2, 3, 4, 5]
        let input = try Tensor.from(data: inputData, shape: [1, 1, 5])
        // weight [1, 1, 1] = 1.0
        let weight = try Tensor.from(data: [1.0], shape: [1, 1, 1])
        let out = input.conv1d(weight: weight, stride: 1, padding: 0)
        XCTAssertEqual(out.shape, [1, 1, 5])
        XCTAssertEqual(out.toArray(), inputData)
    }

    /// k=3 sum kernel [1,1,1] with padding=1 produces a 3-element
    /// moving sum (with zero padding at edges):
    ///   in  = [1,2,3,4,5]
    ///   pad = [0,1,2,3,4,5,0]
    ///   out = [0+1+2, 1+2+3, 2+3+4, 3+4+5, 4+5+0]
    ///       = [3, 6, 9, 12, 9]
    func testKernel3SumProducesExpectedMovingSum() throws {
        let input = try Tensor.from(
            data: [1, 2, 3, 4, 5], shape: [1, 1, 5]
        )
        let weight = try Tensor.from(data: [1, 1, 1], shape: [1, 1, 3])
        let out = input.conv1d(weight: weight, stride: 1, padding: 1)
        XCTAssertEqual(out.toArray(), [3, 6, 9, 12, 9])
    }

    /// Two-channel input, two-channel weight: each output channel is
    /// the sum across input channels.
    ///   in[0] = [1,2,3], in[1] = [4,5,6]
    ///   w[0,0] = [1], w[0,1] = [1]   -> out[0] = in[0] + in[1] = [5,7,9]
    ///   w[1,0] = [2], w[1,1] = [0]   -> out[1] = 2 * in[0] = [2,4,6]
    func testTwoChannelKernel1MultiplexesAndSums() throws {
        let input = try Tensor.from(
            data: [1, 2, 3, 4, 5, 6],
            shape: [1, 2, 3]
        )
        let weight = try Tensor.from(
            // Layout: [c_out, c_in, k]
            //  out=0: in=0 -> 1, in=1 -> 1
            //  out=1: in=0 -> 2, in=1 -> 0
            data: [1, 1,  2, 0],
            shape: [2, 2, 1]
        )
        let out = input.conv1d(weight: weight, stride: 1, padding: 0)
        XCTAssertEqual(out.shape, [1, 2, 3])
        XCTAssertEqual(out.toArray(), [5, 7, 9,  2, 4, 6])
    }

    // MARK: - Composition with downstream ops

    /// Whisper's audio frontend: log-mel spectrogram (input) flows
    /// through Conv1d → GELU → Conv1d (stride 2). Verify shapes
    /// flow correctly end-to-end.
    func testWhisperFrontEndShapes() throws {
        // 30 seconds of audio at 16 kHz produces a 80×3000 mel.
        // Input to conv1: [B=1, C=80, L=3000].
        let nMels = 80, l = 3000
        let mel = try Tensor.from(
            data: Array(repeating: Float(0), count: nMels * l),
            shape: [1, nMels, l]
        )
        let w1 = try Tensor.from(
            data: Array(repeating: Float(0), count: nMels * nMels * 3),
            shape: [nMels, nMels, 3]
        )
        let w2 = try Tensor.from(
            data: Array(repeating: Float(0), count: nMels * nMels * 3),
            shape: [nMels, nMels, 3]
        )

        // Whisper layout:
        //   conv1: stride=1, padding=1, k=3 → preserves L
        //   GELU
        //   conv2: stride=2, padding=1, k=3 → halves L
        let after1 = mel.conv1d(weight: w1, stride: 1, padding: 1).gelu()
        let after2 = after1.conv1d(weight: w2, stride: 2, padding: 1)

        XCTAssertEqual(after1.shape, [1, 80, 3000])
        XCTAssertEqual(after2.shape, [1, 80, 1500])
    }
}
