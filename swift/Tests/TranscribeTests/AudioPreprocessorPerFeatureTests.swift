import XCTest
@testable import Transcribe

/// Tests for the per-feature normalization mode of `AudioPreprocessor`.
///
/// Whisper-mode tests live in `AudioPreprocessorTests.swift`; here we
/// verify the NeMo / Cohere-Transcribe variant that subtracts the
/// per-mel-bin mean and divides by the per-bin standard deviation.
final class AudioPreprocessorPerFeatureTests: XCTestCase {

    /// Per-feature mode: every mel bin should have ~zero mean and
    /// ~unit variance across frames. The inverse-std epsilon prevents
    /// blow-up in silent bins, so we tolerate a small slop.
    func testPerFeatureProducesZeroMeanUnitVar() throws {
        // 1 sec of mixed sine waves so most mel bins have non-trivial
        // energy. Pure silence would leave many bins exactly equal to
        // log10(1e-10) = -10, where var≈0 and the eps dominates.
        let sr = 16_000
        let nSamples = sr
        let samples: [Float] = (0..<nSamples).map { i in
            let t = Float(i) / Float(sr)
            return sinf(2 * .pi * 440 * t)
                 + 0.5 * sinf(2 * .pi * 880 * t)
                 + 0.25 * sinf(2 * .pi * 1760 * t)
        }

        let cfg = AudioPreprocessor.Config(
            sampleRate: 16_000,
            nFFT: 400, hopLength: 160, nMels: 80,
            normalize: .perFeature
        )
        let mel = AudioPreprocessor.logMelSpectrogram(samples: samples, config: cfg)
        let nMels = mel.shape[0]
        let nFrames = mel.shape[1]
        let arr = mel.toArray()

        for m in 0..<nMels {
            var mean: Float = 0
            for f in 0..<nFrames {
                mean += arr[m * nFrames + f]
            }
            mean /= Float(nFrames)
            var sq: Float = 0
            for f in 0..<nFrames {
                let d = arr[m * nFrames + f] - mean
                sq += d * d
            }
            let varVal = sq / Float(nFrames)
            XCTAssertEqual(mean, 0, accuracy: 1e-3,
                "mel bin \(m): mean \(mean) not ~0")
            XCTAssertEqual(varVal, 1, accuracy: 0.05,
                "mel bin \(m): var \(varVal) not ~1")
        }
    }

    /// The Cohere-Transcribe preset matches the `n_fft=512`,
    /// `n_mels=128` shape contract documented in the model's
    /// preprocessor_config.json. Verifying the shape lines up
    /// catches any off-by-one in the FFI argument order.
    func testCohereTranscribePresetShape() throws {
        let nSamples = 16_000
        let samples = [Float](repeating: 0.1, count: nSamples)
        let mel = AudioPreprocessor.logMelSpectrogram(
            samples: samples, config: .cohereTranscribe
        )
        XCTAssertEqual(mel.shape[0], 128, "Cohere preset must use 128 mel bins")
        // n_frames depends on padding convention; just confirm it's
        // in the expected ballpark for 1 sec @ 16 kHz, hop=160.
        XCTAssertGreaterThan(mel.shape[1], 90)
        XCTAssertLessThan(mel.shape[1], 110)
    }
}
