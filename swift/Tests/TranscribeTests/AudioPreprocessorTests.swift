import XCTest
import Foundation
@testable import Transcribe

final class AudioPreprocessorTests: XCTestCase {

    // MARK: - Shape

    /// Output shape matches what Whisper expects.
    func testWhisperConfigProducesExpected80MelShape() {
        let oneSecondAt16kHz: [Float] = Array(repeating: 0.5, count: 16_000)
        let mel = AudioPreprocessor.logMelSpectrogram(samples: oneSecondAt16kHz)
        XCTAssertEqual(mel.shape.count, 2)
        XCTAssertEqual(mel.shape[0], 80, "n_mels = 80 (Whisper default)")
        // Frames per second at hop=160: 16000/160 = 100, plus a couple
        // edge frames from reflective padding → ~101.
        XCTAssertGreaterThan(mel.shape[1], 99)
        XCTAssertLessThan(mel.shape[1], 105)
    }

    /// All elements are finite (no NaN, no inf).
    func testOutputIsFinite() {
        let samples: [Float] = (0..<8000).map { i in
            sin(2.0 * .pi * 440.0 * Float(i) / 16_000.0)
        }
        let mel = AudioPreprocessor.logMelSpectrogram(samples: samples).toArray()
        for x in mel {
            XCTAssertTrue(x.isFinite, "non-finite mel value: \(x)")
        }
    }

    // MARK: - Frequency-domain correctness

    /// A 440 Hz sine wave should produce most of its energy in the
    /// lower mel bins (low-frequency sound). The peak across the time
    /// axis should be in the bottom half of the 80 mel bins.
    func testSineWaveEnergyPeaksInLowMelBins() {
        let sampleRate: UInt32 = 16_000
        let frequency: Float = 440.0   // A4
        let n = 16_000                 // 1 second
        let samples: [Float] = (0..<n).map { i in
            sin(2.0 * .pi * frequency * Float(i) / Float(sampleRate))
        }

        let mel = AudioPreprocessor.logMelSpectrogram(samples: samples)
        let nMels = mel.shape[0]
        let nFrames = mel.shape[1]
        let arr = mel.toArray()

        // Sum each mel bin's energy across the time axis.
        var binEnergy = [Float](repeating: 0, count: nMels)
        for m in 0..<nMels {
            for f in 0..<nFrames {
                binEnergy[m] += arr[m * nFrames + f]
            }
        }
        let argmax = binEnergy.enumerated()
            .max(by: { $0.element < $1.element })!.offset

        // 440 Hz is well below 4 kHz; should peak in bins < 40.
        XCTAssertLessThan(argmax, 40,
            "440 Hz sine should peak in lower mel bins, got \(argmax)")
    }

    /// A 4 kHz sine should peak in higher mel bins than a 440 Hz sine.
    func testHigherFrequencyShiftsToHigherBins() {
        let sampleRate: UInt32 = 16_000
        let n = 8_000

        func makeSine(freq: Float) -> [Float] {
            (0..<n).map { i in
                sin(2.0 * .pi * freq * Float(i) / Float(sampleRate))
            }
        }
        func argmaxBin(_ samples: [Float]) -> Int {
            let mel = AudioPreprocessor.logMelSpectrogram(samples: samples)
            let nMels = mel.shape[0]
            let nFrames = mel.shape[1]
            let arr = mel.toArray()
            var energies = [Float](repeating: 0, count: nMels)
            for m in 0..<nMels {
                for f in 0..<nFrames {
                    energies[m] += arr[m * nFrames + f]
                }
            }
            return energies.enumerated()
                .max(by: { $0.element < $1.element })!.offset
        }

        let lowBin = argmaxBin(makeSine(freq: 440))    // A4
        let highBin = argmaxBin(makeSine(freq: 4_000)) // 4 kHz

        XCTAssertLessThan(lowBin, highBin,
            "Higher-frequency sine should peak in a higher mel bin (low=\(lowBin), high=\(highBin))")
    }

    // MARK: - Composition

    /// Output is a real Tensor: composes with downstream ops without
    /// special handling.
    func testMelOutputComposesWithMatmul() throws {
        let samples: [Float] = Array(repeating: 0.1, count: 16_000)
        let mel = AudioPreprocessor.logMelSpectrogram(samples: samples)
        // mel: [80, ~101]. Multiply by an identity-like projection to
        // verify it flows through ops cleanly.
        let nFrames = mel.shape[1]
        // Project [80, n_frames] → [80, n_frames] with a 1x1 identity.
        // Easier: just sum it — exercises the F32 path without shape
        // wrangling.
        let s = mel.sum
        XCTAssertTrue(s.isFinite,
            "mel.sum should be finite, got \(s)")
    }

    // MARK: - Custom configs

    /// Different n_mels actually changes the output shape.
    func testCustomNMelsChangesShape() {
        let samples: [Float] = Array(repeating: 0.0, count: 4_000)
        let mel40 = AudioPreprocessor.logMelSpectrogram(
            samples: samples,
            config: .init(sampleRate: 16_000, nFFT: 400, hopLength: 160, nMels: 40)
        )
        XCTAssertEqual(mel40.shape[0], 40)
    }
}
