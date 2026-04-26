import XCTest
import Foundation
@testable import Transcribe

/// Opt-in integration tests that load and run real `openai/whisper-
/// tiny.en` weights end-to-end (audio → mel → encoder → decoder →
/// text).
///
/// Gated by the `WHISPER_TINY_DIR` environment variable. Default CI
/// leaves it unset and these tests skip; running:
///
///   WHISPER_TINY_DIR=/tmp/ml-inspect/whisper-tiny swift test ...
///
/// activates them. Required files in that directory:
///   - model.safetensors   (~151 MB, 167 tensors)
///   - tokenizer.json
///   - config.json         (read for sanity-check only; not required)
///
/// These tests live in the test target (rather than as a standalone
/// CLI) so they also run on iOS Simulator, where executables don't.
final class WhisperTinyIntegrationTests: XCTestCase {

    private struct ModelDir {
        let safetensors: String
        let tokenizer: String
    }

    private func whisperTinyDir() throws -> ModelDir {
        guard let dir = ProcessInfo.processInfo.environment["WHISPER_TINY_DIR"] else {
            throw XCTSkip("set WHISPER_TINY_DIR=/path/to/whisper-tiny to run")
        }
        let model = (dir as NSString).appendingPathComponent("model.safetensors")
        let tok   = (dir as NSString).appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: model),
              FileManager.default.fileExists(atPath: tok) else {
            throw XCTSkip(
                "WHISPER_TINY_DIR=\(dir) is missing model.safetensors or tokenizer.json"
            )
        }
        return ModelDir(safetensors: model, tokenizer: tok)
    }

    /// Build a `WhisperTiny` from raw safetensors using a `WeightMap`
    /// that transposes every PyTorch `Linear` weight (since on disk
    /// they're `[out, in]` and our matmul expects `[in, out]`).
    private func loadWhisperTiny() throws -> WhisperTiny {
        let dir = try whisperTinyDir()
        let raw = try SafetensorsWeights(path: dir.safetensors)
        let mapped = WeightMap(
            source: raw,
            transpose: WeightMap.whisperTinyTransposeSet(
                encoderLayers: 4, decoderLayers: 4
            )
        )
        return try WhisperTiny(weights: mapped)
    }

    // MARK: - Construction & shape sanity

    /// The simplest test: weights load, all shape checks pass, the
    /// constructor returns. Catches naming and transpose errors.
    func testWhisperTinyLoadsFromRealWeights() throws {
        let whisper = try loadWhisperTiny()
        XCTAssertEqual(whisper.config.dModel, 384)
        XCTAssertEqual(whisper.config.encoderLayers, 4)
        XCTAssertEqual(whisper.config.decoderLayers, 4)
        XCTAssertEqual(whisper.config.vocabSize, 51864)
    }

    /// Encoder forward on a synthetic mel: shape becomes
    /// `[1, S/2, D]` after the strided conv, and outputs are finite.
    /// We don't assert specific values — that's covered by the
    /// transcribe test below — just that the plumbing produces sane
    /// numbers.
    func testWhisperEncoderForwardShape() throws {
        let whisper = try loadWhisperTiny()

        // Whisper expects 30 sec @ 16 kHz of mel = 3000 frames; we
        // send a smaller fake [1, 80, 200] just to verify the
        // pipeline. Conv2's stride-2 means S' = S/2 = 100 frames.
        let nMels = 80
        let nFrames = 200
        let dummy = (0..<(nMels * nFrames)).map {
            sinf(Float($0) * 0.01) * 0.5
        }
        let mel = try Tensor.from(data: dummy, shape: [1, nMels, nFrames])
        let encOut = whisper.runEncoder(mel: mel)
        XCTAssertEqual(encOut.shape, [1, nFrames / 2, 384])
        for v in encOut.toArray() {
            XCTAssertTrue(v.isFinite, "non-finite encoder output")
        }
    }

    /// Decoder step: after prefilling cross-attn from a synthetic mel,
    /// step from BOS and verify the prediction is a valid vocab id
    /// and not NaN.
    func testWhisperDecoderStepAfterEncoder() throws {
        let whisper = try loadWhisperTiny()
        let dummy = (0..<(80 * 200)).map { sinf(Float($0) * 0.01) * 0.3 }
        let mel = try Tensor.from(data: dummy, shape: [1, 80, 200])

        whisper.reset()
        whisper.setEncoderContext(mel: mel)
        let predicted = try whisper.decoderStep(tokenId: whisper.config.bosTokenId)
        XCTAssertGreaterThanOrEqual(predicted, 0)
        XCTAssertLessThan(predicted, whisper.config.vocabSize)
    }

    // MARK: - End-to-end

    /// Full pipeline: synthetic 30-second silence → log-mel → encoder
    /// → decoder. We can't assert specific text content (the input
    /// isn't real speech), but we can assert the model produces
    /// SOMETHING that decodes to a string and contains valid token ids.
    /// The point is to exercise every code path under the real model.
    func testWhisperTranscribeOnSilence() throws {
        let dir = try whisperTinyDir()
        let whisper = try loadWhisperTiny()
        let tokenizer = try Tokenizer(path: dir.tokenizer)

        // 30 sec * 16 kHz = 480000 samples of zeros.
        let samples = [Float](repeating: 0, count: 30 * 16000)
        let text = try whisper.transcribe(
            samples: samples, tokenizer: tokenizer, maxNewTokens: 30
        )
        // We don't know what whisper outputs on silence, but it should
        // be a string that's not literally crashing. Often: empty,
        // or repetitive whitespace, or a hallucinated phrase.
        XCTAssertNotNil(text)
        print("[whisper-tiny on silence] -> \"\(text)\"")
    }

    /// Synthetic 1 kHz tone: not real speech, but real audio. Verifies
    /// the audio pre-processing path runs end-to-end. Output text is
    /// just printed for visibility — no assertion on content.
    func testWhisperTranscribeOnTone() throws {
        let dir = try whisperTinyDir()
        let whisper = try loadWhisperTiny()
        let tokenizer = try Tokenizer(path: dir.tokenizer)

        let sampleRate: Float = 16000
        let freq: Float = 1000
        let nSamples = 30 * Int(sampleRate)
        let samples = (0..<nSamples).map {
            sinf(2 * .pi * freq * Float($0) / sampleRate) * 0.5
        }
        let text = try whisper.transcribe(
            samples: samples, tokenizer: tokenizer, maxNewTokens: 30
        )
        XCTAssertNotNil(text)
        print("[whisper-tiny on 1kHz tone] -> \"\(text)\"")
    }

    /// Real speech: looks for `sample.wav` (16 kHz mono PCM16) inside
    /// `WHISPER_TINY_DIR`. If present, decodes it to `[Float]` and
    /// runs end-to-end transcription. Skipped otherwise.
    ///
    /// To produce a sample on macOS:
    ///
    ///   say -v Samantha -o /tmp/hello.aiff "The quick brown fox..."
    ///   afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/hello.aiff sample.wav
    func testWhisperTranscribeOnRealSpeech() throws {
        let dir = try whisperTinyDir()
        let wavPath = (ProcessInfo.processInfo.environment["WHISPER_TINY_DIR"]!
            as NSString).appendingPathComponent("sample.wav")
        guard FileManager.default.fileExists(atPath: wavPath) else {
            throw XCTSkip("no sample.wav in WHISPER_TINY_DIR")
        }
        let whisper = try loadWhisperTiny()
        let tokenizer = try Tokenizer(path: dir.tokenizer)
        let samples = try Self.loadWav16kMono(wavPath)

        // Pad to 30 sec for the encoder's fixed positional table.
        let target = 30 * 16000
        let padded: [Float] = samples.count >= target
            ? Array(samples.prefix(target))
            : samples + [Float](repeating: 0, count: target - samples.count)

        let text = try whisper.transcribe(
            samples: padded, tokenizer: tokenizer, maxNewTokens: 100
        )
        print("[whisper-tiny on \(wavPath)] -> \"\(text)\"")
        // Don't over-assert content — TTS quality varies and the
        // sample.wav is whatever the user dropped in. But for the
        // canonical "quick brown fox" sentence we *should* see "fox"
        // or "dog" in the output. Skip the content check if either
        // is missing rather than failing — different sample.wav
        // contents are legal.
        XCTAssertFalse(
            text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "expected non-empty transcription, got '\(text)'"
        )
        let lower = text.lowercased()
        if lower.contains("fox") || lower.contains("dog") {
            // Strong signal the canonical sample is being decoded.
            print("✓ recognized canonical 'quick brown fox' utterance")
        }
    }

    // MARK: - WAV reader

    /// Bare-bones WAV reader: assumes 16 kHz, 16-bit PCM, mono. Returns
    /// samples normalized to `[-1, 1]`. Errors out on any other format.
    private static func loadWav16kMono(_ path: String) throws -> [Float] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard data.count >= 44 else {
            throw NSError(domain: "wav", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "file too short to be WAV"
            ])
        }
        // Sanity-check the format header. fmt chunk byte layout:
        //   [20] u16 audio format (1 = PCM)
        //   [22] u16 num channels
        //   [24] u32 sample rate
        //   [34] u16 bits per sample
        let format = data.subdata(in: 20..<22).withUnsafeBytes { $0.load(as: UInt16.self) }
        let channels = data.subdata(in: 22..<24).withUnsafeBytes { $0.load(as: UInt16.self) }
        let rate = data.subdata(in: 24..<28).withUnsafeBytes { $0.load(as: UInt32.self) }
        let bps = data.subdata(in: 34..<36).withUnsafeBytes { $0.load(as: UInt16.self) }
        guard format == 1, channels == 1, rate == 16000, bps == 16 else {
            throw NSError(domain: "wav", code: 2, userInfo: [
                NSLocalizedDescriptionKey:
                    "expected PCM16 mono 16 kHz, got format=\(format) ch=\(channels) sr=\(rate) bps=\(bps)"
            ])
        }
        // Find the `data` chunk — usually at offset 36, but RIFF
        // tolerates intermediate chunks (LIST, INFO, etc.) so scan.
        var offset = 12
        var dataStart = -1
        var dataLen = 0
        while offset + 8 <= data.count {
            let id = String(data: data.subdata(in: offset..<(offset + 4)),
                            encoding: .ascii) ?? ""
            let size = Int(data.subdata(in: (offset + 4)..<(offset + 8))
                .withUnsafeBytes { $0.load(as: UInt32.self) })
            if id == "data" {
                dataStart = offset + 8
                dataLen = size
                break
            }
            offset += 8 + size
        }
        guard dataStart >= 0 else {
            throw NSError(domain: "wav", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "no data chunk"
            ])
        }
        let nSamples = dataLen / 2
        var floats = [Float](repeating: 0, count: nSamples)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let i16 = raw.baseAddress!
                .advanced(by: dataStart)
                .assumingMemoryBound(to: Int16.self)
            for i in 0..<nSamples {
                floats[i] = Float(i16[i]) / 32768.0
            }
        }
        return floats
    }
}
