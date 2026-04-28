import XCTest
import Foundation
@testable import Transcribe

/// Layer-by-layer comparator: load the PyTorch reference activations
/// dumped by `scripts/dump_cohere_activations.py` and compare them
/// against our pipeline's intermediate states on the same audio.
/// The first checkpoint where cosine similarity drops below ~0.99
/// tells us where the bug starts.
///
/// To run:
///   1. Generate the reference dump:
///        /tmp/cohere_diff_venv/bin/python \
///          scripts/dump_cohere_activations.py
///   2. Run this test:
///        COHERE_TRANSCRIBE_DIR=/tmp/ml-inspect/cohere-transcribe \
///        COHERE_RUN_E2E=1 \
///        COHERE_REFERENCE_DUMP=/tmp/cohere_pytorch_activations.safetensors \
///        swift test --filter CohereLayerDiffTest
final class CohereLayerDiffTest: XCTestCase {

    private struct DiffStats {
        let oursL2: Float
        let theirsL2: Float
        let cosSim: Float
        let maxAbsDiff: Float
        let meanAbsDiff: Float
    }

    /// Compute per-element diff statistics. `theirs` and `ours` must
    /// have the same flat element count.
    private func diff(theirs: [Float], ours: [Float]) -> DiffStats {
        precondition(theirs.count == ours.count,
            "shape mismatch: theirs=\(theirs.count) ours=\(ours.count)")
        var dot: Float = 0, n1: Float = 0, n2: Float = 0
        var maxAbs: Float = 0, sumAbs: Float = 0
        for i in 0..<theirs.count {
            let a = theirs[i], b = ours[i]
            dot += a * b
            n1 += a * a
            n2 += b * b
            let d = abs(a - b)
            if d > maxAbs { maxAbs = d }
            sumAbs += d
        }
        let cos = dot / (sqrtf(n1) * sqrtf(n2) + 1e-12)
        return DiffStats(
            oursL2: sqrtf(n2),
            theirsL2: sqrtf(n1),
            cosSim: cos,
            maxAbsDiff: maxAbs,
            meanAbsDiff: sumAbs / Float(theirs.count)
        )
    }

    private func report(label: String, stats: DiffStats, stderr: FileHandle) {
        let cosCol = stats.cosSim > 0.99 ? "✓" : (stats.cosSim > 0.5 ? "?" : "✗")
        let line = String(
            format: "  %@ %-38@ L2 ours=%.2f theirs=%.2f ratio=%.3f cos=%.5f max|Δ|=%.3f\n",
            cosCol as NSString,
            label as NSString,
            stats.oursL2,
            stats.theirsL2,
            stats.oursL2 / max(stats.theirsL2, 1e-9),
            stats.cosSim,
            stats.maxAbsDiff
        )
        stderr.write(line.data(using: .utf8)!)
    }

    func testCohereLayerDiff() throws {
        guard ProcessInfo.processInfo.environment["COHERE_RUN_E2E"] == "1"
        else { throw XCTSkip("set COHERE_RUN_E2E=1") }
        guard let modelDir = ProcessInfo.processInfo.environment["COHERE_TRANSCRIBE_DIR"]
        else { throw XCTSkip("set COHERE_TRANSCRIBE_DIR") }
        let dumpPath = ProcessInfo.processInfo.environment["COHERE_REFERENCE_DUMP"]
            ?? "/tmp/cohere_pytorch_activations.safetensors"
        guard FileManager.default.fileExists(atPath: dumpPath) else {
            throw XCTSkip("PyTorch reference dump not at \(dumpPath); " +
                "run scripts/dump_cohere_activations.py first")
        }

        let stderr = FileHandle.standardError
        func say(_ s: String) {
            stderr.write(("[diff] " + s + "\n").data(using: .utf8)!)
        }
        say("loading PyTorch reference activations from \(dumpPath)...")
        let ref = try SafetensorsWeights(path: dumpPath, keepF16: false)
        say("got \(ref.count) reference activations:")
        for k in ref.keys.sorted() {
            if let t = ref[k] {
                say("    \(k): \(t.shape)")
            }
        }

        say("loading our model...")
        let safetensors = (modelDir as NSString).appendingPathComponent("model.safetensors")
        let config = CohereTranscribe.Config(bosTokenId: 4, eosTokenId: 3, padTokenId: 2)
        let weights = try CohereTranscribeWeightMap.load(
            from: safetensors, config: config, keepF16: true
        )
        let model = CohereTranscribe(config: config, weights: weights)

        // Use the same TTS sample.
        let wavPath = (modelDir as NSString).appendingPathComponent("tts_sample.wav")
        let samples = try Self.loadWav16kMono(wavPath)

        // ---- Stage 1: input mel ----
        // PyTorch processor produces input_features shape [1, 128, T_mel].
        // Our AudioPreprocessor.logMelSpectrogram (Cohere config + stored
        // fb override) should produce the same.
        say("stage 1: mel features...")
        let mel = AudioPreprocessor.logMelSpectrogram(
            samples: samples, config: .cohereTranscribe,
            filterbankOverride: model.preprocessorFb
        )
        let melArr = mel.toArray()
        guard let pyInput = ref["input_features"] else {
            throw XCTSkip("input_features missing from reference dump")
        }
        let pyMelArr = pyInput.toArray()
        say("  ours mel shape \(mel.shape) (= [\(mel.shape[0]), \(mel.shape[1])]); " +
            "py shape \(pyInput.shape) (= [B, n_mels, T])")
        // Our mel: [n_mels=128, T_mel]. PyTorch: [1, 128, T_mel].
        // Compare by flattening if shapes are compatible.
        if melArr.count == pyMelArr.count {
            report(label: "input_features",
                   stats: diff(theirs: pyMelArr, ours: melArr), stderr: stderr)
            // The mel divergence is HUGE; print samples to bisect why.
            let oursPrev = Array(melArr.prefix(10)).map { String(format: "%+.4f", $0) }
            let theirsPrev = Array(pyMelArr.prefix(10)).map { String(format: "%+.4f", $0) }
            let comma = ", "
            say("    first 10 ours:   [" + oursPrev.joined(separator: comma) + "]")
            say("    first 10 theirs: [" + theirsPrev.joined(separator: comma) + "]")
            // Per-row stats: PyTorch should have ~unit variance per
            // mel bin if normalize=per_feature. Ours has the same.
            // Mismatch in per-row stats → different normalization.
            let nMels = mel.shape[0]
            let nFrames = mel.shape[1]
            say("    per-row stats (rows 0, 32, 64, 96, 127):")
            for m in [0, 32, 64, 96, 127] where m < nMels {
                var oursMean: Float = 0, theirsMean: Float = 0
                for t in 0..<nFrames {
                    oursMean += melArr[m * nFrames + t]
                    theirsMean += pyMelArr[m * nFrames + t]
                }
                oursMean /= Float(nFrames); theirsMean /= Float(nFrames)
                var oursVar: Float = 0, theirsVar: Float = 0
                for t in 0..<nFrames {
                    let do_ = melArr[m * nFrames + t] - oursMean
                    let dt_ = pyMelArr[m * nFrames + t] - theirsMean
                    oursVar += do_ * do_
                    theirsVar += dt_ * dt_
                }
                oursVar /= Float(nFrames); theirsVar /= Float(nFrames)
                say(String(format:
                    "      bin %3d: ours mean=%+.4f std=%.4f | theirs mean=%+.4f std=%.4f",
                    m, oursMean, sqrtf(oursVar), theirsMean, sqrtf(theirsVar)))
            }
        } else {
            say("  shape mismatch: ours has \(melArr.count) elems, py has \(pyMelArr.count)")
        }

        // ---- Stage 2..N: encoder layer-by-layer ----
        say("stage 2: encoder forward with checkpoints...")
        // Match Python: input shape is [1, 128, T_mel].
        let melB = mel.reshape([1, mel.shape[0], mel.shape[1]])
        let result = model.encoder.forwardWithCheckpoints(
            mel: melB, everyNLayers: 12
        )

        // post-subsampling
        if let py = ref["encoder.pre_encode"] {
            let oursArr = result.postSubsampling.toArray()
            let pyArr = py.toArray()
            if oursArr.count == pyArr.count {
                report(label: "encoder.pre_encode",
                       stats: diff(theirs: pyArr, ours: oursArr), stderr: stderr)
            } else {
                say("  encoder.pre_encode shape mismatch: ours \(result.postSubsampling.shape) py \(py.shape)")
            }
        }

        // per-layer
        for (idx, out) in result.layerOutputs {
            let key = "encoder.layers.\(idx)"
            guard let py = ref[key] else { continue }
            let oursArr = out.toArray()
            let pyArr = py.toArray()
            if oursArr.count == pyArr.count {
                report(label: key,
                       stats: diff(theirs: pyArr, ours: oursArr), stderr: stderr)
            }
        }

        // encoder_final = output of last layer (which we already report
        // above as encoder.layers.47).
        if let py = ref["encoder_final"] {
            let oursArr = result.finalOutput.toArray()
            let pyArr = py.toArray()
            if oursArr.count == pyArr.count {
                report(label: "encoder_final",
                       stats: diff(theirs: pyArr, ours: oursArr), stderr: stderr)
            }
        }

        // encoder→decoder projection
        if let py = ref["encoder_decoder_proj"] {
            let projected = model.runEncoder(mel: melB)  // includes proj
            let oursArr = projected.toArray()
            let pyArr = py.toArray()
            if oursArr.count == pyArr.count {
                report(label: "encoder_decoder_proj",
                       stats: diff(theirs: pyArr, ours: oursArr), stderr: stderr)
            }
        }

        // ---- Stage 3: decoder, autoregressive vs PyTorch teacher-forced ----
        // We run our decoder one prompt token at a time, capturing
        // intermediate states at four checkpoints per step. PyTorch's
        // `decoder.embedding`, `decoder.layer0`, `decoder.layer7`,
        // `decoder.final_norm` shapes are [1, 9, 1024] — for each, we
        // stack our 9 per-step [1, 1, 1024] tensors and compare.
        say("stage 3: decoder (autoregressive vs teacher-forced)...")
        model.reset()
        model.setEncoderContext(mel: melB)
        let prompt = [7, 4, 16, 62, 62, 5, 9, 11, 13]
        let D = config.decoderHidden
        var oursLogitsAll: [Float] = []
        var oursEmbedAll: [Float] = []
        var oursLayer0All: [Float] = []
        var oursLayer7All: [Float] = []
        var oursFinalAll: [Float] = []
        for tok in prompt {
            let dbg = try model.decoderStepDebug(tokenId: tok)
            oursLogitsAll.append(contentsOf: dbg.logits)
            oursEmbedAll.append(contentsOf: dbg.embedding.toArray())
            oursLayer0All.append(contentsOf: dbg.layer0.toArray())
            oursLayer7All.append(contentsOf: dbg.layer7.toArray())
            oursFinalAll.append(contentsOf: dbg.finalNorm.toArray())
        }
        // Compare each stacked vs PyTorch's [1, 9, 1024] (= 9216 elems).
        for (label, ours) in [
            ("decoder.embedding", oursEmbedAll),
            ("decoder.layer0", oursLayer0All),
            ("decoder.layer7", oursLayer7All),
            ("decoder.final_norm", oursFinalAll),
        ] {
            guard let py = ref[label] else { continue }
            let pyArr = py.toArray()
            if ours.count == pyArr.count {
                report(label: label, stats: diff(theirs: pyArr, ours: ours), stderr: stderr)
                // Per-position breakdown.
                for i in 0..<prompt.count {
                    let oursStep = Array(ours[(i * D)..<((i + 1) * D)])
                    let pyStep = Array(pyArr[(i * D)..<((i + 1) * D)])
                    let s = diff(theirs: pyStep, ours: oursStep)
                    let mark = s.cosSim > 0.99 ? "✓" : (s.cosSim > 0.5 ? "?" : "✗")
                    say(String(format: "    %@ %@[pos %d] L2 ratio=%.3f cos=%.5f",
                        mark, label, i, s.oursL2 / max(s.theirsL2, 1e-9), s.cosSim))
                }
            }
        }
        _ = D
        if let pyLogits = ref["head_logits"] {
            let pyArr = pyLogits.toArray()
            if oursLogitsAll.count == pyArr.count {
                // Compare argmax (which is invariant to monotonic
                // transforms like log_softmax that PyTorch's head
                // applies but we skip).
                let V = config.vocabSize
                say("argmax comparison (top-1 prediction at each prompt position):")
                var nMatches = 0
                for i in 0..<prompt.count {
                    let oursStep = Array(oursLogitsAll[(i * V)..<((i + 1) * V)])
                    let pyStep = Array(pyArr[(i * V)..<((i + 1) * V)])
                    var oursMax = oursStep[0], oursMaxIdx = 0
                    var pyMax = pyStep[0], pyMaxIdx = 0
                    for j in 1..<V {
                        if oursStep[j] > oursMax { oursMax = oursStep[j]; oursMaxIdx = j }
                        if pyStep[j] > pyMax { pyMax = pyStep[j]; pyMaxIdx = j }
                    }
                    let mark = oursMaxIdx == pyMaxIdx ? "✓" : "✗"
                    if oursMaxIdx == pyMaxIdx { nMatches += 1 }
                    say(String(format: "  %@ pos %d: input=%d  ours_top1=%d (logit %.2f)  py_top1=%d (logit %.2f)",
                        mark, i, prompt[i], oursMaxIdx, oursMax, pyMaxIdx, pyMax))
                }
                say("argmax matches: \(nMatches)/\(prompt.count)")

                // Also compare hidden states pre-head (decoder.final_norm).
                // Those should match exactly with our pipeline if everything
                // upstream is correct.
                if let pyFinalNorm = ref["decoder.final_norm"] {
                    let _ = pyFinalNorm  // captured for use below
                }
            } else {
                say("  size mismatch: ours \(oursLogitsAll.count) vs py \(pyArr.count)")
            }
        }

        // Also dump our head's raw logits to a file so we can do
        // log_softmax comparison externally if needed.
        let logitsDumpPath = "/tmp/cohere_swift_head_logits.bin"
        let bytes = oursLogitsAll.withUnsafeBufferPointer { Data(buffer: $0) }
        try bytes.write(to: URL(fileURLWithPath: logitsDumpPath))
        say("wrote our head logits to \(logitsDumpPath) (\(bytes.count) bytes)")

        say("done. Look for the first row where cos sim drops below 0.99 — that's the bug.")
    }

    // MARK: - WAV helper (same as other tests)
    private static func loadWav16kMono(_ path: String) throws -> [Float] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard data.count >= 44 else { throw NSError(domain: "wav", code: 1) }
        let rate = data.subdata(in: 24..<28).withUnsafeBytes { $0.load(as: UInt32.self) }
        let bps = data.subdata(in: 34..<36).withUnsafeBytes { $0.load(as: UInt16.self) }
        guard rate == 16000, bps == 16 else {
            throw NSError(domain: "wav", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "want 16kHz PCM16, got \(rate)/\(bps)"])
        }
        var offset = 12
        var dataStart = -1
        var dataLen = 0
        while offset + 8 <= data.count {
            let id = String(data: data.subdata(in: offset..<(offset + 4)),
                            encoding: .ascii) ?? ""
            let size = Int(data.subdata(in: (offset + 4)..<(offset + 8))
                .withUnsafeBytes { $0.load(as: UInt32.self) })
            if id == "data" { dataStart = offset + 8; dataLen = size; break }
            offset += 8 + size
        }
        guard dataStart >= 0 else { throw NSError(domain: "wav", code: 3) }
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
