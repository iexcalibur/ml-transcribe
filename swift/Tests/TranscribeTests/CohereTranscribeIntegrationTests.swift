import XCTest
import Foundation
@testable import Transcribe

/// Opt-in integration tests for the real Cohere Transcribe-2B model.
///
/// Gated by `COHERE_TRANSCRIBE_DIR` — default CI leaves it unset and
/// these tests skip. Running:
///
///   COHERE_TRANSCRIBE_DIR=/tmp/ml-inspect/cohere-transcribe \
///       swift test --filter CohereTranscribeIntegrationTests
///
/// activates them.
///
/// Required files in that directory (download with
/// `huggingface-cli download CohereLabs/cohere-transcribe-03-2026
/// --local-dir /tmp/ml-inspect/cohere-transcribe`, after accepting
/// the license at huggingface.co/CohereLabs/cohere-transcribe-03-2026):
///
///   - model.safetensors      (~4.13 GB, ~1100 tensors)
///   - config.json
///   - tokenizer.json
///   - tokenizer_config.json
///   - tokenizer.model        (SentencePiece byte-fallback BPE)
///
/// As of this commit (Phase 4), the weight loader maps Cohere's
/// PyTorch state-dict names onto our `Weights` bundle. The
/// `testCohereSafetensorsReadable` test only requires the
/// safetensors file to be readable; the full end-to-end test is
/// implemented but currently gated behind a SECOND env var
/// `COHERE_RUN_E2E=1` since the weight-name mapping needs to be
/// finalized once the actual safetensors layout is inspected.
final class CohereTranscribeIntegrationTests: XCTestCase {

    private struct ModelDir {
        let safetensors: String
        let tokenizer: String
        let config: String
    }

    private func cohereDir() throws -> ModelDir {
        guard let dir = ProcessInfo.processInfo.environment["COHERE_TRANSCRIBE_DIR"]
        else {
            throw XCTSkip(
                "set COHERE_TRANSCRIBE_DIR=/path/to/cohere-transcribe to run"
            )
        }
        let model = (dir as NSString).appendingPathComponent("model.safetensors")
        let tok   = (dir as NSString).appendingPathComponent("tokenizer.json")
        let cfg   = (dir as NSString).appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: model) else {
            throw XCTSkip("missing \(model)")
        }
        guard FileManager.default.fileExists(atPath: tok) else {
            throw XCTSkip("missing \(tok) — did the download include the tokenizer?")
        }
        guard FileManager.default.fileExists(atPath: cfg) else {
            throw XCTSkip("missing \(cfg)")
        }
        return ModelDir(safetensors: model, tokenizer: tok, config: cfg)
    }

    /// Step 1: the file is reachable, parses as a valid safetensors
    /// container, and the tensor count is in the expected ballpark
    /// for a 48-layer Conformer + 8-layer decoder + heads (~1000+
    /// tensors).
    ///
    /// Loads with `keepF16: true` to halve memory — Cohere's
    /// safetensors is ~4.13 GB on disk; F32-mode would balloon to
    /// ~16 GB during BF16→F32 conversion.
    func testCohereSafetensorsReadable() throws {
        let dir = try cohereDir()

        // Load with keepF16. A bf16 source still up-converts to F32
        // but f16 sources stay as f16.
        let weights = try SafetensorsWeights(
            path: dir.safetensors, keepF16: true
        )
        XCTAssertGreaterThan(weights.count, 100,
            "expected hundreds of tensors in a 2B-param model, got \(weights.count)")
        print("[cohere-transcribe] loaded \(weights.count) tensors from \(dir.safetensors)")

        // Print a summary of the top-level prefixes.
        var groups: [String: Int] = [:]
        for key in weights.keys {
            let head = key.split(separator: ".", maxSplits: 1)
                .first.map(String.init) ?? key
            groups[head, default: 0] += 1
        }
        print("[cohere-transcribe] top-level groups:")
        for (prefix, count) in groups.sorted(by: { $0.value > $1.value }) {
            print("    \(prefix): \(count)")
        }

        // Print the first 10 tensor names to give a feel for the
        // layout — useful when wiring up the WeightMap.
        let preview = weights.keys.sorted().prefix(10)
        print("[cohere-transcribe] first 10 tensor names (alphabetical):")
        for k in preview {
            if let t = weights[k] {
                print("    \(k) shape=\(t.shape)")
            }
        }
    }

    /// Step 2: the SentencePiece-based tokenizer.json round-trips
    /// through our HuggingFace-tokenizers FFI without crashing.
    /// Cohere's tokenizer.json declares a 16k BPE with byte fallback,
    /// which the `tokenizers` crate handles natively.
    func testCohereTokenizerRoundTrip() throws {
        let dir = try cohereDir()
        let tokenizer = try Tokenizer(path: dir.tokenizer)
        let sample = "Hello world."
        let ids = try tokenizer.encode(sample, addSpecialTokens: false)
        XCTAssertGreaterThan(ids.count, 0,
            "encoded token list should be non-empty")
        let decoded = try tokenizer.decode(ids, skipSpecialTokens: true)
        // Tokenizers normalize whitespace and casing in different ways;
        // just check the key content survives.
        XCTAssertTrue(decoded.lowercased().contains("hello"),
            "tokenizer round-trip dropped 'hello': \(decoded)")
        print("[cohere-transcribe] '\(sample)' -> ids=\(ids) -> '\(decoded)'")
    }

    /// Step 3: build the real `CohereTranscribe.Weights` bundle from
    /// the downloaded safetensors and verify the model constructs
    /// without weight-shape errors. This exercises the entire
    /// `CohereTranscribeWeightMap.load` path — every name lookup,
    /// every `[out,in]→[in,out]` transpose, and every
    /// `CohereTranscribe.Weights` shape constraint that the
    /// constructor checks.
    func testCohereModelLoadsFromRealWeights() throws {
        let dir = try cohereDir()
        let config = CohereTranscribe.Config.cohereTranscribe2B
        let weights = try CohereTranscribeWeightMap.load(
            from: dir.safetensors, config: config, keepF16: true
        )
        let model = CohereTranscribe(config: config, weights: weights)
        XCTAssertEqual(model.config.encoder.nLayers, 48)
        XCTAssertEqual(model.config.decoderLayers, 8)
        XCTAssertEqual(model.config.vocabSize, 16384)
        print("[cohere-transcribe] model constructed successfully")
    }

    /// Step 3.4: ConvSubsampling-only forward — the audio frontend
    /// alone, before any Conformer layer runs. If this fails the bug
    /// is in the conv2d/Linear chain; if it passes the bug is in the
    /// Conformer block.
    func testCohereConvSubsamplingOnly() throws {
        let dir = try cohereDir()
        guard ProcessInfo.processInfo.environment["COHERE_RUN_E2E"] == "1"
        else { throw XCTSkip("set COHERE_RUN_E2E=1 to run") }
        let stderr = FileHandle.standardError
        func say(_ s: String) {
            stderr.write(("[step] " + s + "\n").data(using: .utf8)!)
        }
        say("load weights...")
        let raw = try SafetensorsWeights(path: dir.safetensors, keepF16: true)
        let mapped = WeightMap(
            source: raw,
            transpose: CohereTranscribeWeightMap.cohereTransposeSet(
                encoderLayers: 48, decoderLayers: 8
            )
        )
        say("loaded; building ConvSubsampling weights...")
        let subWeights = try ConvSubsampling.Weights(
            conv1Weight: mapped["encoder.pre_encode.conv.0.weight"]!,
            conv1Bias:   mapped["encoder.pre_encode.conv.0.bias"]!,
            conv2Weight: mapped["encoder.pre_encode.conv.2.weight"]!,
            conv2Bias:   mapped["encoder.pre_encode.conv.2.bias"]!,
            conv3Weight: mapped["encoder.pre_encode.conv.3.weight"]!,
            conv3Bias:   mapped["encoder.pre_encode.conv.3.bias"]!,
            conv4Weight: mapped["encoder.pre_encode.conv.5.weight"]!,
            conv4Bias:   mapped["encoder.pre_encode.conv.5.bias"]!,
            conv5Weight: mapped["encoder.pre_encode.conv.6.weight"]!,
            conv5Bias:   mapped["encoder.pre_encode.conv.6.bias"]!,
            outProjWeight: mapped["encoder.pre_encode.out.weight"]!,
            outProjBias:   mapped["encoder.pre_encode.out.bias"]!
        )
        say("constructing ConvSubsampling...")
        let sub = ConvSubsampling(
            config: ConformerEncoder.Config.cohereTranscribe2B,
            weights: subWeights
        )
        // 5 sec @ 16 kHz, mel → [128, 501].
        let samples = [Float](repeating: 0, count: 5 * 16_000)
        say("computing mel...")
        let mel = AudioPreprocessor.logMelSpectrogram(
            samples: samples, config: .cohereTranscribe
        )
        say("  mel shape: \(mel.shape)")
        let melBatched = mel.reshape([1, mel.shape[0], mel.shape[1]])
        say("running ConvSubsampling.forward...")
        let t = Date()
        let out = sub.forward(mel: melBatched)
        say("  conv out shape: \(out.shape)  (\(String(format: "%.1f", Date().timeIntervalSince(t)))s)")
        for v in out.toArray().prefix(50) {
            XCTAssertTrue(v.isFinite, "non-finite ConvSubsampling output: \(v)")
        }
        say("ok — ConvSubsampling produced finite output")
    }

    /// Compare Cohere's stored mel filterbank values against the
    /// HTK-Slaney filterbank we compute internally. If these differ
    /// meaningfully (e.g. >1e-3 RMSE in the row sums), our DSP is
    /// emitting features the model wasn't trained on.
    func testCohereStoredFilterbankSanityCheck() throws {
        let dir = try cohereDir()
        guard ProcessInfo.processInfo.environment["COHERE_RUN_E2E"] == "1"
        else { throw XCTSkip("set COHERE_RUN_E2E=1 to run") }
        let weights = try SafetensorsWeights(path: dir.safetensors, keepF16: true)
        guard let fb = weights["preprocessor.featurizer.fb"] else {
            throw XCTSkip("no preprocessor.featurizer.fb in safetensors")
        }
        let stderr = FileHandle.standardError
        stderr.write(("fb shape: \(fb.shape)\n").data(using: .utf8)!)
        let fbData = fb.toArray()
        let nMels = fb.shape[1]
        let nFreqs = fb.shape[2]
        // Print row sums (should be ~1 for area-normalized Slaney
        // filterbank, or ~triangular peak for HTK).
        stderr.write("  row sums (first 10):\n".data(using: .utf8)!)
        for m in 0..<10 {
            let off = m * nFreqs
            var sum: Float = 0
            for k in 0..<nFreqs {
                sum += fbData[off + k]
            }
            stderr.write("    row \(m): sum=\(sum)\n".data(using: .utf8)!)
        }
        // Print row 5's nonzero columns
        stderr.write("  row 5 nonzero columns:\n".data(using: .utf8)!)
        let row5Off = 5 * nFreqs
        for k in 0..<nFreqs {
            let v = fbData[row5Off + k]
            if v != 0 {
                stderr.write("    col \(k): \(v)\n".data(using: .utf8)!)
            }
        }
        _ = nMels
    }

    /// Decode each generated token id individually to see what
    /// SentencePiece thinks it is. Useful when the integrated decode
    /// (`skipSpecialTokens=true`) is hiding what's actually being
    /// produced.
    func testCohereDecodeIndividualTokens() throws {
        let dir = try cohereDir()
        guard ProcessInfo.processInfo.environment["COHERE_RUN_E2E"] == "1"
        else { throw XCTSkip("set COHERE_RUN_E2E=1 to run") }
        let tokenizer = try Tokenizer(path: dir.tokenizer)
        let stderr = FileHandle.standardError
        // From the latest E2E run: prompt + first 14 generated tokens.
        let allIds: [UInt32] = [
            7, 4, 16, 62, 62, 5, 9, 11, 13,             // prompt
            5, 13785, 1617, 752, 832, 832, 752, 752,    // generated
            752, 13785, 752, 1899, 1899, 651,
        ]
        for id in allIds {
            let decoded = try tokenizer.decode([id], skipSpecialTokens: false)
            stderr.write("  \(id) → \(decoded.debugDescription)\n".data(using: .utf8)!)
        }
    }

    /// Capture the top-K predicted logits + probabilities at each
    /// decoder step. If the top probabilities are diffuse (all
    /// roughly equal), the model is uncertain — explains repetition.
    /// If top-1 is sharply peaked but on the wrong token, that's a
    /// different bug.
    func testCohereTopKLogitsPerStep() throws {
        let dir = try cohereDir()
        guard ProcessInfo.processInfo.environment["COHERE_RUN_E2E"] == "1"
        else { throw XCTSkip("set COHERE_RUN_E2E=1 to run") }
        let stderr = FileHandle.standardError
        let config = CohereTranscribe.Config(bosTokenId: 4, eosTokenId: 3, padTokenId: 2)
        let weights = try CohereTranscribeWeightMap.load(
            from: dir.safetensors, config: config, keepF16: true
        )
        let model = CohereTranscribe(config: config, weights: weights)
        let tokenizer = try Tokenizer(path: dir.tokenizer)

        let dirPath = ProcessInfo.processInfo.environment["COHERE_TRANSCRIBE_DIR"]!
        let wavPath = (dirPath as NSString).appendingPathComponent("tts_sample.wav")
        let samples = try Self.loadWav16kMono(wavPath)
        let mel = AudioPreprocessor.logMelSpectrogram(
            samples: samples, config: .cohereTranscribe,
            filterbankOverride: model.preprocessorFb
        )
        let melB = mel.reshape([1, mel.shape[0], mel.shape[1]])
        model.reset()
        model.setEncoderContext(mel: melB)

        // Step through each prompt token and print top-K of what
        // the model predicts NEXT.
        let prompt = [7, 4, 16, 62, 62, 5, 9, 11, 13]
        for (stepIdx, tok) in prompt.enumerated() {
            let logits = try model.decoderStepLogits(tokenId: tok)
            let label = "after prompt[\(stepIdx)]=\(tok)"
            printTopK(logits: logits, label: label,
                      tokenizer: tokenizer, k: 8, stderr: stderr)
        }
        // 5 more generation steps following greedy.
        var lastPred = argmax(try model.decoderStepLogits(tokenId: prompt.last!))
        // Note: we already ran the last prompt step above, so the
        // KV cache is now at length=9. We can't run again; instead
        // do greedy continuation.
        var emitted: [Int] = []
        for _ in 0..<8 {
            if lastPred == config.eosTokenId { break }
            emitted.append(lastPred)
            let logits = try model.decoderStepLogits(tokenId: lastPred)
            printTopK(logits: logits, label: "after gen=\(lastPred)",
                      tokenizer: tokenizer, k: 5, stderr: stderr)
            lastPred = argmax(logits)
        }
        stderr.write(("[result] generated tokens: \(emitted)\n").data(using: .utf8)!)
        let text = try tokenizer.decode(
            emitted.map { UInt32($0) }, skipSpecialTokens: true)
        stderr.write(("[result] text: \(text.debugDescription)\n").data(using: .utf8)!)
    }

    private func argmax(_ xs: [Float]) -> Int {
        var best = 0; var bestVal = xs[0]
        for i in 1..<xs.count where xs[i] > bestVal { bestVal = xs[i]; best = i }
        return best
    }

    private func printTopK(logits: [Float], label: String,
                           tokenizer: Tokenizer, k: Int,
                           stderr: FileHandle) {
        var indexed = logits.enumerated().map { ($0.offset, $0.element) }
        indexed.sort { $0.1 > $1.1 }
        let top = Array(indexed.prefix(k))
        let maxLogit = top[0].1
        var partition: Float = 0
        for v in logits { partition += expf(v - maxLogit) }
        var lines = ["[\(label)] top-\(k) (max logit \(String(format: "%.2f", maxLogit))):\n"]
        for (i, v) in top {
            let prob = expf(v - maxLogit) / partition
            let tok = (try? tokenizer.decode([UInt32(i)], skipSpecialTokens: false)) ?? "<?>"
            lines.append("    \(i) (\(tok)): logit=\(String(format: "%+.3f", v)) prob=\(String(format: "%.4f", prob))\n")
        }
        stderr.write(lines.joined().data(using: .utf8)!)
    }

    /// Inspect the decoder's embedding LN gamma + final LN gamma.
    /// If either has tiny gammas concentrated in specific dimensions,
    /// the residual stream may project onto a direction that
    /// over-favors specific tokens.
    func testCohereDecoderNormStats() throws {
        let dir = try cohereDir()
        guard ProcessInfo.processInfo.environment["COHERE_RUN_E2E"] == "1"
        else { throw XCTSkip("set COHERE_RUN_E2E=1 to run") }
        let stderr = FileHandle.standardError
        let raw = try SafetensorsWeights(path: dir.safetensors, keepF16: true)
        for name in [
            "transf_decoder._embedding.layer_norm.weight",
            "transf_decoder._embedding.layer_norm.bias",
            "transf_decoder._decoder.final_layer_norm.weight",
            "transf_decoder._decoder.final_layer_norm.bias",
        ] {
            guard let t = raw[name] else { continue }
            let arr = t.toArray()
            let mn = arr.min()!, mx = arr.max()!
            let mean = arr.reduce(0, +) / Float(arr.count)
            let absMean = arr.map { abs($0) }.reduce(0, +) / Float(arr.count)
            let nearZero = arr.filter { abs($0) < 0.05 }.count
            stderr.write((
                "  \(name): min=\(String(format: "%+.3f", mn)) " +
                "max=\(String(format: "%+.3f", mx)) " +
                "mean=\(String(format: "%+.3f", mean)) " +
                "avg|x|=\(String(format: "%.3f", absMean)) " +
                "near_zero=\(nearZero)/\(arr.count)\n"
            ).data(using: .utf8)!)
        }
    }

    /// Look at the LM head bias and weight. If certain tokens have
    /// huge biases (e.g. BOS=4, "the"=527 if those are over-biased)
    /// the model would predict them regardless of decoder hidden
    /// state.
    func testCohereHeadBiasAndWeightsTopK() throws {
        let dir = try cohereDir()
        guard ProcessInfo.processInfo.environment["COHERE_RUN_E2E"] == "1"
        else { throw XCTSkip("set COHERE_RUN_E2E=1 to run") }
        let stderr = FileHandle.standardError
        let raw = try SafetensorsWeights(path: dir.safetensors, keepF16: true)
        let bias = raw["log_softmax.mlp.layer0.bias"]!
        let arr = bias.toArray()
        // Sort by descending value, show top 10 + bottom 10.
        var indexed = arr.enumerated().map { ($0.offset, $0.element) }
        indexed.sort { $0.1 > $1.1 }
        stderr.write("HEAD BIAS top 15:\n".data(using: .utf8)!)
        for (i, v) in indexed.prefix(15) {
            stderr.write(("  id=\(i) bias=\(String(format: "%+.3f", v))\n").data(using: .utf8)!)
        }
        stderr.write("HEAD BIAS bottom 5:\n".data(using: .utf8)!)
        for (i, v) in indexed.suffix(5) {
            stderr.write(("  id=\(i) bias=\(String(format: "%+.3f", v))\n").data(using: .utf8)!)
        }
        // Specific tokens we care about.
        let watchIds: [Int] = [3, 4, 5, 7, 9, 11, 13, 16, 62, 527, 752, 832, 13764, 13785, 1617]
        stderr.write("HEAD BIAS for specific tokens:\n".data(using: .utf8)!)
        for i in watchIds {
            stderr.write(("  id=\(i) bias=\(String(format: "%+.3f", arr[i]))\n").data(using: .utf8)!)
        }

        // Check stats
        let mean = arr.reduce(0, +) / Float(arr.count)
        let mn = arr.min()!, mx = arr.max()!
        let std = self.stddev(arr)
        stderr.write(("HEAD BIAS stats: mean=\(mean) std=\(std) min=\(mn) max=\(mx)\n").data(using: .utf8)!)
    }

    /// Inspect cross-attention attention scores. After running the
    /// encoder + first decoder step, dump Q@K.T softmax scores per
    /// head. If scores are uniform (~1/T_enc everywhere), the
    /// decoder isn't focusing on specific encoder time steps;
    /// repetition is the natural consequence.
    func testCohereCrossAttentionScores() throws {
        let dir = try cohereDir()
        guard ProcessInfo.processInfo.environment["COHERE_RUN_E2E"] == "1"
        else { throw XCTSkip("set COHERE_RUN_E2E=1 to run") }
        let stderr = FileHandle.standardError

        let config = CohereTranscribe.Config(bosTokenId: 4, eosTokenId: 3, padTokenId: 2)
        let weights = try CohereTranscribeWeightMap.load(
            from: dir.safetensors, config: config, keepF16: true
        )
        let dirPath = ProcessInfo.processInfo.environment["COHERE_TRANSCRIBE_DIR"]!
        let wavPath = (dirPath as NSString).appendingPathComponent("tts_sample.wav")
        let samples = try Self.loadWav16kMono(wavPath)

        // Run encoder, project to decoder dim.
        let mel = AudioPreprocessor.logMelSpectrogram(
            samples: samples, config: .cohereTranscribe,
            filterbankOverride: weights.preprocessorFb.map {
                $0.shape.count == 3 && $0.shape[0] == 1
                    ? $0.reshape([$0.shape[1], $0.shape[2]])
                    : $0
            }
        )
        let melB = mel.reshape([1, mel.shape[0], mel.shape[1]])

        // Compute encoder + projection manually.
        let model = CohereTranscribe(config: config, weights: weights)
        let encOut = model.runEncoder(mel: melB)    // [1, T_enc, 1024]
        let T_enc = encOut.shape[1]
        let decDim = config.decoderHidden
        let H = config.decoderHeads
        let Dh = config.decoderHeadDim
        stderr.write(("[cross-attn] T_enc=\(T_enc) D=\(decDim) H=\(H) Dh=\(Dh)\n").data(using: .utf8)!)

        // Look at layer 0's cross-attn weights.
        let layerW = weights.decoderLayers[0]
        // K = encOut @ caWK + caBK; reshape to [H, T_enc, Dh].
        let bK = layerW.caBK.reshape([1, 1, decDim])
        let kFull = encOut.matmul(layerW.caWK).add(bK)
        let kH = kFull.reshape([1, T_enc, H, Dh])
            .permute([0, 2, 1, 3]).contiguous()
            .reshape([H, T_enc, Dh])

        // Build a synthetic Q (random-but-deterministic so the test
        // is reproducible). What we want is to see how K varies
        // across encoder time steps for a fixed Q.
        var rng = SeededRNG(seed: 42)
        let qData = (0..<(H * Dh)).map { _ in Float.random(in: -1.0...1.0, using: &rng) }
        let q = try Tensor.from(data: qData, shape: [H, 1, Dh])

        // Compute attention: softmax(Q @ K.T / sqrt(Dh))
        let kT = kH.permute([0, 2, 1]).contiguous()    // [H, Dh, T_enc]
        let scores = q.matmul(kT)                       // [H, 1, T_enc]
        let scoresArr = scores.toArray()
        // Print per-head: min, max, range. If max/min ratio is huge,
        // softmax will peak. If close to 1, softmax is uniform.
        for h in 0..<H {
            let off = h * T_enc
            var row: [Float] = []
            for t in 0..<T_enc { row.append(scoresArr[off + t]) }
            let mn = row.min()!
            let mx = row.max()!
            let mean = row.reduce(0, +) / Float(T_enc)
            stderr.write(("  head \(h): score min=\(String(format: "%.2f", mn)) max=\(String(format: "%.2f", mx)) mean=\(String(format: "%.2f", mean)) range=\(String(format: "%.2f", mx - mn))\n").data(using: .utf8)!)
        }

        // Also K stats across time per head.
        let kArr = kH.toArray()
        for h in 0..<min(2, H) {
            var l2s: [Float] = []
            for t in 0..<T_enc {
                let off = (h * T_enc + t) * Dh
                var s: Float = 0
                for d in 0..<Dh { let v = kArr[off + d]; s += v * v }
                l2s.append(sqrtf(s))
            }
            let mn = l2s.min()!, mx = l2s.max()!
            let avg = l2s.reduce(0, +) / Float(T_enc)
            stderr.write(("  head \(h) K L2: min=\(String(format: "%.3f", mn)) max=\(String(format: "%.3f", mx)) avg=\(String(format: "%.3f", avg))\n").data(using: .utf8)!)
        }
    }

    /// Compare different prompt strategies on the SAME audio sample.
    /// The README's quick-start uses just `[decoder_start_token_id=13764]`,
    /// `model.transcribe()` uses the full build_prompt. Try a few
    /// permutations and see which produces sensible English.
    func testCoherePromptVariants() throws {
        let dir = try cohereDir()
        guard ProcessInfo.processInfo.environment["COHERE_RUN_E2E"] == "1"
        else { throw XCTSkip("set COHERE_RUN_E2E=1 to run") }
        let stderr = FileHandle.standardError
        let config = CohereTranscribe.Config(bosTokenId: 4, eosTokenId: 3, padTokenId: 2)
        let weights = try CohereTranscribeWeightMap.load(
            from: dir.safetensors, config: config, keepF16: true
        )
        let model = CohereTranscribe(config: config, weights: weights)
        let tokenizer = try Tokenizer(path: dir.tokenizer)
        let dirPath = ProcessInfo.processInfo.environment["COHERE_TRANSCRIBE_DIR"]!
        let wavPath = (dirPath as NSString).appendingPathComponent("tts_sample.wav")
        let samples = try Self.loadWav16kMono(wavPath)

        // Run encoder once and reuse for all prompts.
        let mel = AudioPreprocessor.logMelSpectrogram(
            samples: samples, config: .cohereTranscribe,
            filterbankOverride: model.preprocessorFb
        )
        let melB = mel.reshape([1, mel.shape[0], mel.shape[1]])

        let promptVariants: [(String, [Int])] = [
            ("just decoder_start (13764)", [13764]),
            ("build_prompt (NeMo style)", [7, 4, 16, 62, 62, 5, 9, 11, 13]),
            ("13764 + build_prompt", [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]),
            ("just BOS (4)", [4]),
            ("BOS + build_prompt", [4, 16, 62, 62, 5, 9, 11, 13]),
        ]

        for (label, promptTokens) in promptVariants {
            model.reset()
            model.setEncoderContext(mel: melB)
            var nextTok = -1
            for tok in promptTokens {
                nextTok = try model.decoderStep(tokenId: tok)
            }
            var tokens: [Int] = []
            for _ in 0..<25 {
                if nextTok == config.eosTokenId { break }
                tokens.append(nextTok)
                nextTok = try model.decoderStep(tokenId: nextTok)
            }
            let text = try tokenizer.decode(
                tokens.map { UInt32($0) }, skipSpecialTokens: true)
            stderr.write((
                "[\(label)] tokens=\(tokens.prefix(10)) text=\(text.debugDescription)\n"
            ).data(using: .utf8)!)
        }
    }

    /// Decode the `decoder_start_token_id` (13764) from
    /// generation_config.json. The README quick-start example does
    /// NOT pass text= to the processor, so the actual default
    /// decoder prompt is just `[13764]`, not the long build_prompt
    /// sequence (which is only used by `model.transcribe()`).
    func testCohereDecodeMagicStartToken() throws {
        let dir = try cohereDir()
        guard ProcessInfo.processInfo.environment["COHERE_RUN_E2E"] == "1"
        else { throw XCTSkip("set COHERE_RUN_E2E=1 to run") }
        let tokenizer = try Tokenizer(path: dir.tokenizer)
        let stderr = FileHandle.standardError
        for id: UInt32 in [4, 5, 7, 9, 11, 13, 16, 62, 13764] {
            let s = try tokenizer.decode([id], skipSpecialTokens: false)
            stderr.write(("  \(id) → \(s.debugDescription)\n").data(using: .utf8)!)
        }
    }

    /// Verify the prompt token order I hardcoded matches what NeMo's
    /// `build_prompt(language='en', punctuation=True)` produces when
    /// fed through the SentencePiece tokenizer with
    /// `add_special_tokens=False`. If the actual encoding differs,
    /// the model is being primed wrong.
    func testCohereBuildPromptTokenization() throws {
        let dir = try cohereDir()
        guard ProcessInfo.processInfo.environment["COHERE_RUN_E2E"] == "1"
        else { throw XCTSkip("set COHERE_RUN_E2E=1 to run") }
        let tokenizer = try Tokenizer(path: dir.tokenizer)
        let prompt = "<|startofcontext|><|startoftranscript|><|emo:undefined|><|en|><|en|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>"
        let stderr = FileHandle.standardError
        let ids = try tokenizer.encode(prompt, addSpecialTokens: false)
        stderr.write(("[prompt] '\(prompt)'\n").data(using: .utf8)!)
        stderr.write(("[prompt] ids (no special): \(ids)\n").data(using: .utf8)!)
        let withSpecial = try tokenizer.encode(prompt, addSpecialTokens: true)
        stderr.write(("[prompt] ids (with special): \(withSpecial)\n").data(using: .utf8)!)

        // Decode each id back to verify
        for id in ids {
            let s = try tokenizer.decode([id], skipSpecialTokens: false)
            stderr.write(("  \(id) → \(s.debugDescription)\n").data(using: .utf8)!)
        }
    }

    /// Inspect the per-layer `norm_out.weight` (LayerNorm gamma)
    /// for the encoder. If the gammas are tiny (close to 0) for
    /// the last layers, the model's design legitimately produces a
    /// small-magnitude encoder output and the encoder→decoder
    /// projection is supposed to scale it back up. If they're
    /// uniform (~1.0) and the magnitude still drops, our forward
    /// pass has a bug.
    func testCohereNormOutGammaPerLayer() throws {
        let dir = try cohereDir()
        guard ProcessInfo.processInfo.environment["COHERE_RUN_E2E"] == "1"
        else { throw XCTSkip("set COHERE_RUN_E2E=1 to run") }
        let raw = try SafetensorsWeights(path: dir.safetensors, keepF16: true)
        let stderr = FileHandle.standardError
        for i in 0..<48 {
            let name = "encoder.layers.\(i).norm_out.weight"
            guard let g = raw[name] else { continue }
            let arr = g.toArray()
            var minV = arr[0], maxV = arr[0]
            var sumAbs: Float = 0
            for v in arr {
                if v < minV { minV = v }
                if v > maxV { maxV = v }
                sumAbs += abs(v)
            }
            let avgAbs = sumAbs / Float(arr.count)
            let line = String(
                format: "  layer %2d norm_out: min=%+.3f max=%+.3f avg|x|=%.3f\n",
                i, minV, maxV, avgAbs)
            stderr.write(line.data(using: .utf8)!)
        }
    }

    /// Diagnostic for "encoder produces flat features" hypothesis.
    /// Runs the encoder with checkpoints at layers 0, 12, 24, 36, 47,
    /// and the post-subsampling input. For each checkpoint, prints
    /// adjacent-time-step cosine similarities. If adjacent t's are
    /// nearly identical, that layer is collapsing time-variation.
    /// Bisecting tells us WHERE the bug is.
    func testCohereEncoderTimeVariance() throws {
        let dir = try cohereDir()
        guard ProcessInfo.processInfo.environment["COHERE_RUN_E2E"] == "1"
        else { throw XCTSkip("set COHERE_RUN_E2E=1 to run") }
        let stderr = FileHandle.standardError
        func say(_ s: String) {
            stderr.write(("[diag] " + s + "\n").data(using: .utf8)!)
        }
        let config = CohereTranscribe.Config.cohereTranscribe2B
        say("loading weights...")
        let weights = try CohereTranscribeWeightMap.load(
            from: dir.safetensors, config: config, keepF16: true
        )
        let model = CohereTranscribe(config: config, weights: weights)

        // Use the TTS "quick brown fox" sample.
        let dirPath = ProcessInfo.processInfo.environment["COHERE_TRANSCRIBE_DIR"]!
        let wavCandidates = [
            (dirPath as NSString).appendingPathComponent("tts_sample.wav"),
            "/tmp/tts_sample.wav",
        ]
        var wavPath: String?
        for p in wavCandidates where FileManager.default.fileExists(atPath: p) {
            wavPath = p; break
        }
        guard let wp = wavPath else {
            let joined = wavCandidates.joined(separator: ", ")
            throw XCTSkip("no tts_sample.wav at \(joined)")
        }
        let samples = try Self.loadWav16kMono(wp)
        say("audio: \(samples.count / 16_000) sec from \(wp)")

        say("computing mel + checkpoints...")
        let mel = AudioPreprocessor.logMelSpectrogram(
            samples: samples, config: .cohereTranscribe,
            filterbankOverride: model.preprocessorFb
        )
        let melB = mel.reshape([1, mel.shape[0], mel.shape[1]])

        // Bisect aggressively around the suspicious last-layer drop:
        // checkpoint every 3 layers so we can see exactly where L2
        // magnitude collapses.
        let result = model.encoder.forwardWithCheckpoints(
            mel: melB, everyNLayers: 3
        )
        report(label: "post-subsampling (input to layer 0)",
               t: result.postSubsampling, say: say)
        for (idx, out) in result.layerOutputs {
            report(label: "after layer \(idx)", t: out, say: say)
        }
        report(label: "encoder final (= last layer)",
               t: result.finalOutput, say: say)
    }

    /// Print time-variance stats for a [1, T, D] tensor.
    private func report(label: String, t: Tensor, say: (String) -> Void) {
        let shape = t.shape
        guard shape.count == 3, shape[0] == 1 else {
            say("[\(label)] unexpected shape \(shape)")
            return
        }
        let T = shape[1]
        let D = shape[2]
        let arr = t.toArray()

        // L2 norm per t.
        var l2: [Float] = []
        for ti in 0..<T {
            let off = ti * D
            var s: Float = 0
            for d in 0..<D { let v = arr[off + d]; s += v * v }
            l2.append(sqrtf(s))
        }
        // Adjacent cosine similarity.
        var adj: [Float] = []
        for ti in 0..<(T - 1) {
            let o1 = ti * D, o2 = (ti + 1) * D
            var dot: Float = 0, n1: Float = 0, n2: Float = 0
            for d in 0..<D {
                let a = arr[o1 + d], b = arr[o2 + d]
                dot += a * b; n1 += a * a; n2 += b * b
            }
            adj.append(dot / (sqrtf(n1) * sqrtf(n2) + 1e-9))
        }
        let l2Avg = l2.reduce(0, +) / Float(T)
        let adjAvg = adj.reduce(0, +) / Float(max(1, adj.count))
        say("[\(label)] T=\(T) D=\(D) l2_avg=\(String(format: "%.3f", l2Avg)) " +
            "l2_std=\(String(format: "%.3f", stddev(l2))) " +
            "adj_cos_avg=\(String(format: "%.4f", adjAvg)) " +
            "(min=\(String(format: "%.4f", adj.min() ?? 0)) " +
            "max=\(String(format: "%.4f", adj.max() ?? 0)))")
    }

    /// Sample standard deviation helper.
    private func stddev(_ xs: [Float]) -> Float {
        let mean = xs.reduce(0, +) / Float(xs.count)
        let sq = xs.map { ($0 - mean) * ($0 - mean) }.reduce(0, +)
        return sqrtf(sq / Float(max(1, xs.count - 1)))
    }

    /// Bisect helper: feed silence vs real speech through the same
    /// model and print the tokens each produces. If the outputs
    /// differ, the encoder is reading audio (good); if identical,
    /// the encoder is being ignored (bad).
    func testCohereEncoderDifferentiatesSilenceFromSpeech() throws {
        let dir = try cohereDir()
        guard ProcessInfo.processInfo.environment["COHERE_RUN_E2E"] == "1"
        else { throw XCTSkip("set COHERE_RUN_E2E=1 to run") }

        let stderr = FileHandle.standardError
        func say(_ s: String) {
            stderr.write(("[bisect] " + s + "\n").data(using: .utf8)!)
        }
        let config = CohereTranscribe.Config(bosTokenId: 4, eosTokenId: 3, padTokenId: 2)
        let weights = try CohereTranscribeWeightMap.load(
            from: dir.safetensors, config: config, keepF16: true
        )
        let model = CohereTranscribe(config: config, weights: weights)
        let tokenizer = try Tokenizer(path: dir.tokenizer)
        let prompt = [7, 4, 16, 62, 62, 5, 9, 11, 13]

        func decode(samples: [Float], label: String) throws {
            model.reset()
            let mel = AudioPreprocessor.logMelSpectrogram(
                samples: samples, config: .cohereTranscribe
            )
            let melB = mel.reshape([1, mel.shape[0], mel.shape[1]])
            model.setEncoderContext(mel: melB)
            var nextTok = -1
            for tok in prompt {
                nextTok = try model.decoderStep(tokenId: tok)
            }
            var tokens: [Int] = []
            for _ in 0..<10 {
                if nextTok == config.eosTokenId { break }
                tokens.append(nextTok)
                nextTok = try model.decoderStep(tokenId: nextTok)
            }
            let text = try tokenizer.decode(
                tokens.map { UInt32($0) }, skipSpecialTokens: true
            )
            say("\(label) -> tokens=\(tokens), text=\(text.debugDescription)")
        }

        say("decoding silence...")
        try decode(samples: [Float](repeating: 0, count: 5 * 16_000), label: "silence")
        say("decoding voxpopuli sample...")
        let wavPath = (ProcessInfo.processInfo.environment["COHERE_TRANSCRIBE_DIR"]!
            as NSString).appendingPathComponent("demo/voxpopuli_test_en_demo.wav")
        let speech = try Self.loadWav16kMono(wavPath)
        try decode(samples: speech, label: "voxpopuli")
    }

    /// Step 3.5: encoder-only forward on 5 seconds of silence.
    /// Lighter than full E2E so we can see whether the Conformer
    /// encoder runs end-to-end on real weights without crashing,
    /// before paying for the decoder loop.
    func testCohereEncoderOnlyForward() throws {
        let dir = try cohereDir()
        guard ProcessInfo.processInfo.environment["COHERE_RUN_E2E"] == "1"
        else {
            throw XCTSkip("set COHERE_RUN_E2E=1 to run; encoder forward is slow on CPU")
        }
        let stderr = FileHandle.standardError
        func say(_ s: String) {
            // Unbuffered write so we see progress even mid-test if
            // a panic kills the process.
            stderr.write(("[step] " + s + "\n").data(using: .utf8)!)
        }
        say("loading weights...")
        let t0 = Date()
        let config = CohereTranscribe.Config.cohereTranscribe2B
        let weights = try CohereTranscribeWeightMap.load(
            from: dir.safetensors, config: config, keepF16: true
        )
        say("loaded in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")

        say("constructing model...")
        let t1 = Date()
        let model = CohereTranscribe(config: config, weights: weights)
        say("constructed in \(String(format: "%.1f", Date().timeIntervalSince(t1)))s")

        // 5 sec of silence → ~500 frames before subsampling, ~62 after.
        let samples = [Float](repeating: 0, count: 5 * 16_000)
        say("running mel + encoder forward...")
        let t2 = Date()
        let mel = AudioPreprocessor.logMelSpectrogram(
            samples: samples, config: .cohereTranscribe
        )
        say("  mel shape: \(mel.shape)")
        let melBatched = mel.reshape([1, mel.shape[0], mel.shape[1]])
        let encOut = model.runEncoder(mel: melBatched)
        say("  encoder out shape: \(encOut.shape)")
        say("encoder forward took \(String(format: "%.1f", Date().timeIntervalSince(t2)))s")

        XCTAssertEqual(encOut.shape[0], 1)
        XCTAssertEqual(encOut.shape[2], config.encDecProjToDim)
        for v in encOut.toArray().prefix(100) {
            XCTAssertTrue(v.isFinite, "non-finite encoder output: \(v)")
        }
        say("ok — encoder produced finite output")
    }

    /// Step 4: actual end-to-end transcription. Skipped unless
    /// `COHERE_RUN_E2E=1` is also set (the full encoder forward over
    /// 48 Conformer layers at d_model=1280 takes minutes on CPU).
    ///
    /// Uses the bundled `demo/voxpopuli_test_en_demo.wav` if it
    /// exists, else falls back to `sample.wav` next to the
    /// safetensors (any 16 kHz mono PCM16 file works).
    ///
    /// Per the model card: `decoder_start_token_id = 13764` is the
    /// canonical start-of-prompt token for this checkpoint. We prime
    /// the decoder with just that token and let it generate.
    func testCohereTranscribeEndToEnd() throws {
        let dir = try cohereDir()
        guard ProcessInfo.processInfo.environment["COHERE_RUN_E2E"] == "1"
        else {
            throw XCTSkip(
                "set COHERE_RUN_E2E=1 to run; expect minutes on CPU"
            )
        }
        // Pick an audio sample. `COHERE_AUDIO_SAMPLE` env var lets you
        // override; otherwise prefer the canonical voxpopuli demo,
        // then a tts_sample.wav (clean slow speech for sanity), then
        // any sample.wav at the directory root.
        let dirPath = ProcessInfo.processInfo.environment["COHERE_TRANSCRIBE_DIR"]!
        let envSample = ProcessInfo.processInfo.environment["COHERE_AUDIO_SAMPLE"]
        let candidates = [
            envSample,
            (dirPath as NSString).appendingPathComponent("demo/voxpopuli_test_en_demo.wav"),
            (dirPath as NSString).appendingPathComponent("tts_sample.wav"),
            (dirPath as NSString).appendingPathComponent("sample.wav"),
        ].compactMap { $0 }
        var samplePath: String?
        for p in candidates where FileManager.default.fileExists(atPath: p) {
            samplePath = p; break
        }
        guard let wavPath = samplePath else {
            throw XCTSkip(
                "no audio sample found at \(candidates.joined(separator: ", "))"
            )
        }

        // Build model from real weights.
        let config = CohereTranscribe.Config(
            // Override BOS/EOS/PAD with values from generation_config.json.
            // (Defaults in CohereTranscribe.Config use placeholder 0s.)
            bosTokenId: 4,
            eosTokenId: 3,
            padTokenId: 2
        )
        // Try F32 (keepF16=false). BF16 → F32 conversion increases
        // memory ~2x but eliminates F16 precision questions. The
        // peak memory is ~14 GiB; needs ~13 GiB free to succeed.
        let useF16 = ProcessInfo.processInfo.environment["COHERE_USE_F16"] != "0"
        let weights = try CohereTranscribeWeightMap.load(
            from: dir.safetensors, config: config, keepF16: useF16
        )
        let model = CohereTranscribe(config: config, weights: weights)
        let tokenizer = try Tokenizer(path: dir.tokenizer)
        let prec = useF16 ? "F16" : "F32"
        FileHandle.standardError.write(
            ("[step] precision: " + prec + "\n").data(using: .utf8)!
        )

        // Read 16 kHz mono PCM16 WAV (same helper as Whisper test).
        let samples = try Self.loadWav16kMono(wavPath)
        // Pad to max_audio_clip_s only if audio is shorter — but
        // also cap at 30 sec to keep encoder forward time bounded.
        // 5 sec of audio is enough to see whether transcription works.
        let cap = min(samples.count, 30 * 16_000)
        let padded = Array(samples.prefix(cap))

        // Decoder priming: the full prompt format from
        // `modeling_cohere_asr.py.build_prompt(language='en',
        // punctuation=True)`:
        //   <|startofcontext|><|startoftranscript|><|emo:undefined|>
        //   <|en|><|en|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>
        //
        // Token IDs (looked up from tokenizer.json's added_tokens):
        let promptTokens: [Int] = [
            7,    // <|startofcontext|>
            4,    // <|startoftranscript|>
            16,   // <|emo:undefined|>
            62,   // <|en|>            (source_lang)
            62,   // <|en|>            (target_lang)
            5,    // <|pnc|>
            9,    // <|noitn|>
            11,   // <|notimestamp|>
            13,   // <|nodiarize|>
        ]
        let stderr = FileHandle.standardError
        func say(_ s: String) {
            stderr.write(("[step] " + s + "\n").data(using: .utf8)!)
        }
        say("audio: \(padded.count / 16_000) sec, \(padded.count) samples")
        say("prompt: \(promptTokens)")

        say("running mel + encoder + cross-attn prefill...")
        model.reset()
        let mel = AudioPreprocessor.logMelSpectrogram(
            samples: padded, config: .cohereTranscribe,
            filterbankOverride: model.preprocessorFb
        )
        let melBatched = mel.reshape([1, mel.shape[0], mel.shape[1]])
        let t0 = Date()
        model.setEncoderContext(mel: melBatched)
        say("  encoder + prefill done in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")

        say("greedy decode (max 30 tokens)...")
        let t1 = Date()
        var tokens: [Int] = []
        var nextTok = -1
        for tok in promptTokens {
            nextTok = try model.decoderStep(tokenId: tok)
        }
        for i in 0..<30 {
            if nextTok == config.eosTokenId {
                say("  eos at step \(i)")
                break
            }
            tokens.append(nextTok)
            say("  step \(i) → token \(nextTok)")
            nextTok = try model.decoderStep(tokenId: nextTok)
        }
        say("decode done in \(String(format: "%.1f", Date().timeIntervalSince(t1)))s")

        let text = try tokenizer.decode(
            tokens.map { UInt32($0) },
            skipSpecialTokens: true
        )
        print("[cohere-transcribe] '\(wavPath)' -> \"\(text)\"")
        XCTAssertGreaterThan(tokens.count, 0,
            "expected non-empty transcription, got '\(text)'")
    }

    // MARK: - WAV reader (same as WhisperTinyIntegrationTests)

    private static func loadWav16kMono(_ path: String) throws -> [Float] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard data.count >= 44 else {
            throw NSError(domain: "wav", code: 1)
        }
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
        guard dataStart >= 0 else {
            throw NSError(domain: "wav", code: 3)
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
