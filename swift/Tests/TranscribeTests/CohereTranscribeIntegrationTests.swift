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
}
