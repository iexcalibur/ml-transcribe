import XCTest
import Foundation
@testable import Transcribe

/// Opt-in integration tests that load and run a real pretrained
/// LLaMA-family model (TinyLlama-1.1B by default).
///
/// These are gated by the `TINYLLAMA_DIR` environment variable. The
/// regular CI suite leaves it unset and these tests skip; running:
///
///   TINYLLAMA_DIR=/tmp/ml-inspect/tinyllama swift test ...
///
/// activates them. Both `model.safetensors` and `tokenizer.json` must
/// be present in that directory.
///
/// Why these live as XCTest cases (not just CLI invocations of
/// inspect-model): we want to run them on iOS Simulator too, where
/// CLI executables don't run but the Swift Package's test target does.
final class TinyLlamaIntegrationTests: XCTestCase {

    private struct ModelDir {
        let safetensors: String
        let tokenizer: String
    }

    /// Returns the configured directory or skips the test.
    private func tinyLlamaDir() throws -> ModelDir {
        guard let dir = ProcessInfo.processInfo.environment["TINYLLAMA_DIR"] else {
            throw XCTSkip("set TINYLLAMA_DIR=/path/to/tinyllama to run this test")
        }
        let model = (dir as NSString).appendingPathComponent("model.safetensors")
        let tok   = (dir as NSString).appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: model),
              FileManager.default.fileExists(atPath: tok) else {
            throw XCTSkip("TINYLLAMA_DIR=\(dir) is missing model.safetensors or tokenizer.json")
        }
        return ModelDir(safetensors: model, tokenizer: tok)
    }

    /// Build a TinyLlama-1.1B-Chat config matching the official
    /// HuggingFace `config.json` (hidden_size=2048, num_attention_heads=32,
    /// num_key_value_heads=4 (GQA), intermediate_size=5632,
    /// num_hidden_layers=22, vocab_size=32000).
    private func tinyLlamaConfig() -> DecoderLM.Config {
        .llamaStyle(
            vocabSize: 32000,
            modelDim: 2048,
            numHeads: 32,
            ffnDim: 5632,
            numLayers: 22,
            maxSeqLen: 256,
            numKVHeads: 4
        )
    }

    /// End-to-end test: load the file, build a `DecoderLM` from it
    /// via `WeightMap` (HF naming + transposes), run a forward step.
    /// Memory-aware: uses `keepF16: true` to halve resident weights.
    func testTinyLlamaLoadAndForwardStep() throws {
        let dir = try tinyLlamaDir()

        let raw = try SafetensorsWeights(path: dir.safetensors, keepF16: true)
        XCTAssertEqual(raw.count, 201, "TinyLlama has 201 tensors")

        let mapped = WeightMap(
            source: raw,
            renames: WeightMap.huggingFaceLlamaRenames(numLayers: 22),
            transpose: WeightMap.huggingFaceLlamaTransposeSet(numLayers: 22)
        )
        let lm = try DecoderLM(weights: mapped, config: tinyLlamaConfig())

        // One forward step from the BOS token (id 1 for LLaMA family).
        let predicted = try lm.step(tokenId: 1)
        XCTAssertGreaterThanOrEqual(predicted, 0)
        XCTAssertLessThan(predicted, 32000,
            "predicted token must lie in the vocab range")
    }

    /// Generate real text and verify the output looks reasonable.
    /// Asserts the model produces valid token ids and that "Paris"
    /// appears for the canonical France-capital prompt.
    func testTinyLlamaGeneratesParisForFranceCapital() throws {
        let dir = try tinyLlamaDir()

        let raw = try SafetensorsWeights(path: dir.safetensors, keepF16: true)
        let mapped = WeightMap(
            source: raw,
            renames: WeightMap.huggingFaceLlamaRenames(numLayers: 22),
            transpose: WeightMap.huggingFaceLlamaTransposeSet(numLayers: 22)
        )
        let lm = try DecoderLM(weights: mapped, config: tinyLlamaConfig())
        let tokenizer = try Tokenizer(path: dir.tokenizer)

        let prompt = "The capital of France is"
        let promptIds = try tokenizer.encode(prompt, addSpecialTokens: true)
        let generated = try lm.generate(
            prompt: promptIds.map { Int($0) },
            maxNewTokens: 8
        )
        let decoded = try tokenizer.decode(
            generated.map { UInt32($0) },
            skipSpecialTokens: true
        )
        // Trained models should latch onto "Paris" almost always for
        // this prompt. Tolerate any whitespace / casing.
        XCTAssertTrue(
            decoded.lowercased().contains("paris"),
            "Expected continuation to contain 'paris', got '\(decoded)'"
        )
    }
}
