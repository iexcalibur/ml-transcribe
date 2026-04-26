import Foundation
import SwiftUI
import Transcribe

/// Loads TinyLlama-1.1B from the app bundle and exposes a single
/// `generate(prompt:maxNewTokens:)` async method that runs the full
/// stack (tokenize → embed → 22 cached transformer layers → decode).
///
/// Both `model.safetensors` and `tokenizer.json` are expected as
/// bundle resources. They are too large to commit to git — see
/// `apps/TinyLlamaDemo/README.md` for how to add them.
@MainActor
final class TinyLlamaGenerator: ObservableObject {
    @Published private(set) var status: String = "not loaded"
    @Published private(set) var isReady: Bool = false

    private var lm: DecoderLM?
    private var tokenizer: Tokenizer?
    private var loadTask: Task<Void, Error>?

    /// Idempotent: first call kicks off the load; subsequent calls
    /// are no-ops while the load is in flight.
    func loadIfNeeded() async {
        if loadTask == nil {
            loadTask = Task.detached(priority: .userInitiated) {
                try await self.load()
            }
        }
        do { try await loadTask?.value } catch {
            await MainActor.run { self.status = "load failed: \(error)" }
        }
    }

    nonisolated private func load() async throws {
        await MainActor.run { self.status = "locating model files…" }

        guard let modelURL = Bundle.main.url(
                forResource: "model", withExtension: "safetensors"),
              let tokenizerURL = Bundle.main.url(
                forResource: "tokenizer", withExtension: "json")
        else {
            throw GeneratorError.missingResources
        }

        await MainActor.run {
            self.status = "loading 2.2 GB safetensors as F16 (one-time)…"
        }

        // F16 storage halves the resident memory of weights — roughly
        // 2.2 GB instead of 4.4 GB. Mandatory for a 1.1B model on
        // iPhone-class memory budgets.
        let raw = try SafetensorsWeights(
            path: modelURL.path, keepF16: true
        )
        let mapped = WeightMap(
            source: raw,
            renames: WeightMap.huggingFaceLlamaRenames(numLayers: 22),
            transpose: WeightMap.huggingFaceLlamaTransposeSet(numLayers: 22)
        )

        // TinyLlama-1.1B-Chat config (matches HF config.json).
        let config = DecoderLM.Config.llamaStyle(
            vocabSize: 32000,
            modelDim: 2048,
            numHeads: 32,
            ffnDim: 5632,
            numLayers: 22,
            maxSeqLen: 256,
            numKVHeads: 4
        )

        let builtLm = try DecoderLM(weights: mapped, config: config)
        let builtTokenizer = try Tokenizer(path: tokenizerURL.path)

        await MainActor.run {
            self.lm = builtLm
            self.tokenizer = builtTokenizer
            self.status = "ready (TinyLlama-1.1B, F16 weights, GQA 32→4)"
            self.isReady = true
        }
    }

    /// Run the full pipeline: tokenize → DecoderLM.generate → decode.
    /// Heavy work; run off the main actor (the `Task.detached` below
    /// does this implicitly because DecoderLM/Tokenizer are
    /// thread-safe value-like wrappers over the engine FFI).
    func generate(prompt: String, maxNewTokens: Int) async throws -> String {
        guard let lm = lm, let tokenizer = tokenizer else {
            throw GeneratorError.notLoaded
        }
        return try await Task.detached(priority: .userInitiated) {
            let promptIds = try tokenizer.encode(prompt, addSpecialTokens: true)
            let generated = try lm.generate(
                prompt: promptIds.map { Int($0) },
                maxNewTokens: maxNewTokens
            )
            let full = try tokenizer.decode(
                (promptIds.map { Int($0) } + generated).map { UInt32($0) },
                skipSpecialTokens: true
            )
            return full
        }.value
    }

    enum GeneratorError: Error, CustomStringConvertible {
        case missingResources
        case notLoaded

        var description: String {
            switch self {
            case .missingResources:
                return "model.safetensors or tokenizer.json missing from bundle"
            case .notLoaded:
                return "model not yet loaded"
            }
        }
    }
}
