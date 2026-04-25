import XCTest
import Foundation
@testable import Transcribe

final class DecoderLMTests: XCTestCase {

    private func tmp(_ name: String) -> String {
        (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("ml_transcribe_lm_\(name)_\(UUID().uuidString).safetensors")
    }

    private func writeFixture(
        _ pairs: [(String, [Float], [Int])],
        to path: String
    ) throws {
        let tensors: [(String, Tensor)] = try pairs.map { name, data, shape in
            (name, try Tensor.from(data: data, shape: shape))
        }
        try SafetensorsWeights.save(tensors, to: path)
    }

    // MARK: - Weight builders (new "norm1.weight / norm1.bias / gate_proj" naming)

    /// Per-layer zero-projection weights for a given config. Emits only
    /// the keys that config actually reads:
    ///   - LayerNorm: norm{1,2}.bias is present
    ///   - RMSNorm:   norm{1,2}.bias is omitted
    ///   - useBias=true: all projection biases present
    ///   - useBias=false: projection biases omitted
    ///   - SwiGLU: ffn.gate_proj.weight present
    ///   - GELU:   gate_proj omitted
    private func zeroLayerWeights(
        prefix: String, config: DecoderLM.Config
    ) -> [(String, [Float], [Int])] {
        let D = config.modelDim
        let F = config.ffnDim
        // Under GQA, K/V project to a smaller [D, Dkv] than Q's [D, D].
        let Dkv = config.numKVHeads * config.headDim
        func zeros(_ n: Int) -> [Float] { Array(repeating: 0, count: n) }
        func ones(_ n: Int)  -> [Float] { Array(repeating: 1, count: n) }

        var out: [(String, [Float], [Int])] = [
            (prefix + "norm1.weight", ones(D), [D]),
        ]
        if config.normType == .layerNorm {
            out.append((prefix + "norm1.bias", zeros(D), [D]))
        }
        out.append(contentsOf: [
            (prefix + "attn.q_proj.weight", zeros(D * D),   [D, D]),
            (prefix + "attn.k_proj.weight", zeros(D * Dkv), [D, Dkv]),
            (prefix + "attn.v_proj.weight", zeros(D * Dkv), [D, Dkv]),
            (prefix + "attn.o_proj.weight", zeros(D * D),   [D, D]),
        ])
        if config.useBias {
            out.append(contentsOf: [
                (prefix + "attn.q_proj.bias", zeros(D),   [1, 1, D]),
                (prefix + "attn.k_proj.bias", zeros(Dkv), [1, 1, Dkv]),
                (prefix + "attn.v_proj.bias", zeros(Dkv), [1, 1, Dkv]),
                (prefix + "attn.o_proj.bias", zeros(D),   [1, 1, D]),
            ])
        }
        out.append((prefix + "norm2.weight", ones(D), [D]))
        if config.normType == .layerNorm {
            out.append((prefix + "norm2.bias", zeros(D), [D]))
        }
        if config.ffnType == .swiGLU {
            out.append((prefix + "ffn.gate_proj.weight", zeros(D * F), [D, F]))
        }
        out.append(contentsOf: [
            (prefix + "ffn.up_proj.weight",   zeros(D * F), [D, F]),
            (prefix + "ffn.down_proj.weight", zeros(F * D), [F, D]),
        ])
        if config.useBias {
            out.append(contentsOf: [
                (prefix + "ffn.up_proj.bias",   zeros(F), [1, 1, F]),
                (prefix + "ffn.down_proj.bias", zeros(D), [1, 1, D]),
            ])
        }
        return out
    }

    /// Identity LM: embedding = I, all N layers zero-projection, LM head = I.
    /// Works for any Config (GPT-style or LLaMA-style): with zero
    /// projections every sub-layer output is 0, residuals pass the
    /// one-hot through, and argmax lands on the input token.
    private func identityWeights(
        config: DecoderLM.Config
    ) -> [(String, [Float], [Int])] {
        let D = config.modelDim
        let V = config.vocabSize
        precondition(V == D, "identity test requires vocabSize == modelDim")

        func zeros(_ n: Int) -> [Float] { Array(repeating: 0, count: n) }
        func ones(_ n: Int)  -> [Float] { Array(repeating: 1, count: n) }
        func eye(_ n: Int) -> [Float] {
            var m = [Float](repeating: 0, count: n * n)
            for i in 0..<n { m[i * n + i] = 1 }
            return m
        }

        var fixtures: [(String, [Float], [Int])] = [
            ("embedding",         eye(D),   [V, D]),
            ("final_norm.weight", ones(D),  [D]),
        ]
        if config.normType == .layerNorm {
            fixtures.append(("final_norm.bias", zeros(D), [D]))
        }
        fixtures.append(("lm_head", eye(D), [D, V]))

        for i in 0..<config.numLayers {
            fixtures.append(contentsOf: zeroLayerWeights(
                prefix: "layers.\(i).", config: config
            ))
        }
        return fixtures
    }

    private func randomLayerWeights(
        prefix: String, config: DecoderLM.Config, rng: inout SeededRNG
    ) -> [(String, [Float], [Int])] {
        let D = config.modelDim
        let F = config.ffnDim
        let Dkv = config.numKVHeads * config.headDim
        func randn(_ n: Int, scale: Float, rng: inout SeededRNG) -> [Float] {
            (0..<n).map { _ in Float.random(in: -scale...scale, using: &rng) }
        }
        let zerosD   = Array(repeating: Float(0), count: D)
        let zerosDkv = Array(repeating: Float(0), count: Dkv)
        let zerosF   = Array(repeating: Float(0), count: F)
        let onesD    = Array(repeating: Float(1), count: D)

        var out: [(String, [Float], [Int])] = [
            (prefix + "norm1.weight", onesD, [D]),
        ]
        if config.normType == .layerNorm {
            out.append((prefix + "norm1.bias", zerosD, [D]))
        }
        out.append(contentsOf: [
            (prefix + "attn.q_proj.weight", randn(D * D,   scale: 0.1, rng: &rng), [D, D]),
            (prefix + "attn.k_proj.weight", randn(D * Dkv, scale: 0.1, rng: &rng), [D, Dkv]),
            (prefix + "attn.v_proj.weight", randn(D * Dkv, scale: 0.1, rng: &rng), [D, Dkv]),
            (prefix + "attn.o_proj.weight", randn(D * D,   scale: 0.1, rng: &rng), [D, D]),
        ])
        if config.useBias {
            out.append(contentsOf: [
                (prefix + "attn.q_proj.bias", zerosD,   [1, 1, D]),
                (prefix + "attn.k_proj.bias", zerosDkv, [1, 1, Dkv]),
                (prefix + "attn.v_proj.bias", zerosDkv, [1, 1, Dkv]),
                (prefix + "attn.o_proj.bias", zerosD,   [1, 1, D]),
            ])
        }
        out.append((prefix + "norm2.weight", onesD, [D]))
        if config.normType == .layerNorm {
            out.append((prefix + "norm2.bias", zerosD, [D]))
        }
        if config.ffnType == .swiGLU {
            out.append((prefix + "ffn.gate_proj.weight",
                        randn(D * F, scale: 0.1, rng: &rng), [D, F]))
        }
        out.append(contentsOf: [
            (prefix + "ffn.up_proj.weight",
             randn(D * F, scale: 0.1, rng: &rng), [D, F]),
            (prefix + "ffn.down_proj.weight",
             randn(F * D, scale: 0.1, rng: &rng), [F, D]),
        ])
        if config.useBias {
            out.append(contentsOf: [
                (prefix + "ffn.up_proj.bias",   zerosF, [1, 1, F]),
                (prefix + "ffn.down_proj.bias", zerosD, [1, 1, D]),
            ])
        }
        return out
    }

    private func randomWeights(
        config: DecoderLM.Config, seed: UInt64
    ) -> [(String, [Float], [Int])] {
        let D = config.modelDim
        let V = config.vocabSize
        var rng = SeededRNG(seed: seed)
        func randn(_ n: Int, scale: Float, rng: inout SeededRNG) -> [Float] {
            (0..<n).map { _ in Float.random(in: -scale...scale, using: &rng) }
        }
        var fixtures: [(String, [Float], [Int])] = [
            ("embedding",         randn(V * D, scale: 0.5, rng: &rng), [V, D]),
            ("final_norm.weight", Array(repeating: Float(1), count: D), [D]),
        ]
        if config.normType == .layerNorm {
            fixtures.append(
                ("final_norm.bias", Array(repeating: Float(0), count: D), [D])
            )
        }
        fixtures.append(("lm_head", randn(D * V, scale: 0.1, rng: &rng), [D, V]))
        for i in 0..<config.numLayers {
            fixtures.append(contentsOf: randomLayerWeights(
                prefix: "layers.\(i).", config: config, rng: &rng
            ))
        }
        return fixtures
    }

    // =========================================================================
    // GPT-style tests (LayerNorm + biases + GELU)
    // =========================================================================

    private func gptConfig(
        vocab: Int, D: Int, H: Int, F: Int, layers: Int, maxSeq: Int = 16
    ) -> DecoderLM.Config {
        .gptStyle(
            vocabSize: vocab, modelDim: D, numHeads: H, ffnDim: F,
            numLayers: layers, maxSeqLen: maxSeq
        )
    }

    func testGPTIdentityModelEchoesEveryToken() throws {
        let path = tmp("gpt_identity_1")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let config = gptConfig(vocab: 8, D: 8, H: 2, F: 16, layers: 1)
        try writeFixture(identityWeights(config: config), to: path)

        let lm = try DecoderLM(
            weights: try SafetensorsWeights(path: path), config: config
        )
        for t in 0..<config.vocabSize {
            lm.reset()
            XCTAssertEqual(try lm.step(tokenId: t), t)
        }
    }

    func testGPTThreeLayerIdentityEchoes() throws {
        let path = tmp("gpt_identity_3")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let config = gptConfig(vocab: 8, D: 8, H: 2, F: 16, layers: 3)
        try writeFixture(identityWeights(config: config), to: path)

        let lm = try DecoderLM(
            weights: try SafetensorsWeights(path: path), config: config
        )
        for t in 0..<config.vocabSize {
            lm.reset()
            XCTAssertEqual(try lm.step(tokenId: t), t)
        }
    }

    func testGPTCacheLengthTracks() throws {
        let path = tmp("gpt_cache_len")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let config = gptConfig(vocab: 8, D: 8, H: 2, F: 16, layers: 1)
        try writeFixture(identityWeights(config: config), to: path)

        let lm = try DecoderLM(
            weights: try SafetensorsWeights(path: path), config: config
        )
        XCTAssertEqual(lm.currentPosition, 0)
        _ = try lm.step(tokenId: 3)
        _ = try lm.step(tokenId: 5)
        XCTAssertEqual(lm.currentPosition, 2)
        lm.reset()
        XCTAssertEqual(lm.currentPosition, 0)
    }

    func testGPTGenerationIsDeterministic() throws {
        let path = tmp("gpt_det")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let config = gptConfig(vocab: 32, D: 16, H: 4, F: 32, layers: 2)
        try writeFixture(randomWeights(config: config, seed: 123), to: path)

        let lm = try DecoderLM(
            weights: try SafetensorsWeights(path: path), config: config
        )
        let prompt = [1, 2, 3, 4]
        let first = try lm.generate(prompt: prompt, maxNewTokens: 8)
        for _ in 0..<3 {
            XCTAssertEqual(
                try lm.generate(prompt: prompt, maxNewTokens: 8), first
            )
        }
        for t in first {
            XCTAssertTrue((0..<config.vocabSize).contains(t))
        }
    }

    func testGPTMissingWeightThrows() throws {
        let path = tmp("gpt_missing")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let config = gptConfig(vocab: 8, D: 8, H: 2, F: 16, layers: 2)
        var fixtures = identityWeights(config: config)
        fixtures.removeAll { $0.0 == "layers.1.attn.q_proj.weight" }
        try writeFixture(fixtures, to: path)

        let weights = try SafetensorsWeights(path: path)
        XCTAssertThrowsError(try DecoderLM(weights: weights, config: config)) {
            guard let err = $0 as? DecoderLM.LoadError else {
                return XCTFail("expected LoadError, got \($0)")
            }
            if case .missing(let name) = err {
                XCTAssertEqual(name, "layers.1.attn.q_proj.weight")
            } else {
                XCTFail("expected .missing, got \(err)")
            }
        }
    }

    // =========================================================================
    // LLaMA-style tests (RMSNorm + no biases + SwiGLU)
    // =========================================================================

    private func llamaConfig(
        vocab: Int, D: Int, H: Int, F: Int, layers: Int, maxSeq: Int = 16
    ) -> DecoderLM.Config {
        .llamaStyle(
            vocabSize: vocab, modelDim: D, numHeads: H, ffnDim: F,
            numLayers: layers, maxSeqLen: maxSeq
        )
    }

    /// The analogous identity test for the LLaMA variant.
    /// With zero gate_proj, `silu(0) = 0`, so SwiGLU's gate·up product
    /// is 0. With zero up_proj and zero down_proj, the FFN output is
    /// 0. With zero attention projections (and no biases), attention
    /// output is 0 too. Residuals carry the input through unchanged.
    /// Final RMSNorm of one-hot still has a unique max at the same
    /// index, and identity LM head argmax echoes the input.
    func testLLaMAIdentityModelEchoesEveryToken() throws {
        let path = tmp("llama_identity_1")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let config = llamaConfig(vocab: 8, D: 8, H: 2, F: 16, layers: 1)
        try writeFixture(identityWeights(config: config), to: path)

        let lm = try DecoderLM(
            weights: try SafetensorsWeights(path: path), config: config
        )
        for t in 0..<config.vocabSize {
            lm.reset()
            XCTAssertEqual(
                try lm.step(tokenId: t), t,
                "LLaMA-style identity should echo token \(t)")
        }
    }

    func testLLaMAThreeLayerIdentityEchoes() throws {
        let path = tmp("llama_identity_3")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let config = llamaConfig(vocab: 8, D: 8, H: 2, F: 16, layers: 3)
        try writeFixture(identityWeights(config: config), to: path)

        let lm = try DecoderLM(
            weights: try SafetensorsWeights(path: path), config: config
        )
        for t in 0..<config.vocabSize {
            lm.reset()
            XCTAssertEqual(try lm.step(tokenId: t), t)
        }
    }

    func testLLaMARandomGenerationRuns() throws {
        let path = tmp("llama_random")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let config = llamaConfig(vocab: 32, D: 16, H: 4, F: 32, layers: 4)
        try writeFixture(randomWeights(config: config, seed: 42), to: path)

        let lm = try DecoderLM(
            weights: try SafetensorsWeights(path: path), config: config
        )
        let prompt = [1, 2, 3]
        let gen = try lm.generate(prompt: prompt, maxNewTokens: 10)
        XCTAssertEqual(gen.count, 10)
        XCTAssertEqual(lm.currentPosition, prompt.count + 10)
        for t in gen {
            XCTAssertTrue((0..<config.vocabSize).contains(t))
        }
        // Determinism.
        XCTAssertEqual(
            try lm.generate(prompt: prompt, maxNewTokens: 10), gen
        )
    }

    /// The two variants should produce different outputs on the same
    /// seed: RMSNorm ≠ LayerNorm, GELU ≠ SwiGLU, and LLaMA uses no
    /// biases while GPT does. Same model dims, same prompt, same seed
    /// → meaningfully different generated sequences.
    func testLLaMADiffersFromGPTStyle() throws {
        let gpt   = gptConfig(vocab: 32, D: 16, H: 4, F: 32, layers: 2)
        let llama = llamaConfig(vocab: 32, D: 16, H: 4, F: 32, layers: 2)

        let gptPath   = tmp("gpt_variant")
        let llamaPath = tmp("llama_variant")
        defer {
            try? FileManager.default.removeItem(atPath: gptPath)
            try? FileManager.default.removeItem(atPath: llamaPath)
        }
        try writeFixture(randomWeights(config: gpt,   seed: 7), to: gptPath)
        try writeFixture(randomWeights(config: llama, seed: 7), to: llamaPath)

        let gptLM = try DecoderLM(
            weights: try SafetensorsWeights(path: gptPath), config: gpt
        )
        let llamaLM = try DecoderLM(
            weights: try SafetensorsWeights(path: llamaPath), config: llama
        )

        let prompt = [5, 10, 15]
        let gptGen   = try gptLM.generate(prompt: prompt, maxNewTokens: 6)
        let llamaGen = try llamaLM.generate(prompt: prompt, maxNewTokens: 6)
        XCTAssertNotEqual(gptGen, llamaGen)
    }

    func testLLaMALookupWithoutBiasKeyWorks() throws {
        // LLaMA fixtures intentionally omit `attn.q_proj.bias` etc.
        // The init path must NOT complain about missing biases (that
        // would be a regression).
        let path = tmp("llama_nobias")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let config = llamaConfig(vocab: 8, D: 8, H: 2, F: 16, layers: 1)
        let fixtures = identityWeights(config: config)
        // Double-check: no bias key appears in the fixture set.
        XCTAssertFalse(fixtures.contains { $0.0.contains(".bias") },
            "LLaMA fixture set must not contain any .bias keys")

        try writeFixture(fixtures, to: path)
        _ = try DecoderLM(
            weights: try SafetensorsWeights(path: path), config: config
        )
    }

    /// Mixing the wrong fixtures with the wrong variant surfaces a
    /// load error (expected: SwiGLU gate_proj is missing in GPT-style
    /// fixtures, so LLaMA config against GPT-style weights fails).
    func testLLaMAConfigAgainstGPTFixturesFails() throws {
        let path = tmp("llama_config_gpt_fixtures")
        defer { try? FileManager.default.removeItem(atPath: path) }

        // Build GPT-style fixtures (no gate_proj, with biases, with beta).
        let gpt   = gptConfig(vocab: 8, D: 8, H: 2, F: 16, layers: 1)
        let llama = llamaConfig(vocab: 8, D: 8, H: 2, F: 16, layers: 1)
        try writeFixture(identityWeights(config: gpt), to: path)

        let weights = try SafetensorsWeights(path: path)
        XCTAssertThrowsError(try DecoderLM(weights: weights, config: llama)) {
            guard let err = $0 as? DecoderLM.LoadError else {
                return XCTFail("expected LoadError, got \($0)")
            }
            if case .missing(let name) = err {
                XCTAssertTrue(name.contains("gate_proj"),
                    "expected a gate_proj miss, got '\(name)'")
            } else {
                XCTFail("expected .missing, got \(err)")
            }
        }
    }

    // =========================================================================
    // Tokenizer-driven (GPT-style, identity → echo)
    // =========================================================================

    // MARK: - Tied embeddings

    /// When `lm_head` is absent from the weights, DecoderLM must fall
    /// back to using `embedding.transpose()` (the GPT-2 / LLaMA tied
    /// embeddings convention). Test: strip the explicit lm_head from
    /// an identity fixture and verify the model still echoes — because
    /// `eye(D).transpose() == eye(D)`.
    func testTiedEmbeddingsFallbackEchoes() throws {
        let path = tmp("tied_identity")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let config = gptConfig(vocab: 8, D: 8, H: 2, F: 16, layers: 1)
        var fixtures = identityWeights(config: config)
        // Remove the explicit lm_head; the tied-embedding fallback
        // should kick in and use `embedding.transpose() == eye(D)`.
        fixtures.removeAll { $0.0 == "lm_head" }
        try writeFixture(fixtures, to: path)

        let lm = try DecoderLM(
            weights: try SafetensorsWeights(path: path), config: config
        )
        for t in 0..<config.vocabSize {
            lm.reset()
            XCTAssertEqual(
                try lm.step(tokenId: t), t,
                "tied-embedding identity LM should echo \(t)")
        }
    }

    /// Tied-embedding generation must produce IDENTICAL output to an
    /// explicit-lm_head model whose lm_head is literally
    /// embedding.transpose(). Same random weights, same prompt:
    /// argmax choices must match step-for-step.
    func testTiedEmbeddingsEqualsExplicitTranspose() throws {
        let tiedPath     = tmp("tied_random")
        let explicitPath = tmp("explicit_random")
        defer {
            try? FileManager.default.removeItem(atPath: tiedPath)
            try? FileManager.default.removeItem(atPath: explicitPath)
        }

        let config = gptConfig(vocab: 16, D: 8, H: 2, F: 16, layers: 2)
        var fixtures = randomWeights(config: config, seed: 42)

        // Find the embedding entry to build an explicit lm_head from its
        // transpose.
        let embed = fixtures.first { $0.0 == "embedding" }!
        let embedData = embed.1
        let V = config.vocabSize
        let D = config.modelDim

        // Transpose the embedding [V, D] -> [D, V] in plain Swift.
        var lmHeadData = [Float](repeating: 0, count: D * V)
        for i in 0..<V {
            for j in 0..<D {
                lmHeadData[j * V + i] = embedData[i * D + j]
            }
        }

        // Write the TIED variant: no lm_head entry.
        var tiedFixtures = fixtures
        tiedFixtures.removeAll { $0.0 == "lm_head" }
        try writeFixture(tiedFixtures, to: tiedPath)

        // Write the EXPLICIT variant: lm_head is embedding-transposed.
        fixtures.removeAll { $0.0 == "lm_head" }
        fixtures.append(("lm_head", lmHeadData, [D, V]))
        try writeFixture(fixtures, to: explicitPath)

        let tiedLM = try DecoderLM(
            weights: try SafetensorsWeights(path: tiedPath), config: config
        )
        let explicitLM = try DecoderLM(
            weights: try SafetensorsWeights(path: explicitPath), config: config
        )

        let prompt = [1, 2, 3]
        let tiedGen     = try tiedLM.generate(prompt: prompt, maxNewTokens: 8)
        let explicitGen = try explicitLM.generate(prompt: prompt, maxNewTokens: 8)
        XCTAssertEqual(tiedGen, explicitGen,
            "tied fallback should match an explicit transposed lm_head")
    }

    /// Tied fallback still reports shape errors when `embedding` itself
    /// is missing — we need *something* to build lm_head from.
    func testMissingBothEmbeddingAndLMHeadThrows() throws {
        let path = tmp("missing_both")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let config = gptConfig(vocab: 8, D: 8, H: 2, F: 16, layers: 1)
        var fixtures = identityWeights(config: config)
        fixtures.removeAll { $0.0 == "embedding" || $0.0 == "lm_head" }
        try writeFixture(fixtures, to: path)

        let weights = try SafetensorsWeights(path: path)
        XCTAssertThrowsError(try DecoderLM(weights: weights, config: config)) {
            guard let err = $0 as? DecoderLM.LoadError else {
                return XCTFail("expected LoadError, got \($0)")
            }
            if case .missing(let name) = err {
                XCTAssertEqual(name, "embedding")
            } else {
                XCTFail("expected .missing(embedding), got \(err)")
            }
        }
    }

    func testTokenizerDrivenGeneration() throws {
        let tokJSON = """
        {
            "version": "1.0",
            "pre_tokenizer": {"type": "Whitespace"},
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "a": 0, "b": 1, "c": 2, "d": 3,
                    "e": 4, "f": 5, "g": 6, "[UNK]": 7
                },
                "unk_token": "[UNK]"
            }
        }
        """
        let tokPath = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("ml_transcribe_lm_tok_\(UUID().uuidString).json")
        try tokJSON.write(toFile: tokPath, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(atPath: tokPath) }

        let wPath = tmp("tok_driven")
        defer { try? FileManager.default.removeItem(atPath: wPath) }

        let config = gptConfig(vocab: 8, D: 8, H: 2, F: 16, layers: 1)
        try writeFixture(identityWeights(config: config), to: wPath)

        let tokenizer = try Tokenizer(path: tokPath)
        let lm = try DecoderLM(
            weights: try SafetensorsWeights(path: wPath), config: config
        )
        let ids = try tokenizer.encode("a b c", addSpecialTokens: false)
        XCTAssertEqual(ids, [0, 1, 2])
        XCTAssertEqual(
            try lm.generate(prompt: ids.map { Int($0) }, maxNewTokens: 4),
            [2, 2, 2, 2]
        )
    }
}
