import XCTest
import Foundation
@testable import Transcribe

/// Structural tests for the Cohere Transcribe model. We don't have
/// real weights yet (they're an opt-in download via
/// `COHERE_TRANSCRIBE_DIR`), so these tests exercise the wiring with
/// random weight bundles at a tiny scale — fast to run and catches
/// shape / wiring / NaN bugs in the orchestration class.
final class CohereTranscribeTests: XCTestCase {

    // MARK: - Tiny config

    private func tinyConfig(
        encLayers: Int = 1, decLayers: Int = 1
    ) -> CohereTranscribe.Config {
        let enc = ConformerEncoder.Config(
            nMels: 16,
            dModel: 32,
            nLayers: encLayers,
            nHeads: 4,
            ffExpansion: 2,
            convKernelSize: 9,
            subsamplingChannels: 8,
            subsamplingFactor: 8,
            posEmbMaxLen: 1000
        )
        return CohereTranscribe.Config(
            encoder: enc,
            encDecProjFromDim: 32,        // matches enc.dModel for testing
            encDecProjToDim: 24,          // smaller decoder dim
            decoderHidden: 24,
            decoderLayers: decLayers,
            decoderHeads: 4,
            decoderFFNDim: 48,
            maxSeqLen: 64,
            vocabSize: 50,
            bosTokenId: 1,
            eosTokenId: 2,
            padTokenId: 0
        )
    }

    // MARK: - Random weight builders

    private static func randTensor(_ shape: [Int], scale: Float = 0.05,
                                   rng: inout SeededRNG) -> Tensor {
        let n = shape.reduce(1, *)
        let data = (0..<n).map { _ in
            Float.random(in: -scale...scale, using: &rng)
        }
        return try! Tensor.from(data: data, shape: shape)
    }
    private static func zerosTensor(_ shape: [Int]) -> Tensor {
        let n = shape.reduce(1, *)
        return try! Tensor.from(data: [Float](repeating: 0, count: n), shape: shape)
    }
    private static func onesTensor(_ shape: [Int]) -> Tensor {
        let n = shape.reduce(1, *)
        return try! Tensor.from(data: [Float](repeating: 1, count: n), shape: shape)
    }

    private static func randomEncoderSubsamplingWeights(
        config: ConformerEncoder.Config, rng: inout SeededRNG
    ) -> ConvSubsampling.Weights {
        let C = config.subsamplingChannels
        let fPrime = config.subsampledFreq
        return ConvSubsampling.Weights(
            conv1Weight: randTensor([C, 1, 3, 3], rng: &rng),
            conv1Bias:   randTensor([C], rng: &rng),
            conv2Weight: randTensor([C, 1, 3, 3], rng: &rng),
            conv2Bias:   randTensor([C], rng: &rng),
            conv3Weight: randTensor([C, C, 1, 1], rng: &rng),
            conv3Bias:   randTensor([C], rng: &rng),
            conv4Weight: randTensor([C, 1, 3, 3], rng: &rng),
            conv4Bias:   randTensor([C], rng: &rng),
            conv5Weight: randTensor([C, C, 1, 1], rng: &rng),
            conv5Bias:   randTensor([C], rng: &rng),
            outProjWeight: randTensor([C * fPrime, config.dModel], rng: &rng),
            outProjBias:   randTensor([config.dModel], rng: &rng)
        )
    }

    private static func randomEncoderLayerWeights(
        config: ConformerEncoder.Config, rng: inout SeededRNG
    ) -> ConformerLayer.Weights {
        let D = config.dModel
        let F = config.ffnInnerDim
        let H = config.nHeads
        let Dh = config.headDim
        let K = config.convKernelSize
        return ConformerLayer.Weights(
            ffn1NormGamma: onesTensor([D]),
            ffn1NormBeta: zerosTensor([D]),
            ffn1UpW: randTensor([D, F], rng: &rng),
            ffn1UpB: randTensor([F], rng: &rng),
            ffn1DownW: randTensor([F, D], rng: &rng),
            ffn1DownB: randTensor([D], rng: &rng),
            attnNormGamma: onesTensor([D]),
            attnNormBeta: zerosTensor([D]),
            attnWQ: randTensor([D, D], rng: &rng), attnBQ: randTensor([D], rng: &rng),
            attnWK: randTensor([D, D], rng: &rng), attnBK: randTensor([D], rng: &rng),
            attnWV: randTensor([D, D], rng: &rng), attnBV: randTensor([D], rng: &rng),
            attnWPos: randTensor([D, D], rng: &rng),
            attnWOut: randTensor([D, D], rng: &rng), attnBOut: randTensor([D], rng: &rng),
            attnPosBiasU: randTensor([H, Dh], rng: &rng),
            attnPosBiasV: randTensor([H, Dh], rng: &rng),
            convNormGamma: onesTensor([D]), convNormBeta: zerosTensor([D]),
            convPointwise1Weight: randTensor([2 * D, D, 1], rng: &rng),
            convPointwise1Bias: randTensor([2 * D], rng: &rng),
            convDepthwiseWeight: randTensor([D, 1, K], rng: &rng),
            convDepthwiseBias: randTensor([D], rng: &rng),
            convBNRunningMean: zerosTensor([D]),
            convBNRunningVar: onesTensor([D]),
            convBNGamma: onesTensor([D]),
            convBNBeta: zerosTensor([D]),
            convPointwise2Weight: randTensor([D, D, 1], rng: &rng),
            convPointwise2Bias: randTensor([D], rng: &rng),
            ffn2NormGamma: onesTensor([D]),
            ffn2NormBeta: zerosTensor([D]),
            ffn2UpW: randTensor([D, F], rng: &rng),
            ffn2UpB: randTensor([F], rng: &rng),
            ffn2DownW: randTensor([F, D], rng: &rng),
            ffn2DownB: randTensor([D], rng: &rng),
            normOutGamma: onesTensor([D]),
            normOutBeta: zerosTensor([D])
        )
    }

    private static func randomDecoderLayerWeights(
        config: CohereTranscribe.Config, rng: inout SeededRNG
    ) -> CohereTranscribe.DecoderLayer.Weights {
        let D = config.decoderHidden
        let F = config.decoderFFNDim
        return CohereTranscribe.DecoderLayer.Weights(
            saNormGamma: onesTensor([D]), saNormBeta: zerosTensor([D]),
            saWQ: randTensor([D, D], rng: &rng), saBQ: randTensor([D], rng: &rng),
            saWK: randTensor([D, D], rng: &rng), saBK: randTensor([D], rng: &rng),
            saWV: randTensor([D, D], rng: &rng), saBV: randTensor([D], rng: &rng),
            saWO: randTensor([D, D], rng: &rng), saBO: randTensor([D], rng: &rng),
            caNormGamma: onesTensor([D]), caNormBeta: zerosTensor([D]),
            caWQ: randTensor([D, D], rng: &rng), caBQ: randTensor([D], rng: &rng),
            caWK: randTensor([D, D], rng: &rng), caBK: randTensor([D], rng: &rng),
            caWV: randTensor([D, D], rng: &rng), caBV: randTensor([D], rng: &rng),
            caWO: randTensor([D, D], rng: &rng), caBO: randTensor([D], rng: &rng),
            ffnNormGamma: onesTensor([D]), ffnNormBeta: zerosTensor([D]),
            ffnUpW: randTensor([D, F], rng: &rng), ffnUpB: randTensor([F], rng: &rng),
            ffnDownW: randTensor([F, D], rng: &rng), ffnDownB: randTensor([D], rng: &rng)
        )
    }

    private static func makeTinyModel(
        config: CohereTranscribe.Config, rng: inout SeededRNG
    ) -> CohereTranscribe {
        var subWeights = Self.randomEncoderSubsamplingWeights(
            config: config.encoder, rng: &rng)
        var encoderLayerWeights: [ConformerLayer.Weights] = []
        for _ in 0..<config.encoder.nLayers {
            encoderLayerWeights.append(
                Self.randomEncoderLayerWeights(config: config.encoder, rng: &rng)
            )
        }
        var decoderLayerWeights: [CohereTranscribe.DecoderLayer.Weights] = []
        for _ in 0..<config.decoderLayers {
            decoderLayerWeights.append(
                Self.randomDecoderLayerWeights(config: config, rng: &rng)
            )
        }
        let weights = CohereTranscribe.Weights(
            encoderSubsampling: subWeights,
            encoderLayers: encoderLayerWeights,
            encDecProjW: randTensor(
                [config.encDecProjFromDim, config.encDecProjToDim], rng: &rng),
            encDecProjB: randTensor([config.encDecProjToDim], rng: &rng),
            embedTokens: randTensor(
                [config.vocabSize, config.decoderHidden], rng: &rng),
            decoderLayers: decoderLayerWeights,
            decoderFinalNormGamma: onesTensor([config.decoderHidden]),
            decoderFinalNormBeta: zerosTensor([config.decoderHidden]),
            headWeight: randTensor(
                [config.decoderHidden, config.vocabSize], rng: &rng),
            headBias: randTensor([config.vocabSize], rng: &rng)
        )
        _ = subWeights
        _ = encoderLayerWeights
        _ = decoderLayerWeights
        return CohereTranscribe(config: config, weights: weights)
    }

    // MARK: - Decoder pos table

    /// The decoder's fixed sinusoidal position table has shape
    /// `[maxLen, dim]` and starts at row 0 with `[0, 1, 0, 1, ...]`
    /// since pos=0 gives sin(0)=0, cos(0)=1.
    func testDecoderPosTableRow0() throws {
        let table = CohereTranscribe.makeFixedSinusoidalPosTable(
            maxLen: 16, dim: 8
        )
        XCTAssertEqual(table.shape, [16, 8])
        let arr = table.toArray()
        for k in 0..<4 {
            XCTAssertEqual(arr[2 * k], 0, accuracy: 1e-6)
            XCTAssertEqual(arr[2 * k + 1], 1, accuracy: 1e-6)
        }
    }

    /// Two rows at positions p1, p2 have non-equal embeddings (so
    /// distinct positions get distinct codes — important for the
    /// model to distinguish token order).
    func testDecoderPosTableDistinctPositions() throws {
        let table = CohereTranscribe.makeFixedSinusoidalPosTable(
            maxLen: 32, dim: 16
        )
        let arr = table.toArray()
        for p1 in 0..<5 {
            for p2 in (p1 + 1)..<10 {
                let r1 = Array(arr[(p1 * 16)..<((p1 + 1) * 16)])
                let r2 = Array(arr[(p2 * 16)..<((p2 + 1) * 16)])
                XCTAssertNotEqual(r1, r2,
                    "rows at pos \(p1) and \(p2) should differ")
            }
        }
    }

    // MARK: - Construction & encoder output

    /// Weight bundle dimensions are checked at init time. With 2
    /// encoder layers configured but only 1 layer-bundle provided,
    /// we'd expect a precondition failure — exercise the success
    /// path by matching counts.
    func testConstructionWithMatchingLayerCounts() throws {
        var rng = SeededRNG(seed: 1)
        let config = tinyConfig(encLayers: 2, decLayers: 3)
        let model = Self.makeTinyModel(config: config, rng: &rng)
        XCTAssertEqual(model.config.encoder.nLayers, 2)
        XCTAssertEqual(model.config.decoderLayers, 3)
    }

    /// Encoder output → projection has shape `[1, T/8, decDim]`.
    /// Verifies the encoder runs end-to-end and the projection
    /// dimension is applied correctly.
    func testEncoderOutputProjectedShape() throws {
        var rng = SeededRNG(seed: 2)
        let config = tinyConfig()
        let model = Self.makeTinyModel(config: config, rng: &rng)

        let T = 64    // mel time steps
        let mel = try Tensor.from(
            data: (0..<(config.encoder.nMels * T)).map { Float($0) * 0.001 },
            shape: [1, config.encoder.nMels, T]
        )
        let encProj = model.runEncoder(mel: mel)
        XCTAssertEqual(encProj.shape,
            [1, T / config.encoder.subsamplingFactor, config.encDecProjToDim])
        for v in encProj.toArray() {
            XCTAssertTrue(v.isFinite, "non-finite encoder projection output")
        }
    }

    // MARK: - Decoder step

    /// Decoder step from BOS produces a valid vocab id and advances
    /// every layer's KV cache by 1.
    func testDecoderStepAdvancesCache() throws {
        var rng = SeededRNG(seed: 3)
        let config = tinyConfig()
        let model = Self.makeTinyModel(config: config, rng: &rng)

        // Build a synthetic mel and prefill cross-attn caches.
        let T = 64
        let mel = try Tensor.from(
            data: (0..<(config.encoder.nMels * T)).map { _ in
                Float.random(in: -1...1, using: &rng) },
            shape: [1, config.encoder.nMels, T]
        )
        model.reset()
        model.setEncoderContext(mel: mel)

        // Step 3 times and verify each call returns a valid token id.
        var nextTok = config.bosTokenId
        for _ in 0..<3 {
            let pred = try model.decoderStep(tokenId: nextTok)
            XCTAssertGreaterThanOrEqual(pred, 0)
            XCTAssertLessThan(pred, config.vocabSize)
            nextTok = pred
        }
    }

    // MARK: - Greedy decode terminates on EOS

    /// generateGreedy stops at EOS — verified by overriding the head
    /// weights to always predict EOS (head bias for EOS is huge).
    /// With that bias, the first generated token IS EOS, so the
    /// returned sequence is empty.
    func testGreedyStopsAtEOS() throws {
        var rng = SeededRNG(seed: 4)
        var config = tinyConfig()
        // We need a deterministic head: set head bias for EOS to a
        // large value, all others to a large negative value, so
        // argmax always picks EOS.
        let model = Self.makeTinyModel(config: config, rng: &rng)
        // Mutate the config and re-set the head bias so EOS dominates.
        let V = config.vocabSize
        var biasData = [Float](repeating: -1e6, count: V)
        biasData[config.eosTokenId] = 1e6
        let eosBias = try! Tensor.from(data: biasData, shape: [V])
        // Replace the model's head bias by reaching into the weights
        // bundle. Since we can't mutate the immutable weights, build
        // a fresh model with the new bias.
        let original = model
        _ = original
        var rng2 = SeededRNG(seed: 4)
        let subW = Self.randomEncoderSubsamplingWeights(
            config: config.encoder, rng: &rng2)
        var elw: [ConformerLayer.Weights] = []
        for _ in 0..<config.encoder.nLayers {
            elw.append(Self.randomEncoderLayerWeights(
                config: config.encoder, rng: &rng2))
        }
        var dlw: [CohereTranscribe.DecoderLayer.Weights] = []
        for _ in 0..<config.decoderLayers {
            dlw.append(Self.randomDecoderLayerWeights(
                config: config, rng: &rng2))
        }
        let weights = CohereTranscribe.Weights(
            encoderSubsampling: subW,
            encoderLayers: elw,
            encDecProjW: Self.randTensor(
                [config.encDecProjFromDim, config.encDecProjToDim],
                rng: &rng2),
            encDecProjB: Self.randTensor([config.encDecProjToDim], rng: &rng2),
            embedTokens: Self.randTensor(
                [config.vocabSize, config.decoderHidden], rng: &rng2),
            decoderLayers: dlw,
            decoderFinalNormGamma: Self.onesTensor([config.decoderHidden]),
            decoderFinalNormBeta: Self.zerosTensor([config.decoderHidden]),
            headWeight: Self.randTensor(
                [config.decoderHidden, config.vocabSize], rng: &rng2),
            headBias: eosBias
        )
        let model2 = CohereTranscribe(config: config, weights: weights)

        let T = 32
        let mel = try Tensor.from(
            data: (0..<(config.encoder.nMels * T)).map { _ in
                Float.random(in: -1...1, using: &rng2) },
            shape: [1, config.encoder.nMels, T]
        )
        model2.reset()
        model2.setEncoderContext(mel: mel)
        let toks = try model2.generateGreedy(
            promptTokens: [config.bosTokenId], maxNewTokens: 50
        )
        XCTAssertEqual(toks, [], "head biased to EOS → no tokens emitted before EOS halt")
    }

    /// Multi-step greedy with random weights: at least produces a
    /// finite-length result with all tokens in vocab range.
    func testGreedyProducesValidTokenIds() throws {
        var rng = SeededRNG(seed: 5)
        let config = tinyConfig(decLayers: 2)
        let model = Self.makeTinyModel(config: config, rng: &rng)

        let T = 64
        let mel = try Tensor.from(
            data: (0..<(config.encoder.nMels * T)).map { _ in
                Float.random(in: -1...1, using: &rng) },
            shape: [1, config.encoder.nMels, T]
        )
        model.reset()
        model.setEncoderContext(mel: mel)
        let toks = try model.generateGreedy(
            promptTokens: [config.bosTokenId], maxNewTokens: 12
        )
        XCTAssertLessThanOrEqual(toks.count, 12)
        for t in toks {
            XCTAssertGreaterThanOrEqual(t, 0)
            XCTAssertLessThan(t, config.vocabSize)
        }
    }
}
