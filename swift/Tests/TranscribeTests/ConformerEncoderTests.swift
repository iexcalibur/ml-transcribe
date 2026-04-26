import XCTest
import Foundation
@testable import Transcribe

/// Tests for the Cohere-Transcribe-style Conformer encoder. We don't
/// have real Cohere weights yet (Phase 4 territory), so these tests
/// build random weight bundles and verify:
///   - shape contracts at every stage of the pipeline
///   - finite outputs (no NaN / Inf from the conv module / batchnorm)
///   - structural identities (e.g. zero-attention behaves like the
///     identity through the attention sub-layer)
final class ConformerEncoderTests: XCTestCase {

    // MARK: - Test config

    /// Minimal config for fast tests: 2 layers, d_model=32, 4 heads,
    /// 4 mel bins (must divide subsamplingFactor=8? No — subsampling
    /// factor stride-2 cleaves input, so n_mels just needs to be
    /// divisible by 8. We bump to n_mels=16 = 2× subsampled freq).
    private func tinyConfig(nLayers: Int = 2) -> ConformerEncoder.Config {
        ConformerEncoder.Config(
            nMels: 16,
            dModel: 32,
            nLayers: nLayers,
            nHeads: 4,
            ffExpansion: 2,
            convKernelSize: 9,
            subsamplingChannels: 8,    // smaller than full 256 for speed
            subsamplingFactor: 8,
            layerNormEps: 1e-5,
            posEmbMaxLen: 1000
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

    private static func randomSubsamplingWeights(
        config: ConformerEncoder.Config,
        rng: inout SeededRNG
    ) -> ConvSubsampling.Weights {
        let C = config.subsamplingChannels
        let nMels = config.nMels
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
        _ = nMels   // (referenced for clarity)
    }

    private static func randomLayerWeights(
        config: ConformerEncoder.Config,
        rng: inout SeededRNG
    ) -> ConformerLayer.Weights {
        let D = config.dModel
        let F = config.ffnInnerDim
        let H = config.nHeads
        let Dh = config.headDim
        let K = config.convKernelSize
        return ConformerLayer.Weights(
            ffn1NormGamma: onesTensor([D]),
            ffn1NormBeta:  zerosTensor([D]),
            ffn1UpW:   randTensor([D, F], rng: &rng),
            ffn1UpB:   randTensor([F], rng: &rng),
            ffn1DownW: randTensor([F, D], rng: &rng),
            ffn1DownB: randTensor([D], rng: &rng),
            attnNormGamma: onesTensor([D]),
            attnNormBeta:  zerosTensor([D]),
            attnWQ: randTensor([D, D], rng: &rng), attnBQ: randTensor([D], rng: &rng),
            attnWK: randTensor([D, D], rng: &rng), attnBK: randTensor([D], rng: &rng),
            attnWV: randTensor([D, D], rng: &rng), attnBV: randTensor([D], rng: &rng),
            attnWPos: randTensor([D, D], rng: &rng),
            attnWOut: randTensor([D, D], rng: &rng), attnBOut: randTensor([D], rng: &rng),
            attnPosBiasU: randTensor([H, Dh], rng: &rng),
            attnPosBiasV: randTensor([H, Dh], rng: &rng),
            convNormGamma: onesTensor([D]),
            convNormBeta:  zerosTensor([D]),
            convPointwise1Weight: randTensor([2 * D, D, 1], rng: &rng),
            convPointwise1Bias:   randTensor([2 * D], rng: &rng),
            convDepthwiseWeight: randTensor([D, 1, K], rng: &rng),
            convDepthwiseBias:   randTensor([D], rng: &rng),
            convBNRunningMean:   zerosTensor([D]),
            convBNRunningVar:    onesTensor([D]),
            convBNGamma: onesTensor([D]),
            convBNBeta:  zerosTensor([D]),
            convPointwise2Weight: randTensor([D, D, 1], rng: &rng),
            convPointwise2Bias:   randTensor([D], rng: &rng),
            ffn2NormGamma: onesTensor([D]),
            ffn2NormBeta:  zerosTensor([D]),
            ffn2UpW:   randTensor([D, F], rng: &rng),
            ffn2UpB:   randTensor([F], rng: &rng),
            ffn2DownW: randTensor([F, D], rng: &rng),
            ffn2DownB: randTensor([D], rng: &rng),
            normOutGamma: onesTensor([D]),
            normOutBeta:  zerosTensor([D])
        )
    }

    // MARK: - ConvSubsampling shape

    /// 8× time subsampling: input mel `[1, 16, 64]` → embedding
    /// `[1, 8, dModel]` (after the conv stack and linear projection).
    func testConvSubsamplingShape() throws {
        let config = tinyConfig()
        var rng = SeededRNG(seed: 1)
        let weights = Self.randomSubsamplingWeights(config: config, rng: &rng)
        let sub = ConvSubsampling(config: config, weights: weights)

        let T = 64
        let mel = try Tensor.from(
            data: (0..<(config.nMels * T)).map { Float($0) * 0.001 },
            shape: [1, config.nMels, T]
        )
        let out = sub.forward(mel: mel)
        XCTAssertEqual(out.shape, [1, T / config.subsamplingFactor, config.dModel])
        for v in out.toArray() {
            XCTAssertTrue(v.isFinite, "non-finite ConvSubsampling output")
        }
    }

    // MARK: - ConformerLayer shape & finiteness

    /// Single-layer forward keeps `[1, T, D]` shape and produces
    /// finite values on random weights.
    func testConformerLayerShapeAndFinite() throws {
        let config = tinyConfig(nLayers: 1)
        var rng = SeededRNG(seed: 2)
        let lw = Self.randomLayerWeights(config: config, rng: &rng)
        let layer = ConformerLayer(config: config, weights: lw)

        let T = 12
        let x = try Tensor.from(
            data: (0..<(T * config.dModel)).map { _ in
                Float.random(in: -1.0...1.0, using: &rng)
            },
            shape: [1, T, config.dModel]
        )
        let posEmb = RelPosSelfAttention.makeSinusoidalPosEmb(
            seqLen: T, dModel: config.dModel
        )
        let out = layer.forward(x, posEmb: posEmb)
        XCTAssertEqual(out.shape, [1, T, config.dModel])
        for v in out.toArray() {
            XCTAssertTrue(v.isFinite, "non-finite ConformerLayer output")
        }
    }

    // MARK: - Macaron half-scaling identity

    /// With every "delta-producing" sub-layer zeroed (FFN/attn/conv
    /// all output zero), the residual stream passes through unchanged
    /// modulo the four LayerNorms. Setting LN gamma=1 / beta=0 with
    /// pre-normalized input (zero mean / unit variance per row) keeps
    /// the residual stream itself untouched, so the final output
    /// equals the input within FP tolerance.
    ///
    /// We construct that scenario by zeroing the FFN-down,
    /// attn-out, and conv-pointwise2 weights — that's enough to
    /// drive every sub-layer's output to zero. The Macaron half
    /// scaling drops zeros, so all four residual additions are
    /// no-ops.
    func testZeroDeltaSublayersBecomeIdentity() throws {
        let config = tinyConfig(nLayers: 1)
        var rng = SeededRNG(seed: 3)
        var lw = Self.randomLayerWeights(config: config, rng: &rng)
        // Zero out the OUTPUT projections of all three sub-layers
        // (FFN1 down, attn out, conv pointwise2, FFN2 down). With
        // zero-output projections every "delta" component is 0, so
        // residuals stay equal to the input.
        let D = config.dModel
        let F = config.ffnInnerDim
        let zeroFD = Self.zerosTensor([F, D])
        let zeroDD = Self.zerosTensor([D, D])
        let zeroDD1 = Self.zerosTensor([D, D, 1])
        lw = ConformerLayer.Weights(
            ffn1NormGamma: lw.ffn1NormGamma, ffn1NormBeta: lw.ffn1NormBeta,
            ffn1UpW: lw.ffn1UpW, ffn1UpB: lw.ffn1UpB,
            ffn1DownW: zeroFD, ffn1DownB: Self.zerosTensor([D]),
            attnNormGamma: lw.attnNormGamma, attnNormBeta: lw.attnNormBeta,
            attnWQ: lw.attnWQ, attnBQ: lw.attnBQ,
            attnWK: lw.attnWK, attnBK: lw.attnBK,
            attnWV: lw.attnWV, attnBV: lw.attnBV,
            attnWPos: lw.attnWPos,
            attnWOut: zeroDD, attnBOut: Self.zerosTensor([D]),
            attnPosBiasU: lw.attnPosBiasU, attnPosBiasV: lw.attnPosBiasV,
            convNormGamma: lw.convNormGamma, convNormBeta: lw.convNormBeta,
            convPointwise1Weight: lw.convPointwise1Weight,
            convPointwise1Bias: lw.convPointwise1Bias,
            convDepthwiseWeight: lw.convDepthwiseWeight,
            convDepthwiseBias: lw.convDepthwiseBias,
            convBNRunningMean: lw.convBNRunningMean,
            convBNRunningVar: lw.convBNRunningVar,
            convBNGamma: lw.convBNGamma, convBNBeta: lw.convBNBeta,
            convPointwise2Weight: zeroDD1, convPointwise2Bias: Self.zerosTensor([D]),
            ffn2NormGamma: lw.ffn2NormGamma, ffn2NormBeta: lw.ffn2NormBeta,
            ffn2UpW: lw.ffn2UpW, ffn2UpB: lw.ffn2UpB,
            ffn2DownW: zeroFD, ffn2DownB: Self.zerosTensor([D]),
            normOutGamma: lw.normOutGamma, normOutBeta: lw.normOutBeta
        )
        let layer = ConformerLayer(config: config, weights: lw)

        // Use a known input with ~zero mean / unit variance per row
        // so the final LN doesn't distort it.
        let T = 8
        var inputData: [Float] = []
        for _ in 0..<T {
            // Random row, then normalize to mean 0 / std 1.
            var row = (0..<D).map { _ in Float.random(in: -1...1, using: &rng) }
            let mean = row.reduce(0, +) / Float(D)
            for i in 0..<D { row[i] -= mean }
            let v = row.map { $0 * $0 }.reduce(0, +) / Float(D)
            let std = sqrtf(v + 1e-6)
            for i in 0..<D { row[i] /= std }
            inputData += row
        }
        let x = try Tensor.from(data: inputData, shape: [1, T, D])
        let posEmb = RelPosSelfAttention.makeSinusoidalPosEmb(
            seqLen: T, dModel: D
        )
        let out = layer.forward(x, posEmb: posEmb).toArray()

        // Output should equal input within FP tolerance (final LN
        // recentres but input is already unit-normalized).
        let inArr = x.toArray()
        for i in 0..<inArr.count {
            XCTAssertEqual(out[i], inArr[i], accuracy: 1e-2,
                "residual identity broken at index \(i): in=\(inArr[i]) out=\(out[i])")
        }
    }

    // MARK: - Full encoder smoke test

    /// 2-layer encoder, 64-frame mel input → 8-frame `[1, 8, 32]`
    /// hidden state, all finite.
    func testFullEncoderForwardShape() throws {
        let config = tinyConfig(nLayers: 2)
        var rng = SeededRNG(seed: 4)
        let subWeights = Self.randomSubsamplingWeights(config: config, rng: &rng)
        var layerWeights: [ConformerLayer.Weights] = []
        for _ in 0..<config.nLayers {
            layerWeights.append(Self.randomLayerWeights(config: config, rng: &rng))
        }
        let encoder = ConformerEncoder(
            config: config,
            subsamplingWeights: subWeights,
            layerWeights: layerWeights
        )
        let T = 64
        let mel = try Tensor.from(
            data: (0..<(config.nMels * T)).map { Float($0) * 0.001 },
            shape: [1, config.nMels, T]
        )
        let out = encoder.forward(mel: mel)
        XCTAssertEqual(out.shape, [1, T / config.subsamplingFactor, config.dModel])
        for v in out.toArray() {
            XCTAssertTrue(v.isFinite, "non-finite encoder output")
        }
    }
}
