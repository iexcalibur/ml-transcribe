import Foundation

/// Fast-Conformer encoder, used by Cohere Transcribe (and NeMo's
/// `EncDecMultiTaskModel`). The audio frontend ("ConvSubsampling")
/// reduces the time axis 8× via three strided 2-D convolutions, then
/// a stack of `ConformerLayer` blocks each combine a Macaron-style
/// half-step feed-forward, relative-positional self-attention, a
/// depthwise convolution module, and another half-step feed-forward.
///
/// Defaults match Cohere Transcribe-2B exactly (48 layers, d_model
/// 1280, 8 heads, FFN expansion 4, conv kernel 9, 8× subsampling).
public final class ConformerEncoder {

    // MARK: - Config

    public struct Config {
        public let nMels: Int               // input mel bins (=128)
        public let dModel: Int              // (=1280)
        public let nLayers: Int             // (=48)
        public let nHeads: Int              // (=8)
        public let ffExpansion: Int         // (=4 → ff_dim=5120)
        public let convKernelSize: Int      // (=9, depthwise inside ConvModule)
        public let subsamplingChannels: Int // (=256, intermediate for ConvSubsampling)
        public let subsamplingFactor: Int   // (=8, fixed for "dw_striding")
        public let layerNormEps: Float
        public let posEmbMaxLen: Int        // (=5000, upper bound on encoder seq_len)

        public var ffnInnerDim: Int { dModel * ffExpansion }
        public var headDim: Int { dModel / nHeads }
        /// Frequency dimension after the conv frontend
        /// (n_mels reduced by the same factor as time).
        public var subsampledFreq: Int { nMels / subsamplingFactor }

        public init(
            nMels: Int = 128,
            dModel: Int = 1280,
            nLayers: Int = 48,
            nHeads: Int = 8,
            ffExpansion: Int = 4,
            convKernelSize: Int = 9,
            subsamplingChannels: Int = 256,
            subsamplingFactor: Int = 8,
            layerNormEps: Float = 1e-5,
            posEmbMaxLen: Int = 5000
        ) {
            self.nMels = nMels
            self.dModel = dModel
            self.nLayers = nLayers
            self.nHeads = nHeads
            self.ffExpansion = ffExpansion
            self.convKernelSize = convKernelSize
            self.subsamplingChannels = subsamplingChannels
            self.subsamplingFactor = subsamplingFactor
            self.layerNormEps = layerNormEps
            self.posEmbMaxLen = posEmbMaxLen
        }

        public static let cohereTranscribe2B = Config()
    }

    public let config: Config

    // MARK: - Submodules

    private let subsampling: ConvSubsampling
    private let layers: [ConformerLayer]

    // MARK: - Init

    /// Construct from a flat list of preloaded `ConformerLayer.Weights`
    /// plus the `ConvSubsampling.Weights` bundle. Higher-level loaders
    /// (e.g. a Cohere-specific `WeightMap`) will fill these in from a
    /// safetensors file in Phase 4; for now, callers can construct
    /// random weights for testing.
    public init(
        config: Config,
        subsamplingWeights: ConvSubsampling.Weights,
        layerWeights: [ConformerLayer.Weights]
    ) {
        precondition(layerWeights.count == config.nLayers,
            "expected \(config.nLayers) layer weight bundles, got \(layerWeights.count)")
        self.config = config
        self.subsampling = ConvSubsampling(
            config: config, weights: subsamplingWeights
        )
        self.layers = layerWeights.map {
            ConformerLayer(config: config, weights: $0)
        }
    }

    // MARK: - Forward

    /// Encode a log-mel spectrogram into the encoder hidden state.
    ///
    /// - `mel`: `[1, n_mels, T]` (n_mels-major, time-minor — exactly
    ///   what `AudioPreprocessor.logMelSpectrogram` returns once
    ///   batched).
    ///
    /// Returns hidden states `[1, T/subsamplingFactor, dModel]`.
    public func forward(mel: Tensor) -> Tensor {
        // Subsample audio into time-major embeddings.
        var x = subsampling.forward(mel: mel)        // [1, T', D]
        let tPrime = x.shape[1]

        // Build the relative-position table for this seq_len once;
        // the posEmb itself is shared by every layer in the stack.
        let posEmb = RelPosSelfAttention.makeSinusoidalPosEmb(
            seqLen: tPrime, dModel: config.dModel
        )

        // Run the layer stack.
        for layer in layers {
            x = layer.forward(x, posEmb: posEmb)
        }
        return x
    }
}

// =============================================================================
// ConvSubsampling — audio frontend
// =============================================================================

/// Reduces the time axis 8× via three strided 2-D convolutions
/// interleaved with two pointwise convolutions, then projects onto
/// `dModel` features. NeMo / Cohere call this "dw_striding"
/// subsampling.
///
/// Layer order (each but the last followed by ReLU):
///   Conv2d(1,   C,  k=3, s=2, p=1)               # dense
///   Conv2d(C,   C,  k=3, s=2, p=1, groups=C)     # depthwise
///   Conv2d(C,   C,  k=1)                          # pointwise
///   Conv2d(C,   C,  k=3, s=2, p=1, groups=C)     # depthwise
///   Conv2d(C,   C,  k=1)                          # pointwise
///   permute + reshape → [1, T', C * (n_mels/8)]
///   Linear(C * n_mels/8 → dModel)
///
/// Where C = config.subsamplingChannels (= 256 for Cohere Transcribe).
public final class ConvSubsampling {

    public struct Weights {
        public let conv1Weight: Tensor      // [C, 1, 3, 3]
        public let conv1Bias: Tensor        // [C]
        public let conv2Weight: Tensor      // [C, 1, 3, 3]   (depthwise — C_in/groups=1)
        public let conv2Bias: Tensor        // [C]
        public let conv3Weight: Tensor      // [C, C, 1, 1]
        public let conv3Bias: Tensor        // [C]
        public let conv4Weight: Tensor      // [C, 1, 3, 3]   (depthwise)
        public let conv4Bias: Tensor        // [C]
        public let conv5Weight: Tensor      // [C, C, 1, 1]
        public let conv5Bias: Tensor        // [C]
        public let outProjWeight: Tensor    // [C * (n_mels/8), dModel]   (in our [in, out] layout)
        public let outProjBias: Tensor      // [dModel]

        public init(
            conv1Weight: Tensor, conv1Bias: Tensor,
            conv2Weight: Tensor, conv2Bias: Tensor,
            conv3Weight: Tensor, conv3Bias: Tensor,
            conv4Weight: Tensor, conv4Bias: Tensor,
            conv5Weight: Tensor, conv5Bias: Tensor,
            outProjWeight: Tensor, outProjBias: Tensor
        ) {
            self.conv1Weight = conv1Weight; self.conv1Bias = conv1Bias
            self.conv2Weight = conv2Weight; self.conv2Bias = conv2Bias
            self.conv3Weight = conv3Weight; self.conv3Bias = conv3Bias
            self.conv4Weight = conv4Weight; self.conv4Bias = conv4Bias
            self.conv5Weight = conv5Weight; self.conv5Bias = conv5Bias
            self.outProjWeight = outProjWeight; self.outProjBias = outProjBias
        }
    }

    private let config: ConformerEncoder.Config
    private let weights: Weights

    init(config: ConformerEncoder.Config, weights: Weights) {
        self.config = config
        self.weights = weights
    }

    /// Forward.
    ///
    /// - `mel`: `[1, n_mels, T]` row-major. Internally we transpose
    ///   to `[1, 1, T, n_mels]` (treating time as image-height and
    ///   freq as image-width) for conv2d.
    ///
    /// Output: `[1, T/8, dModel]`.
    public func forward(mel: Tensor) -> Tensor {
        precondition(mel.shape.count == 3 && mel.shape[0] == 1,
            "ConvSubsampling expects [1, n_mels, T], got \(mel.shape)")
        let nMels = mel.shape[1]
        let T = mel.shape[2]
        precondition(nMels == config.nMels,
            "input has \(nMels) mel bins, config expects \(config.nMels)")

        // [1, n_mels, T] → [1, T, n_mels] → [1, 1, T, n_mels]
        let asImage = mel
            .permute([0, 2, 1]).contiguous()
            .reshape([1, 1, T, nMels])

        let C = config.subsamplingChannels
        // Conv biases are [C]; reshape to [1, C, 1, 1] for broadcast.
        func b(_ bias: Tensor) -> Tensor { bias.reshape([1, C, 1, 1]) }

        // Layer 1: dense conv, k=3, s=2, p=1, ReLU.
        let h1 = asImage
            .conv2d(weight: weights.conv1Weight, stride: 2, padding: 1, groups: 1)
            .add(b(weights.conv1Bias))
            .relu()
        // Layer 2: depthwise, k=3, s=2, p=1, groups=C, ReLU.
        let h2 = h1
            .conv2d(weight: weights.conv2Weight, stride: 2, padding: 1, groups: C)
            .add(b(weights.conv2Bias))
            .relu()
        // Layer 3: pointwise, k=1, ReLU.
        let h3 = h2
            .conv2d(weight: weights.conv3Weight)
            .add(b(weights.conv3Bias))
            .relu()
        // Layer 4: depthwise, k=3, s=2, p=1, groups=C, ReLU.
        let h4 = h3
            .conv2d(weight: weights.conv4Weight, stride: 2, padding: 1, groups: C)
            .add(b(weights.conv4Bias))
            .relu()
        // Layer 5: pointwise, k=1, ReLU.
        let h5 = h4
            .conv2d(weight: weights.conv5Weight)
            .add(b(weights.conv5Bias))
            .relu()
        // h5 shape: [1, C, T/8, n_mels/8]

        let tPrime = h5.shape[2]
        let fPrime = h5.shape[3]
        // Permute [1, C, T/8, F'] → [1, T/8, C, F'] then flatten last
        // two dims into [1, T/8, C * F'].
        let flat = h5
            .permute([0, 2, 1, 3]).contiguous()
            .reshape([1, tPrime, C * fPrime])

        // Final Linear projection onto dModel.
        let bias3 = weights.outProjBias.reshape([1, 1, config.dModel])
        return flat.matmul(weights.outProjWeight).add(bias3)
    }
}

// =============================================================================
// ConformerLayer — one block of the encoder stack
// =============================================================================

/// A single Conformer block. Macaron pre-norm structure:
///
///     x → LN → FFN(0.5×) → +x
///       → LN → RelPosSelfAttention → +x
///       → LN → ConvModule → +x
///       → LN → FFN(0.5×) → +x
///       → LN (final)
public final class ConformerLayer {

    /// Bundle of every tensor needed by one Conformer block.
    public struct Weights {
        // First half-step FFN (Macaron).
        public let ffn1NormGamma, ffn1NormBeta: Tensor          // [D]
        public let ffn1UpW, ffn1UpB: Tensor                     // [D, F], [F]
        public let ffn1DownW, ffn1DownB: Tensor                 // [F, D], [D]

        // Relative-positional self-attention.
        public let attnNormGamma, attnNormBeta: Tensor          // [D]
        public let attnWQ, attnBQ: Tensor
        public let attnWK, attnBK: Tensor
        public let attnWV, attnBV: Tensor
        public let attnWPos: Tensor
        public let attnWOut, attnBOut: Tensor
        public let attnPosBiasU, attnPosBiasV: Tensor           // [H, Dh]

        // Convolution module (depthwise core).
        public let convNormGamma, convNormBeta: Tensor          // [D]
        public let convPointwise1Weight: Tensor                 // [2D, D, 1]
        public let convPointwise1Bias: Tensor                   // [2D]
        public let convDepthwiseWeight: Tensor                  // [D, 1, K]
        public let convDepthwiseBias: Tensor                    // [D]
        public let convBNRunningMean: Tensor                    // [D]
        public let convBNRunningVar: Tensor                     // [D]
        public let convBNGamma, convBNBeta: Tensor              // [D]
        public let convPointwise2Weight: Tensor                 // [D, D, 1]
        public let convPointwise2Bias: Tensor                   // [D]

        // Second half-step FFN (Macaron, mirrored).
        public let ffn2NormGamma, ffn2NormBeta: Tensor          // [D]
        public let ffn2UpW, ffn2UpB: Tensor                     // [D, F], [F]
        public let ffn2DownW, ffn2DownB: Tensor                 // [F, D], [D]

        // Final layer norm.
        public let normOutGamma, normOutBeta: Tensor            // [D]

        public init(
            ffn1NormGamma: Tensor, ffn1NormBeta: Tensor,
            ffn1UpW: Tensor, ffn1UpB: Tensor,
            ffn1DownW: Tensor, ffn1DownB: Tensor,
            attnNormGamma: Tensor, attnNormBeta: Tensor,
            attnWQ: Tensor, attnBQ: Tensor,
            attnWK: Tensor, attnBK: Tensor,
            attnWV: Tensor, attnBV: Tensor,
            attnWPos: Tensor,
            attnWOut: Tensor, attnBOut: Tensor,
            attnPosBiasU: Tensor, attnPosBiasV: Tensor,
            convNormGamma: Tensor, convNormBeta: Tensor,
            convPointwise1Weight: Tensor, convPointwise1Bias: Tensor,
            convDepthwiseWeight: Tensor, convDepthwiseBias: Tensor,
            convBNRunningMean: Tensor, convBNRunningVar: Tensor,
            convBNGamma: Tensor, convBNBeta: Tensor,
            convPointwise2Weight: Tensor, convPointwise2Bias: Tensor,
            ffn2NormGamma: Tensor, ffn2NormBeta: Tensor,
            ffn2UpW: Tensor, ffn2UpB: Tensor,
            ffn2DownW: Tensor, ffn2DownB: Tensor,
            normOutGamma: Tensor, normOutBeta: Tensor
        ) {
            self.ffn1NormGamma = ffn1NormGamma; self.ffn1NormBeta = ffn1NormBeta
            self.ffn1UpW = ffn1UpW; self.ffn1UpB = ffn1UpB
            self.ffn1DownW = ffn1DownW; self.ffn1DownB = ffn1DownB
            self.attnNormGamma = attnNormGamma; self.attnNormBeta = attnNormBeta
            self.attnWQ = attnWQ; self.attnBQ = attnBQ
            self.attnWK = attnWK; self.attnBK = attnBK
            self.attnWV = attnWV; self.attnBV = attnBV
            self.attnWPos = attnWPos
            self.attnWOut = attnWOut; self.attnBOut = attnBOut
            self.attnPosBiasU = attnPosBiasU; self.attnPosBiasV = attnPosBiasV
            self.convNormGamma = convNormGamma; self.convNormBeta = convNormBeta
            self.convPointwise1Weight = convPointwise1Weight
            self.convPointwise1Bias = convPointwise1Bias
            self.convDepthwiseWeight = convDepthwiseWeight
            self.convDepthwiseBias = convDepthwiseBias
            self.convBNRunningMean = convBNRunningMean
            self.convBNRunningVar = convBNRunningVar
            self.convBNGamma = convBNGamma; self.convBNBeta = convBNBeta
            self.convPointwise2Weight = convPointwise2Weight
            self.convPointwise2Bias = convPointwise2Bias
            self.ffn2NormGamma = ffn2NormGamma; self.ffn2NormBeta = ffn2NormBeta
            self.ffn2UpW = ffn2UpW; self.ffn2UpB = ffn2UpB
            self.ffn2DownW = ffn2DownW; self.ffn2DownB = ffn2DownB
            self.normOutGamma = normOutGamma; self.normOutBeta = normOutBeta
        }
    }

    private let config: ConformerEncoder.Config
    private let weights: Weights
    private let attention: RelPosSelfAttention
    private let halfScale: Tensor    // 1-element tensor of value 0.5

    init(config: ConformerEncoder.Config, weights: Weights) {
        self.config = config
        self.weights = weights
        self.attention = RelPosSelfAttention(
            dModel: config.dModel,
            numHeads: config.nHeads,
            wQ: weights.attnWQ, bQ: weights.attnBQ,
            wK: weights.attnWK, bK: weights.attnBK,
            wV: weights.attnWV, bV: weights.attnBV,
            wPos: weights.attnWPos,
            wOut: weights.attnWOut, bOut: weights.attnBOut,
            posBiasU: weights.attnPosBiasU,
            posBiasV: weights.attnPosBiasV
        )
        self.halfScale = (try? Tensor.from(data: [0.5], shape: [1])) ?? {
            preconditionFailure("could not allocate halfScale tensor")
        }()
    }

    /// Forward pass through one Conformer block.
    /// - `x`: `[1, T, D]` (the running residual stream)
    /// - `posEmb`: `[1, 2T-1, D]` (shared across the encoder stack)
    public func forward(_ x: Tensor, posEmb: Tensor) -> Tensor {
        let D = config.dModel
        let F = config.ffnInnerDim
        let K = config.convKernelSize

        // ---- Half-step FFN #1 (Macaron half) -----------------------
        let ffn1Norm = x.layerNorm(
            gamma: weights.ffn1NormGamma, beta: weights.ffn1NormBeta,
            eps: config.layerNormEps
        )
        let ffn1Out = ffnForward(
            ffn1Norm,
            wUp: weights.ffn1UpW,   bUp: weights.ffn1UpB,
            wDown: weights.ffn1DownW, bDown: weights.ffn1DownB,
            ffnInner: F, dModel: D
        )
        let afterFfn1 = x.add(ffn1Out.mul(halfScale))

        // ---- Self-attention ---------------------------------------
        let attnNorm = afterFfn1.layerNorm(
            gamma: weights.attnNormGamma, beta: weights.attnNormBeta,
            eps: config.layerNormEps
        )
        let attnOut = attention.forward(attnNorm, posEmb: posEmb)
        let afterAttn = afterFfn1.add(attnOut)

        // ---- Convolution module -----------------------------------
        let convNorm = afterAttn.layerNorm(
            gamma: weights.convNormGamma, beta: weights.convNormBeta,
            eps: config.layerNormEps
        )
        let convOut = convModuleForward(
            convNorm, D: D, K: K
        )
        let afterConv = afterAttn.add(convOut)

        // ---- Half-step FFN #2 (Macaron half, other side) ----------
        let ffn2Norm = afterConv.layerNorm(
            gamma: weights.ffn2NormGamma, beta: weights.ffn2NormBeta,
            eps: config.layerNormEps
        )
        let ffn2Out = ffnForward(
            ffn2Norm,
            wUp: weights.ffn2UpW,   bUp: weights.ffn2UpB,
            wDown: weights.ffn2DownW, bDown: weights.ffn2DownB,
            ffnInner: F, dModel: D
        )
        let afterFfn2 = afterConv.add(ffn2Out.mul(halfScale))

        // ---- Final LayerNorm --------------------------------------
        return afterFfn2.layerNorm(
            gamma: weights.normOutGamma, beta: weights.normOutBeta,
            eps: config.layerNormEps
        )
    }

    /// Standard pre-norm FFN with SiLU activation:
    /// `x → matmul(W_up) + b_up → SiLU → matmul(W_down) + b_down`.
    /// Caller takes care of the LayerNorm (passed in already normed)
    /// and the half-scaling and residual.
    private func ffnForward(
        _ x: Tensor,
        wUp: Tensor, bUp: Tensor,
        wDown: Tensor, bDown: Tensor,
        ffnInner F: Int, dModel D: Int
    ) -> Tensor {
        let bUp3 = bUp.reshape([1, 1, F])
        let bDown3 = bDown.reshape([1, 1, D])
        return x
            .matmul(wUp).add(bUp3)
            .silu()
            .matmul(wDown).add(bDown3)
    }

    /// Conformer's convolution module:
    ///   pointwise(d → 2d) → GLU(dim=channel) → depthwise(k=K) →
    ///   BatchNorm1d → SiLU → pointwise(d → d).
    ///
    /// Input is already pre-normed `[1, T, D]`. We transpose to
    /// `[1, D, T]` for Conv1d, run the kernels, then transpose back.
    private func convModuleForward(
        _ x: Tensor, D: Int, K: Int
    ) -> Tensor {
        // [1, T, D] → [1, D, T]
        let asChannels = x.permute([0, 2, 1]).contiguous()
        let T = asChannels.shape[2]

        // pointwise(d → 2d), kernel=1, no padding.
        let bP1 = weights.convPointwise1Bias.reshape([1, 2 * D, 1])
        let expanded = asChannels
            .conv1d(weight: weights.convPointwise1Weight)
            .add(bP1)            // [1, 2D, T]
        // GLU along the channel axis → [1, D, T].
        let gated = expanded.glu(dim: 1)

        // depthwise conv, kernel=K, padding=(K-1)/2 to keep T,
        // groups=D for true per-channel conv.
        let bDw = weights.convDepthwiseBias.reshape([1, D, 1])
        let depthwise = gated
            .conv1d(
                weight: weights.convDepthwiseWeight,
                stride: 1, padding: (K - 1) / 2, groups: D
            )
            .add(bDw)            // [1, D, T]

        // BatchNorm1d (inference) → SiLU.
        let normed = depthwise.batchNorm1d(
            runningMean: weights.convBNRunningMean,
            runningVar:  weights.convBNRunningVar,
            gamma:       weights.convBNGamma,
            beta:        weights.convBNBeta,
            eps:         config.layerNormEps
        ).silu()

        // pointwise(d → d), kernel=1.
        let bP2 = weights.convPointwise2Bias.reshape([1, D, 1])
        let collapsed = normed
            .conv1d(weight: weights.convPointwise2Weight)
            .add(bP2)            // [1, D, T]

        // [1, D, T] → [1, T, D]
        _ = T   // silence unused-let warning if compiler ever objects
        return collapsed.permute([0, 2, 1]).contiguous()
    }
}
