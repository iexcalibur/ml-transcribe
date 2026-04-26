import Foundation

/// End-to-end Cohere Transcribe (2B) inference: audio → text.
///
/// The pipeline is the standard encoder-decoder ASR shape:
///   1. Audio → 128-bin per-feature-normalized log-mel spectrogram
///      (via `AudioPreprocessor.Config.cohereTranscribe`).
///   2. Encoder: Fast-Conformer (`ConformerEncoder`), 48 blocks,
///      d_model=1280, 8× time subsampling. Output `[1, T', 1280]`.
///   3. Encoder→decoder projection: `Linear(1280, 1024)` since the
///      decoder runs at a smaller hidden size.
///   4. Cache the projected encoder output as the K/V source for each
///      decoder layer's cross-attention (computed once per inference).
///   5. Decoder: 8-block lightweight transformer, d_model=1024, 8
///      heads, 4096 FFN with ReLU, fixed sinusoidal positions.
///   6. Token classifier head: `Linear(1024, 16384)` → argmax →
///      detokenize.
///
/// Greedy decoding is anchored on a forced prompt prefix that selects
/// the source/target language and task (no-timestamp / pnc / no-itn /
/// etc.) via Cohere's prompt-format tokens. The default prompt
/// targets English transcription with PnC (punctuation+capitalization)
/// and no timestamps; pass a custom `promptTokens` to `transcribe(...)`
/// for a different mode.
public final class CohereTranscribe {

    // -------------------------------------------------------------
    // Config
    // -------------------------------------------------------------

    public struct Config {
        // Encoder
        public let encoder: ConformerEncoder.Config
        // Encoder → decoder projection (since encoder.d_model ≠ decoder.dim).
        public let encDecProjFromDim: Int    // = encoder.dModel = 1280
        public let encDecProjToDim: Int      // = decoder.hidden_size = 1024
        // Decoder
        public let decoderHidden: Int        // = 1024
        public let decoderLayers: Int        // = 8
        public let decoderHeads: Int         // = 8
        public let decoderFFNDim: Int        // = 4096
        public let maxSeqLen: Int            // = 1024
        public let vocabSize: Int            // = 16384
        public let layerNormEps: Float
        // Special token ids (looked up after the SentencePiece tokenizer
        // is loaded; for now the caller passes them in).
        public let bosTokenId: Int
        public let eosTokenId: Int
        public let padTokenId: Int

        public var decoderHeadDim: Int { decoderHidden / decoderHeads }

        public init(
            encoder: ConformerEncoder.Config = .cohereTranscribe2B,
            encDecProjFromDim: Int = 1280,
            encDecProjToDim: Int = 1024,
            decoderHidden: Int = 1024,
            decoderLayers: Int = 8,
            decoderHeads: Int = 8,
            decoderFFNDim: Int = 4096,
            maxSeqLen: Int = 1024,
            vocabSize: Int = 16384,
            layerNormEps: Float = 1e-5,
            bosTokenId: Int = 0,    // placeholder — overridden at construction time
            eosTokenId: Int = 0,
            padTokenId: Int = 0
        ) {
            self.encoder = encoder
            self.encDecProjFromDim = encDecProjFromDim
            self.encDecProjToDim = encDecProjToDim
            self.decoderHidden = decoderHidden
            self.decoderLayers = decoderLayers
            self.decoderHeads = decoderHeads
            self.decoderFFNDim = decoderFFNDim
            self.maxSeqLen = maxSeqLen
            self.vocabSize = vocabSize
            self.layerNormEps = layerNormEps
            self.bosTokenId = bosTokenId
            self.eosTokenId = eosTokenId
            self.padTokenId = padTokenId
        }

        public static let cohereTranscribe2B = Config()
    }

    // -------------------------------------------------------------
    // Weights (full model bundle)
    // -------------------------------------------------------------

    public struct Weights {
        public let encoderSubsampling: ConvSubsampling.Weights
        public let encoderLayers: [ConformerLayer.Weights]
        // Encoder → decoder projection.
        public let encDecProjW: Tensor              // [encDim, decDim] (in our [in,out] convention)
        public let encDecProjB: Tensor              // [decDim]
        // Token embedding.
        public let embedTokens: Tensor              // [V, decDim]
        // LayerNorm applied AFTER token + position embedding sum,
        // before the first decoder layer. NeMo's
        // `TransformerDecoderEmbedding` includes this in the embed
        // module — it's easy to miss.
        public let embedNormGamma: Tensor           // [decDim]
        public let embedNormBeta: Tensor            // [decDim]
        // Decoder layers.
        public let decoderLayers: [DecoderLayer.Weights]
        // Final LayerNorm before the head.
        public let decoderFinalNormGamma: Tensor    // [decDim]
        public let decoderFinalNormBeta: Tensor     // [decDim]
        // Token classifier head: Linear → log_softmax. We omit
        // log_softmax during inference (argmax is monotonic w.r.t.
        // log-softmax, so it doesn't change which token wins).
        public let headWeight: Tensor               // [decDim, V]
        public let headBias: Tensor                 // [V]
        // Optional pre-computed mel filterbank, baked into the
        // safetensors as `preprocessor.featurizer.fb`. Shape after
        // reshape: `[nMels, nFFT/2+1]` = `[128, 257]` for Cohere.
        // Pass when you want the audio frontend to use the model's
        // training-time filterbank verbatim, sidestepping any
        // librosa / HTK / Slaney mel-scale debate.
        public let preprocessorFb: Tensor?

        public init(
            encoderSubsampling: ConvSubsampling.Weights,
            encoderLayers: [ConformerLayer.Weights],
            encDecProjW: Tensor, encDecProjB: Tensor,
            embedTokens: Tensor,
            embedNormGamma: Tensor, embedNormBeta: Tensor,
            decoderLayers: [DecoderLayer.Weights],
            decoderFinalNormGamma: Tensor,
            decoderFinalNormBeta: Tensor,
            headWeight: Tensor, headBias: Tensor,
            preprocessorFb: Tensor? = nil
        ) {
            self.encoderSubsampling = encoderSubsampling
            self.encoderLayers = encoderLayers
            self.encDecProjW = encDecProjW; self.encDecProjB = encDecProjB
            self.embedTokens = embedTokens
            self.embedNormGamma = embedNormGamma
            self.embedNormBeta = embedNormBeta
            self.decoderLayers = decoderLayers
            self.decoderFinalNormGamma = decoderFinalNormGamma
            self.decoderFinalNormBeta = decoderFinalNormBeta
            self.headWeight = headWeight; self.headBias = headBias
            self.preprocessorFb = preprocessorFb
        }
    }

    // -------------------------------------------------------------
    // Internal state
    // -------------------------------------------------------------

    public let config: Config
    private let encoder: ConformerEncoder
    private let weights: Weights
    private let decoderLayers: [DecoderLayer]
    /// Pre-computed sinusoidal position table for the decoder,
    /// shape `[maxSeqLen, decoderHidden]`. We slice rows from this
    /// at each decoder step to add to the token embedding.
    private let decoderPosTable: Tensor

    /// The model's training-time mel filterbank, reshaped to
    /// `[nMels, nFFT/2+1]` if loaded from safetensors. Pass this to
    /// `AudioPreprocessor.logMelSpectrogram(filterbankOverride:)` for
    /// byte-for-byte parity with NeMo's preprocessing.
    public var preprocessorFb: Tensor? {
        guard let stored = weights.preprocessorFb else { return nil }
        let shape = stored.shape
        if shape.count == 3 && shape[0] == 1 {
            return stored.reshape([shape[1], shape[2]])
        }
        return stored
    }

    // -------------------------------------------------------------
    // Init
    // -------------------------------------------------------------

    public init(config: Config = .cohereTranscribe2B, weights: Weights) {
        precondition(weights.encoderLayers.count == config.encoder.nLayers,
            "expected \(config.encoder.nLayers) encoder layer bundles, got \(weights.encoderLayers.count)")
        precondition(weights.decoderLayers.count == config.decoderLayers,
            "expected \(config.decoderLayers) decoder layer bundles, got \(weights.decoderLayers.count)")
        self.config = config
        self.weights = weights
        self.encoder = ConformerEncoder(
            config: config.encoder,
            subsamplingWeights: weights.encoderSubsampling,
            layerWeights: weights.encoderLayers
        )
        self.decoderLayers = weights.decoderLayers.map {
            DecoderLayer(config: config, weights: $0)
        }
        self.decoderPosTable = Self.makeFixedSinusoidalPosTable(
            maxLen: config.maxSeqLen, dim: config.decoderHidden
        )
    }

    // -------------------------------------------------------------
    // Encoder + projection + cross-attn cache prefill
    // -------------------------------------------------------------

    /// Run the encoder on the mel input and project the output onto
    /// the decoder hidden size. This is the K/V source for every
    /// decoder cross-attention; we cache it once and reuse it across
    /// all decoder steps.
    public func runEncoder(mel: Tensor) -> Tensor {
        let encOut = encoder.forward(mel: mel)             // [1, T', encDim]
        let bias = weights.encDecProjB.reshape([1, 1, config.encDecProjToDim])
        return encOut.matmul(weights.encDecProjW).add(bias) // [1, T', decDim]
    }

    /// Reset every decoder layer's per-inference state (self-attn
    /// KV caches and cross-attn encoder K/V).
    public func reset() {
        for layer in decoderLayers {
            layer.reset()
        }
    }

    /// Compute and cache cross-attn K/V for every decoder layer.
    /// Must be called once after each `runEncoder` before any
    /// `decoderStep`.
    public func setEncoderContext(mel: Tensor) {
        let encProj = runEncoder(mel: mel)
        for layer in decoderLayers {
            layer.prepareCrossKV(encoderOut: encProj, config: config)
        }
    }

    // -------------------------------------------------------------
    // Decoder step
    // -------------------------------------------------------------

    /// Run a single decoder step on `tokenId`, advancing every
    /// layer's self-attn cache by one position. Returns the predicted
    /// next-token id.
    public func decoderStep(tokenId: Int) throws -> Int {
        let D = config.decoderHidden
        let pos = decoderLayers[0].cacheLength

        // 1. Token + fixed sinusoidal position embedding, then a
        // LayerNorm — `TransformerDecoderEmbedding` in the modeling
        // code applies LN to the SUM of token and position embeddings
        // before handing off to the layer stack.
        let tokenEmb = weights.embedTokens.embed(tokens: [tokenId])  // [1, 1, D]
        let posRow = decoderPosTable.embed(tokens: [pos])             // [1, 1, D]
        var h = tokenEmb.add(posRow).layerNorm(
            gamma: weights.embedNormGamma,
            beta:  weights.embedNormBeta,
            eps:   config.layerNormEps
        )

        // 2. 8 decoder blocks.
        for layer in decoderLayers {
            h = try layer.step(h, config: config)
        }

        // 3. Final LayerNorm.
        let normed = h.layerNorm(
            gamma: weights.decoderFinalNormGamma,
            beta:  weights.decoderFinalNormBeta,
            eps:   config.layerNormEps
        )

        // 4. Token classifier head.
        let bias3 = weights.headBias.reshape([1, 1, config.vocabSize])
        let logits = normed.matmul(weights.headWeight).add(bias3)
        // Argmax — log_softmax is monotonic so we skip it.
        let arr = logits.toArray()
        var best = 0
        var bestVal = arr[0]
        for i in 1..<arr.count where arr[i] > bestVal {
            bestVal = arr[i]
            best = i
        }
        _ = D
        return best
    }

    /// Greedy autoregressive decoding given a forced prompt prefix.
    ///
    /// `promptTokens` is the full prefix (BOS + any forced tokens for
    /// language / task selection). The model is fed every prefix
    /// token in order; we discard each prediction except the last
    /// one, which becomes the first real generated token.
    public func generateGreedy(
        promptTokens: [Int],
        maxNewTokens: Int
    ) throws -> [Int] {
        precondition(!promptTokens.isEmpty,
            "promptTokens must contain at least BOS")
        var nextToken = -1
        for tok in promptTokens {
            nextToken = try decoderStep(tokenId: tok)
        }
        var generated: [Int] = []
        for _ in 0..<maxNewTokens {
            if nextToken == config.eosTokenId { break }
            generated.append(nextToken)
            nextToken = try decoderStep(tokenId: nextToken)
        }
        return generated
    }

    /// End-to-end transcription. Uses `cohereTranscribe` audio
    /// preprocessing (n_fft=512, n_mels=128, per-feature norm).
    public func transcribe(
        samples: [Float],
        tokenizer: Tokenizer,
        promptTokens: [Int],
        maxNewTokens: Int = 200
    ) throws -> String {
        reset()

        // Use the model's stored mel filterbank if available — gives
        // byte-for-byte parity with NeMo's training-time DSP. The
        // safetensors stores it as [1, nMels, nFreqs]; reshape to
        // [nMels, nFreqs] for the FFI.
        let fbOverride: Tensor?
        if let storedFb = weights.preprocessorFb {
            let shape = storedFb.shape
            if shape.count == 3 && shape[0] == 1 {
                fbOverride = storedFb.reshape([shape[1], shape[2]])
            } else {
                fbOverride = storedFb
            }
        } else {
            fbOverride = nil
        }
        let mel = AudioPreprocessor.logMelSpectrogram(
            samples: samples, config: .cohereTranscribe,
            filterbankOverride: fbOverride
        )
        let nMels = mel.shape[0]
        let nFrames = mel.shape[1]
        let melBatched = mel.reshape([1, nMels, nFrames])

        setEncoderContext(mel: melBatched)
        let tokenIds = try generateGreedy(
            promptTokens: promptTokens, maxNewTokens: maxNewTokens
        )
        return try tokenizer.decode(
            tokenIds.map { UInt32($0) },
            skipSpecialTokens: true
        )
    }

    // -------------------------------------------------------------
    // Fixed sinusoidal positional encoding (decoder).
    // -------------------------------------------------------------

    /// Build the `[maxLen, dim]` sinusoidal pos-encoding table used
    /// by the decoder. Standard transformer formula:
    ///
    ///     pe[pos, 2k]   = sin(pos / 10000^{2k/dim})
    ///     pe[pos, 2k+1] = cos(pos / 10000^{2k/dim})
    ///
    /// We make this a 2-D tensor (rather than 3-D like the encoder's
    /// rel-pos table) so we can use `embed(tokens:)` for fast
    /// row-by-position lookup at each decoder step.
    public static func makeFixedSinusoidalPosTable(
        maxLen: Int, dim: Int
    ) -> Tensor {
        precondition(dim % 2 == 0, "dim must be even, got \(dim)")
        var data = [Float](repeating: 0, count: maxLen * dim)
        for pos in 0..<maxLen {
            for k in 0..<(dim / 2) {
                let exponent = Float(2 * k) / Float(dim)
                let invFreq = expf(-logf(10_000) * exponent)
                let theta = Float(pos) * invFreq
                data[pos * dim + 2 * k]     = sinf(theta)
                data[pos * dim + 2 * k + 1] = cosf(theta)
            }
        }
        return try! Tensor.from(data: data, shape: [maxLen, dim])
    }

    // ===========================================================================
    // DecoderLayer — one block of the lightweight transformer decoder.
    // ===========================================================================

    /// One Cohere decoder block: causal self-attention (with KV cache)
    /// + cross-attention to encoder (with cached encoder K/V) + FFN
    /// (ReLU, not GELU). Pre-LN throughout. The class owns its own
    /// `KVCache` and the cached encoder K/V tensors.
    public final class DecoderLayer {

        public struct Weights {
            // Self-attention
            public let saNormGamma, saNormBeta: Tensor
            public let saWQ, saBQ: Tensor
            public let saWK, saBK: Tensor
            public let saWV, saBV: Tensor
            public let saWO, saBO: Tensor

            // Cross-attention
            public let caNormGamma, caNormBeta: Tensor
            public let caWQ, caBQ: Tensor
            public let caWK, caBK: Tensor
            public let caWV, caBV: Tensor
            public let caWO, caBO: Tensor

            // FFN (pre-LN, ReLU activation between two Linears)
            public let ffnNormGamma, ffnNormBeta: Tensor
            public let ffnUpW, ffnUpB: Tensor
            public let ffnDownW, ffnDownB: Tensor

            public init(
                saNormGamma: Tensor, saNormBeta: Tensor,
                saWQ: Tensor, saBQ: Tensor,
                saWK: Tensor, saBK: Tensor,
                saWV: Tensor, saBV: Tensor,
                saWO: Tensor, saBO: Tensor,
                caNormGamma: Tensor, caNormBeta: Tensor,
                caWQ: Tensor, caBQ: Tensor,
                caWK: Tensor, caBK: Tensor,
                caWV: Tensor, caBV: Tensor,
                caWO: Tensor, caBO: Tensor,
                ffnNormGamma: Tensor, ffnNormBeta: Tensor,
                ffnUpW: Tensor, ffnUpB: Tensor,
                ffnDownW: Tensor, ffnDownB: Tensor
            ) {
                self.saNormGamma = saNormGamma; self.saNormBeta = saNormBeta
                self.saWQ = saWQ; self.saBQ = saBQ
                self.saWK = saWK; self.saBK = saBK
                self.saWV = saWV; self.saBV = saBV
                self.saWO = saWO; self.saBO = saBO
                self.caNormGamma = caNormGamma; self.caNormBeta = caNormBeta
                self.caWQ = caWQ; self.caBQ = caBQ
                self.caWK = caWK; self.caBK = caBK
                self.caWV = caWV; self.caBV = caBV
                self.caWO = caWO; self.caBO = caBO
                self.ffnNormGamma = ffnNormGamma; self.ffnNormBeta = ffnNormBeta
                self.ffnUpW = ffnUpW; self.ffnUpB = ffnUpB
                self.ffnDownW = ffnDownW; self.ffnDownB = ffnDownB
            }
        }

        let weights: Weights
        let saCache: KVCache
        var caEncK: Tensor?
        var caEncV: Tensor?

        var cacheLength: Int { saCache.length }

        init(config: Config, weights: Weights) {
            self.weights = weights
            self.saCache = KVCache(config: .init(
                batchSize: 1,
                numHeads: config.decoderHeads,
                headDim: config.decoderHeadDim,
                maxSeqLen: config.maxSeqLen
            ))
        }

        func reset() {
            saCache.reset()
            caEncK = nil
            caEncV = nil
        }

        /// Compute and cache cross-attn K/V from encoder output.
        /// Stored as `[H, S_enc, D_h]` to match cross-attn layout.
        func prepareCrossKV(encoderOut: Tensor, config: Config) {
            let D = config.decoderHidden
            let H = config.decoderHeads
            let Dh = config.decoderHeadDim
            let Senc = encoderOut.shape[1]

            let bK3 = weights.caBK.reshape([1, 1, D])
            let bV3 = weights.caBV.reshape([1, 1, D])
            let k = encoderOut.matmul(weights.caWK).add(bK3)
            let v = encoderOut.matmul(weights.caWV).add(bV3)

            func splitHeads(_ t: Tensor) -> Tensor {
                t.reshape([1, Senc, H, Dh])
                 .permute([0, 2, 1, 3]).contiguous()
                 .reshape([H, Senc, Dh])
            }
            self.caEncK = splitHeads(k)
            self.caEncV = splitHeads(v)
        }

        /// One decoder step. `x` is the just-embedded token (shape
        /// `[1, 1, D]`). Updates `saCache` and returns the layer's
        /// hidden state.
        func step(_ x: Tensor, config: Config) throws -> Tensor {
            let D = config.decoderHidden
            let H = config.decoderHeads
            let Dh = config.decoderHeadDim
            guard let encK = caEncK, let encV = caEncV else {
                preconditionFailure("CohereTranscribe.DecoderLayer.step: encoder K/V not prepared")
            }

            // ---- Sub-layer 1: causal self-attention ----------------
            let saNormed = x.layerNorm(
                gamma: weights.saNormGamma, beta: weights.saNormBeta,
                eps: config.layerNormEps
            )
            let saBQ3 = weights.saBQ.reshape([1, 1, D])
            let saBK3 = weights.saBK.reshape([1, 1, D])
            let saBV3 = weights.saBV.reshape([1, 1, D])
            let saBO3 = weights.saBO.reshape([1, 1, D])
            let q  = saNormed.matmul(weights.saWQ).add(saBQ3)
            let k  = saNormed.matmul(weights.saWK).add(saBK3)
            let v  = saNormed.matmul(weights.saWV).add(saBV3)

            // Split heads to KVCache shape [1, H, 1, Dh].
            func splitHeads4D(_ t: Tensor) -> Tensor {
                t.reshape([1, 1, H, Dh])
                 .permute([0, 2, 1, 3]).contiguous()
            }
            let qH = splitHeads4D(q)
            let kH = splitHeads4D(k)
            let vH = splitHeads4D(v)
            let saAttn = try saCache.appendAndDecode(
                query: qH, key: kH, value: vH,
                scale: 1.0 / sqrtf(Float(Dh))
            )
            let saMerged = saAttn
                .permute([0, 2, 1, 3]).contiguous()
                .reshape([1, 1, D])
            let saOut = saMerged.matmul(weights.saWO).add(saBO3)
            let afterSA = x.add(saOut)

            // ---- Sub-layer 2: cross-attention to encoder K/V ------
            let caNormed = afterSA.layerNorm(
                gamma: weights.caNormGamma, beta: weights.caNormBeta,
                eps: config.layerNormEps
            )
            let caBQ3 = weights.caBQ.reshape([1, 1, D])
            let caBO3 = weights.caBO.reshape([1, 1, D])
            let qC = caNormed.matmul(weights.caWQ).add(caBQ3)
            // Split Q for cross: [1, 1, D] → [H, 1, Dh].
            let qCh = qC
                .reshape([1, 1, H, Dh])
                .permute([0, 2, 1, 3]).contiguous()
                .reshape([H, 1, Dh])
            let crossAttn = qCh.crossAttention(
                key: encK, value: encV,
                scale: 1.0 / sqrtf(Float(Dh))
            )
            let crossMerged = crossAttn
                .reshape([1, H, 1, Dh])
                .permute([0, 2, 1, 3]).contiguous()
                .reshape([1, 1, D])
            let caOut = crossMerged.matmul(weights.caWO).add(caBO3)
            let afterCA = afterSA.add(caOut)

            // ---- Sub-layer 3: FFN with ReLU activation ------------
            let ffnNormed = afterCA.layerNorm(
                gamma: weights.ffnNormGamma, beta: weights.ffnNormBeta,
                eps: config.layerNormEps
            )
            let ffnUpB3 = weights.ffnUpB.reshape([1, 1, config.decoderFFNDim])
            let ffnDownB3 = weights.ffnDownB.reshape([1, 1, D])
            let ffnOut = ffnNormed
                .matmul(weights.ffnUpW).add(ffnUpB3)
                .relu()                          // ← ReLU, not GELU
                .matmul(weights.ffnDownW).add(ffnDownB3)
            return afterCA.add(ffnOut)
        }
    }
}
