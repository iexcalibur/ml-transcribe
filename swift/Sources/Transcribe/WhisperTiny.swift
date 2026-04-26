import Foundation

/// End-to-end Whisper-tiny inference: audio → transcribed text.
///
/// Whisper is an encoder-decoder transformer:
///   - Encoder ingests an 80-mel-bin log-spectrogram, runs it through
///     two strided 1-D convolutions (subsampling 2×), adds sinusoidal
///     positional embeddings, then 4 transformer blocks.
///   - Decoder ingests text tokens, embeds them, adds learned
///     positional embeddings, then 4 blocks each containing causal
///     self-attention, cross-attention to encoder output, and an FFN.
///   - Greedy decoding: argmax of the logits, append, repeat until
///     EOS or max_length.
///
/// This class wires the entire pipeline using the primitives in
/// `Transcribe`: the `AudioPreprocessor` for log-mel, `Tensor.conv1d`
/// for the conv frontend, `Tensor.attention` for self-attention,
/// `Tensor.crossAttention` for encoder-attention, `KVCache` for the
/// decoder's incremental cached generation.
///
/// Weight naming follows HuggingFace `openai/whisper-tiny.en`:
/// `model.encoder.*`, `model.decoder.*`, with the standard Whisper
/// quirk that `k_proj` projections never have a bias (Q / V / O do).
///
/// One caveat: Whisper's tied lm_head (uses `embed_tokens.T`) is
/// handled internally — we do `decoder_hidden @ embed_tokens.T`
/// rather than loading a separate lm_head weight.
public final class WhisperTiny {

    public struct Config {
        public let nMels: Int
        public let dModel: Int
        public let encoderLayers: Int
        public let encoderHeads: Int
        public let encoderFFNDim: Int
        public let decoderLayers: Int
        public let decoderHeads: Int
        public let decoderFFNDim: Int
        public let maxSourcePositions: Int   // == 1500 for tiny.en
        public let maxTargetPositions: Int   // == 448
        public let vocabSize: Int
        public let bosTokenId: Int           // <|startoftranscript|>
        public let eosTokenId: Int           // <|endoftext|>
        public let layerNormEps: Float

        public var encoderHeadDim: Int { dModel / encoderHeads }
        public var decoderHeadDim: Int { dModel / decoderHeads }

        /// Defaults matching openai/whisper-tiny.en.
        public static let tinyEN = Config(
            nMels: 80,
            dModel: 384,
            encoderLayers: 4,
            encoderHeads: 6,
            encoderFFNDim: 1536,
            decoderLayers: 4,
            decoderHeads: 6,
            decoderFFNDim: 1536,
            maxSourcePositions: 1500,
            maxTargetPositions: 448,
            vocabSize: 51864,
            bosTokenId: 50257,
            eosTokenId: 50256,
            layerNormEps: 1e-5
        )
    }

    public enum LoadError: Error, CustomStringConvertible {
        case missing(String)
        case unexpectedShape(name: String, expected: [Int], got: [Int])

        public var description: String {
            switch self {
            case .missing(let n): return "missing weight: \(n)"
            case .unexpectedShape(let n, let e, let g):
                return "weight '\(n)': expected \(e), got \(g)"
            }
        }
    }

    public let config: Config

    // -------------------------------------------------------------
    // Encoder weights
    // -------------------------------------------------------------
    private let encConv1Weight, encConv1Bias: Tensor
    private let encConv2Weight, encConv2Bias: Tensor
    private let encPosEmbed: Tensor
    private let encLayers: [EncoderLayer]
    private let encFinalNormGamma, encFinalNormBeta: Tensor

    // -------------------------------------------------------------
    // Decoder weights
    // -------------------------------------------------------------
    private let decEmbedTokens: Tensor
    private let decPosEmbed: Tensor
    private let decLayers: [DecoderLayer]
    private let decFinalNormGamma, decFinalNormBeta: Tensor

    // -------------------------------------------------------------
    // Encoder layer
    // -------------------------------------------------------------
    private final class EncoderLayer {
        let saNormGamma, saNormBeta: Tensor
        let wQ, bQ, wK, wV, bV, wO, bO: Tensor   // k_proj has no bias
        let ffnNormGamma, ffnNormBeta: Tensor
        let wUp, bUp, wDown, bDown: Tensor

        init(weights: WeightSource, prefix: String, config: Config) throws {
            let D = config.dModel
            let F = config.encoderFFNDim

            func load(_ name: String, expecting shape: [Int]) throws -> Tensor {
                let full = prefix + name
                guard let t = weights[full] else { throw LoadError.missing(full) }
                guard t.shape == shape else {
                    throw LoadError.unexpectedShape(
                        name: full, expected: shape, got: t.shape)
                }
                return t
            }

            self.saNormGamma = try load("self_attn_layer_norm.weight", expecting: [D])
            self.saNormBeta  = try load("self_attn_layer_norm.bias",   expecting: [D])

            self.wQ = try load("self_attn.q_proj.weight", expecting: [D, D])
            self.bQ = try load("self_attn.q_proj.bias",   expecting: [D])
            self.wK = try load("self_attn.k_proj.weight", expecting: [D, D])
            self.wV = try load("self_attn.v_proj.weight", expecting: [D, D])
            self.bV = try load("self_attn.v_proj.bias",   expecting: [D])
            self.wO = try load("self_attn.out_proj.weight", expecting: [D, D])
            self.bO = try load("self_attn.out_proj.bias",   expecting: [D])

            self.ffnNormGamma = try load("final_layer_norm.weight", expecting: [D])
            self.ffnNormBeta  = try load("final_layer_norm.bias",   expecting: [D])

            self.wUp   = try load("fc1.weight", expecting: [D, F])
            self.bUp   = try load("fc1.bias",   expecting: [F])
            self.wDown = try load("fc2.weight", expecting: [F, D])
            self.bDown = try load("fc2.bias",   expecting: [D])
        }

        /// One encoder block forward. Input/output: `[1, S, D]`.
        /// Self-attention here is bidirectional (no causal mask).
        func forward(_ x: Tensor, config: Config) -> Tensor {
            let D = config.dModel
            let H = config.encoderHeads
            let Dh = config.encoderHeadDim
            let S = x.shape[1]

            // Pre-norm + Q/K/V projections.
            let normed = x.layerNorm(
                gamma: saNormGamma, beta: saNormBeta, eps: config.layerNormEps
            )
            // Bias is shape [D]; reshape to [1, 1, D] for broadcasting against
            // [1, S, D] matmul output.
            let bQ3 = bQ.reshape([1, 1, D])
            let bV3 = bV.reshape([1, 1, D])
            let bO3 = bO.reshape([1, 1, D])
            let q = normed.matmul(wQ).add(bQ3)
            let k = normed.matmul(wK)              // no bias
            let v = normed.matmul(wV).add(bV3)

            // Split heads: [1, S, D] → [1, S, H, Dh] → [1, H, S, Dh] → [H, S, Dh].
            func splitHeads(_ t: Tensor) -> Tensor {
                t.reshape([1, S, H, Dh])
                 .permute([0, 2, 1, 3]).contiguous()
                 .reshape([H, S, Dh])
            }

            let attended = splitHeads(q).attention(
                key: splitHeads(k), value: splitHeads(v),
                scale: 1.0 / sqrtf(Float(Dh)),
                causal: false
            )
            // Merge: [H, S, Dh] → [1, H, S, Dh] → [1, S, H, Dh] → [1, S, D].
            let merged = attended
                .reshape([1, H, S, Dh])
                .permute([0, 2, 1, 3]).contiguous()
                .reshape([1, S, D])
            let attnOut = merged.matmul(wO).add(bO3)
            let afterAttn = x.add(attnOut)

            // Pre-norm + FFN + residual.
            let normed2 = afterAttn.layerNorm(
                gamma: ffnNormGamma, beta: ffnNormBeta, eps: config.layerNormEps
            )
            let bUp3 = bUp.reshape([1, 1, config.encoderFFNDim])
            let bDown3 = bDown.reshape([1, 1, D])
            let ffnOut = normed2
                .matmul(wUp).add(bUp3)
                .gelu()
                .matmul(wDown).add(bDown3)
            return afterAttn.add(ffnOut)
        }
    }

    // -------------------------------------------------------------
    // Decoder layer
    // -------------------------------------------------------------
    private final class DecoderLayer {
        // self-attention
        let saNormGamma, saNormBeta: Tensor
        let saWQ, saBQ, saWK, saWV, saBV, saWO, saBO: Tensor
        let saCache: KVCache
        // cross-attention
        let caNormGamma, caNormBeta: Tensor
        let caWQ, caBQ, caWK, caWV, caBV, caWO, caBO: Tensor
        // cached encoder K/V (computed once per inference)
        var caEncK: Tensor?
        var caEncV: Tensor?
        // FFN
        let ffnNormGamma, ffnNormBeta: Tensor
        let wUp, bUp, wDown, bDown: Tensor

        init(weights: WeightSource, prefix: String, config: Config) throws {
            let D = config.dModel
            let F = config.decoderFFNDim

            func load(_ name: String, expecting shape: [Int]) throws -> Tensor {
                let full = prefix + name
                guard let t = weights[full] else { throw LoadError.missing(full) }
                guard t.shape == shape else {
                    throw LoadError.unexpectedShape(
                        name: full, expected: shape, got: t.shape)
                }
                return t
            }

            self.saNormGamma = try load("self_attn_layer_norm.weight", expecting: [D])
            self.saNormBeta  = try load("self_attn_layer_norm.bias",   expecting: [D])

            self.saWQ = try load("self_attn.q_proj.weight", expecting: [D, D])
            self.saBQ = try load("self_attn.q_proj.bias",   expecting: [D])
            self.saWK = try load("self_attn.k_proj.weight", expecting: [D, D])
            self.saWV = try load("self_attn.v_proj.weight", expecting: [D, D])
            self.saBV = try load("self_attn.v_proj.bias",   expecting: [D])
            self.saWO = try load("self_attn.out_proj.weight", expecting: [D, D])
            self.saBO = try load("self_attn.out_proj.bias",   expecting: [D])

            self.caNormGamma = try load("encoder_attn_layer_norm.weight", expecting: [D])
            self.caNormBeta  = try load("encoder_attn_layer_norm.bias",   expecting: [D])
            self.caWQ = try load("encoder_attn.q_proj.weight", expecting: [D, D])
            self.caBQ = try load("encoder_attn.q_proj.bias",   expecting: [D])
            self.caWK = try load("encoder_attn.k_proj.weight", expecting: [D, D])
            self.caWV = try load("encoder_attn.v_proj.weight", expecting: [D, D])
            self.caBV = try load("encoder_attn.v_proj.bias",   expecting: [D])
            self.caWO = try load("encoder_attn.out_proj.weight", expecting: [D, D])
            self.caBO = try load("encoder_attn.out_proj.bias",   expecting: [D])

            self.ffnNormGamma = try load("final_layer_norm.weight", expecting: [D])
            self.ffnNormBeta  = try load("final_layer_norm.bias",   expecting: [D])

            self.wUp   = try load("fc1.weight", expecting: [D, F])
            self.bUp   = try load("fc1.bias",   expecting: [F])
            self.wDown = try load("fc2.weight", expecting: [F, D])
            self.bDown = try load("fc2.bias",   expecting: [D])

            self.saCache = KVCache(config: .init(
                batchSize: 1,
                numHeads: config.decoderHeads,
                headDim: config.decoderHeadDim,
                maxSeqLen: config.maxTargetPositions
            ))
        }

        /// Compute and cache K, V from encoder output. Called once per
        /// inference (when the encoder runs); thereafter every decoding
        /// step reuses these. Stored as `[H, S_enc, Dh]` so they
        /// match cross-attention's [BH, S, D] expectation.
        func prepareCrossKV(encoderOut: Tensor, config: Config) {
            let D = config.dModel
            let H = config.decoderHeads
            let Dh = config.decoderHeadDim
            let Senc = encoderOut.shape[1]

            let bV3 = caBV.reshape([1, 1, D])
            let k = encoderOut.matmul(caWK)        // [1, Senc, D] (k_proj has no bias)
            let v = encoderOut.matmul(caWV).add(bV3)

            func splitHeads(_ t: Tensor) -> Tensor {
                t.reshape([1, Senc, H, Dh])
                 .permute([0, 2, 1, 3]).contiguous()
                 .reshape([H, Senc, Dh])
            }
            self.caEncK = splitHeads(k)
            self.caEncV = splitHeads(v)
        }

        /// One decoding step. `x` is the just-embedded current token,
        /// shape `[1, 1, D]`. Output is the per-token hidden state.
        func step(_ x: Tensor, config: Config) throws -> Tensor {
            let D = config.dModel
            let H = config.decoderHeads
            let Dh = config.decoderHeadDim
            guard let encK = caEncK, let encV = caEncV else {
                preconditionFailure("DecoderLayer.step: encoder K/V not prepared")
            }

            // Sub-layer 1: pre-norm + causal self-attention with cache + residual.
            let saNormed = x.layerNorm(
                gamma: saNormGamma, beta: saNormBeta, eps: config.layerNormEps
            )
            let bQ3 = saBQ.reshape([1, 1, D])
            let bV3 = saBV.reshape([1, 1, D])
            let bO3 = saBO.reshape([1, 1, D])
            let q = saNormed.matmul(saWQ).add(bQ3)
            let k = saNormed.matmul(saWK)
            let v = saNormed.matmul(saWV).add(bV3)
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
            let saOut = saMerged.matmul(saWO).add(bO3)
            let afterSA = x.add(saOut)

            // Sub-layer 2: pre-norm + cross-attention to encoder K/V + residual.
            let caNormed = afterSA.layerNorm(
                gamma: caNormGamma, beta: caNormBeta, eps: config.layerNormEps
            )
            let caBQ3 = caBQ.reshape([1, 1, D])
            let caBO3 = caBO.reshape([1, 1, D])
            let qC = caNormed.matmul(caWQ).add(caBQ3)
            // Split Q for cross: [1, 1, D] → [H, 1, Dh].
            let qCh = qC
                .reshape([1, 1, H, Dh])
                .permute([0, 2, 1, 3]).contiguous()
                .reshape([H, 1, Dh])
            let crossAttn = qCh.crossAttention(
                key: encK, value: encV,
                scale: 1.0 / sqrtf(Float(Dh))
            )
            // Merge: [H, 1, Dh] → [1, H, 1, Dh] → [1, 1, H, Dh] → [1, 1, D].
            let crossMerged = crossAttn
                .reshape([1, H, 1, Dh])
                .permute([0, 2, 1, 3]).contiguous()
                .reshape([1, 1, D])
            let caOut = crossMerged.matmul(caWO).add(caBO3)
            let afterCA = afterSA.add(caOut)

            // Sub-layer 3: pre-norm + FFN + residual.
            let ffnNormed = afterCA.layerNorm(
                gamma: ffnNormGamma, beta: ffnNormBeta, eps: config.layerNormEps
            )
            let bUp3 = bUp.reshape([1, 1, config.decoderFFNDim])
            let bDown3 = bDown.reshape([1, 1, D])
            let ffnOut = ffnNormed
                .matmul(wUp).add(bUp3)
                .gelu()
                .matmul(wDown).add(bDown3)
            return afterCA.add(ffnOut)
        }
    }

    // -------------------------------------------------------------
    // Init
    // -------------------------------------------------------------
    public init(weights: WeightSource, config: Config = .tinyEN) throws {
        self.config = config
        let D = config.dModel
        let nMels = config.nMels
        let V = config.vocabSize

        func load(_ name: String, expecting shape: [Int]) throws -> Tensor {
            guard let t = weights[name] else { throw LoadError.missing(name) }
            guard t.shape == shape else {
                throw LoadError.unexpectedShape(
                    name: name, expected: shape, got: t.shape)
            }
            return t
        }

        // --- Encoder weights ---
        // Whisper conv weights: [out, in, k]. Our conv1d takes
        // [C_out, C_in, K] — same convention, no transpose needed.
        self.encConv1Weight = try load("model.encoder.conv1.weight", expecting: [D, nMels, 3])
        self.encConv1Bias   = try load("model.encoder.conv1.bias",   expecting: [D])
        self.encConv2Weight = try load("model.encoder.conv2.weight", expecting: [D, D, 3])
        self.encConv2Bias   = try load("model.encoder.conv2.bias",   expecting: [D])
        self.encPosEmbed    = try load("model.encoder.embed_positions.weight",
                                       expecting: [config.maxSourcePositions, D])
        self.encFinalNormGamma = try load("model.encoder.layer_norm.weight", expecting: [D])
        self.encFinalNormBeta  = try load("model.encoder.layer_norm.bias",   expecting: [D])

        var encoderLayers: [EncoderLayer] = []
        for i in 0..<config.encoderLayers {
            encoderLayers.append(try EncoderLayer(
                weights: weights,
                prefix: "model.encoder.layers.\(i).",
                config: config
            ))
        }
        self.encLayers = encoderLayers

        // --- Decoder weights ---
        self.decEmbedTokens = try load("model.decoder.embed_tokens.weight",
                                       expecting: [V, D])
        self.decPosEmbed    = try load("model.decoder.embed_positions.weight",
                                       expecting: [config.maxTargetPositions, D])
        self.decFinalNormGamma = try load("model.decoder.layer_norm.weight", expecting: [D])
        self.decFinalNormBeta  = try load("model.decoder.layer_norm.bias",   expecting: [D])

        var decoderLayers: [DecoderLayer] = []
        for i in 0..<config.decoderLayers {
            decoderLayers.append(try DecoderLayer(
                weights: weights,
                prefix: "model.decoder.layers.\(i).",
                config: config
            ))
        }
        self.decLayers = decoderLayers
    }

    // -------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------

    /// Crop a `[1, M, S]` tensor along the time axis to the first
    /// `maxFrames` frames. No-op if `S <= maxFrames`. Implemented
    /// via permute + embed-as-row-selector, avoiding any host-side
    /// array round-trip.
    private func trimFrames(mel: Tensor, maxFrames: Int) -> Tensor {
        let S = mel.shape[2]
        if S <= maxFrames { return mel }
        let M = mel.shape[1]
        // [1, M, S] → [1, S, M] → [S, M] (treat as a `[V=S, D=M]`
        // embedding table), gather rows 0..<maxFrames → [1, maxFrames,
        // M], then permute back to [1, M, maxFrames].
        let asRows = mel
            .permute([0, 2, 1]).contiguous()
            .reshape([S, M])
        let ids = (0..<maxFrames).map { Int($0) }
        return asRows.embed(tokens: ids)
            .permute([0, 2, 1]).contiguous()
    }

    // -------------------------------------------------------------
    // Forward passes
    // -------------------------------------------------------------

    /// Encode a log-mel spectrogram `[1, n_mels, n_frames]` into the
    /// hidden states `[1, n_frames/2, d_model]` after the strided
    /// conv subsampler and 4 transformer blocks.
    public func runEncoder(mel: Tensor) -> Tensor {
        let D = config.dModel
        // Whisper's reference impl uses STFT(center=True) and then
        // `[..., :-1]` to drop the trailing frame so a 30 sec clip
        // produces exactly 3000 frames. Our Rust log-mel keeps that
        // last frame, so a 30 sec clip arrives here as 3001. Trim to
        // `maxSourcePositions * 2` so the post-conv length doesn't
        // exceed the positional embedding table.
        let maxMelFrames = config.maxSourcePositions * 2     // 3000
        let trimmed = trimFrames(mel: mel, maxFrames: maxMelFrames)

        // 1. Conv1: [1, 80, S] → [1, 384, S], GELU.
        // 2. Conv2: stride 2 → [1, 384, S/2], GELU.
        // bias [384] → reshape to [1, 384, 1] for broadcast over length.
        let b1 = encConv1Bias.reshape([1, D, 1])
        let b2 = encConv2Bias.reshape([1, D, 1])

        let x1 = trimmed
            .conv1d(weight: encConv1Weight, stride: 1, padding: 1)
            .add(b1)
            .gelu()
        let x2 = x1
            .conv1d(weight: encConv2Weight, stride: 2, padding: 1)
            .add(b2)
            .gelu()

        // 3. Permute to [1, S, D] for attention; add positional embed.
        let S = x2.shape[2]
        let perm = x2
            .permute([0, 2, 1])
            .contiguous()                               // [1, S, D]
        // encPosEmbed is [maxSrcPos, D]; we just take the first S rows
        // by reshaping into a 3-D tensor of shape [1, max, D] and
        // doing index gather via embedding.
        let posIds = (0..<S).map { Int($0) }
        let pos = encPosEmbed.embed(tokens: posIds)     // [1, S, D]
        var h = perm.add(pos)

        // 4. 4 encoder blocks.
        for layer in encLayers {
            h = layer.forward(h, config: config)
        }
        // 5. Final LayerNorm.
        return h.layerNorm(
            gamma: encFinalNormGamma,
            beta:  encFinalNormBeta,
            eps:   config.layerNormEps
        )
    }

    /// Reset all per-inference state (decoder KV caches + cross-attn
    /// encoder K/V). Call between independent transcribe() calls.
    public func reset() {
        for layer in decLayers {
            layer.saCache.reset()
            layer.caEncK = nil
            layer.caEncV = nil
        }
    }

    /// Run the encoder and prefill every decoder layer's cross-attn
    /// K/V cache. Must be called before any `decoderStep()`.
    public func setEncoderContext(mel: Tensor) {
        let encOut = runEncoder(mel: mel)
        for layer in decLayers {
            layer.prepareCrossKV(encoderOut: encOut, config: config)
        }
    }

    /// Process one decoder token and return the predicted next-token id.
    /// Updates each decoder layer's self-attention KV cache.
    public func decoderStep(tokenId: Int) throws -> Int {
        let pos = decLayers[0].saCache.length     // current position
        // 1. Token + positional embedding.
        let tokenEmb = decEmbedTokens.embed(tokens: [tokenId])     // [1, 1, D]
        let posEmb = decPosEmbed.embed(tokens: [pos])              // [1, 1, D]
        var h = tokenEmb.add(posEmb)

        // 2. 4 decoder blocks.
        for layer in decLayers {
            h = try layer.step(h, config: config)
        }
        // 3. Final LayerNorm.
        let normed = h.layerNorm(
            gamma: decFinalNormGamma,
            beta:  decFinalNormBeta,
            eps:   config.layerNormEps
        )
        // 4. Tied LM head: logits = h @ embed_tokens.T.
        // embed_tokens is [V, D]; we want [V, D].T = [D, V] for the
        // matmul, but our matmul takes B @ W where W is [in, out]. We
        // can avoid an explicit transpose by computing
        //   logits = h.matmul(embed_tokens.T) — but we don't have a
        // transpose op exposed at full freedom. Instead, materialize
        // the transpose via permute+contiguous (cached on first call
        // would be ideal; for now we do it every step since whisper-
        // tiny's vocab×D is only 51864×384 = 75 MB, cheap).
        let lmHead = decEmbedTokens.transpose(0, 1).contiguous()   // [D, V]
        let logits = normed.matmul(lmHead)                          // [1, 1, V]
        let arr = logits.toArray()
        var best = 0
        var bestVal = arr[0]
        for i in 1..<arr.count where arr[i] > bestVal {
            bestVal = arr[i]
            best = i
        }
        return best
    }

    /// Greedy autoregressive decoding.
    ///
    /// `promptTokens` is the full forced-decoding prefix — including
    /// BOS. For `whisper-tiny.en` use `[bosTokenId, 50362]` so the
    /// model is anchored on `<|startoftranscript|><|notimestamps|>`,
    /// matching the official `forced_decoder_ids = [(1, 50362)]`.
    ///
    /// The model is fed every prefix token in order; we discard each
    /// prediction except the last one, which becomes the first real
    /// generated token. Decoding then continues greedily until EOS
    /// or `maxNewTokens`.
    ///
    /// Caller must have run `setEncoderContext` first. Returns just
    /// the generated tokens (excludes the prefix).
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

    /// End-to-end: audio samples → text. Convenience that combines
    /// preprocessing, encoder, decoder, and tokenizer decoding.
    ///
    /// Uses the standard `tiny.en` prefix `[BOS, <|notimestamps|>]` so
    /// the model emits plain text without timestamp markers. Pass a
    /// custom `promptTokens` for a different decoding prefix
    /// (e.g. multilingual or task-specific tokens).
    public func transcribe(
        samples: [Float],
        tokenizer: Tokenizer,
        maxNewTokens: Int = 200,
        promptTokens: [Int]? = nil
    ) throws -> String {
        reset()

        // 1. Audio → log-mel [1, 80, ~3000].
        let mel = AudioPreprocessor.logMelSpectrogram(samples: samples)
        let nMels = mel.shape[0]
        let nFrames = mel.shape[1]
        let melBatched = mel.reshape([1, nMels, nFrames])

        // 2. Run encoder + prefill cross-attn caches.
        setEncoderContext(mel: melBatched)

        // 3. Greedy decode. tiny.en expects `<|notimestamps|>` (50362)
        // forced at position 1 — see config.json `forced_decoder_ids`.
        let prefix = promptTokens ?? [config.bosTokenId, 50362]
        let tokenIds = try generateGreedy(
            promptTokens: prefix, maxNewTokens: maxNewTokens
        )

        // 4. Detokenize.
        return try tokenizer.decode(
            tokenIds.map { UInt32($0) },
            skipSpecialTokens: true
        )
    }
}
