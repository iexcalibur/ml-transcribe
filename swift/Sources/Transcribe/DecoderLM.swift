import Foundation

/// A hand-coded N-layer decoder-only language model that can be
/// configured as either a GPT-style (LayerNorm + biases + GELU) or
/// LLaMA-style (RMSNorm + no biases + SwiGLU) variant. A single
/// generic class with three knobs covers both architectures.
///
/// Weight naming (HuggingFace convention, minus the optional `model.`
/// prefix that a future weight mapper will strip):
///
///   embedding                                            [V, D]
///   layers.{i}.norm1.weight                              [D]
///   layers.{i}.norm1.bias           (LayerNorm only)     [D]
///   layers.{i}.attn.{q,k,v,o}_proj.weight                [D, D]
///   layers.{i}.attn.{q,k,v,o}_proj.bias  (useBias only)  [1, 1, D]
///   layers.{i}.norm2.weight                              [D]
///   layers.{i}.norm2.bias           (LayerNorm only)     [D]
///   layers.{i}.ffn.gate_proj.weight    (SwiGLU only)     [D, F]
///   layers.{i}.ffn.up_proj.weight                        [D, F]
///   layers.{i}.ffn.up_proj.bias          (useBias only)  [1, 1, F]
///   layers.{i}.ffn.down_proj.weight                      [F, D]
///   layers.{i}.ffn.down_proj.bias        (useBias only)  [1, 1, D]
///   final_norm.weight                                    [D]
///   final_norm.bias                 (LayerNorm only)     [D]
///   lm_head                                              [D, V]
public final class DecoderLM {

    /// Which normalization to use at each pre-norm site. RMSNorm has
    /// no bias tensor and doesn't subtract the mean.
    public enum NormType {
        /// GPT / BERT / original-transformer: y = γ·(x − μ)/σ + β.
        case layerNorm
        /// LLaMA / Mistral / most modern LMs: y = γ·x/√(mean(x²) + ε).
        case rmsNorm
    }

    /// Which feed-forward block to use.
    public enum FFNType {
        /// GPT-style: `y = GELU(x @ W_up + b_up) @ W_down + b_down`.
        case gelu
        /// LLaMA-style SwiGLU:
        /// `y = (silu(x @ W_gate) * (x @ W_up)) @ W_down`. Requires
        /// three projections per layer and never uses up-proj biases.
        case swiGLU
    }

    public struct Config {
        public let vocabSize: Int
        public let modelDim: Int
        public let numHeads: Int
        /// Number of key/value heads. For plain multi-head attention
        /// (MHA), this equals `numHeads`. For Grouped Query Attention
        /// (GQA, LLaMA-2-70B+, LLaMA-3, TinyLlama, SmolLM, Mistral,
        /// Cohere), it's smaller — each K/V head serves a group of
        /// `numHeads / numKVHeads` Q heads.
        public let numKVHeads: Int
        public let ffnDim: Int
        public let numLayers: Int
        public let maxSeqLen: Int
        public let ropeBase: Float
        public let normEps: Float
        public let normType: NormType
        public let useBias: Bool
        public let ffnType: FFNType

        public var headDim: Int { modelDim / numHeads }

        /// Grouping factor: how many Q heads share each K/V head.
        /// 1 means plain MHA, > 1 means GQA.
        public var gqaGroupSize: Int { numHeads / numKVHeads }

        public init(
            vocabSize: Int,
            modelDim: Int,
            numHeads: Int,
            numKVHeads: Int? = nil,
            ffnDim: Int,
            numLayers: Int = 1,
            maxSeqLen: Int,
            ropeBase: Float = 10_000,
            normEps: Float = 1e-5,
            normType: NormType = .layerNorm,
            useBias: Bool = true,
            ffnType: FFNType = .gelu
        ) {
            precondition(modelDim % numHeads == 0,
                "modelDim (\(modelDim)) must be divisible by numHeads (\(numHeads))")
            precondition(numLayers >= 1, "numLayers must be >= 1")
            let kv = numKVHeads ?? numHeads
            precondition(numHeads % kv == 0,
                "numHeads (\(numHeads)) must be divisible by numKVHeads (\(kv))")
            self.vocabSize = vocabSize
            self.modelDim = modelDim
            self.numHeads = numHeads
            self.numKVHeads = kv
            self.ffnDim = ffnDim
            self.numLayers = numLayers
            self.maxSeqLen = maxSeqLen
            self.ropeBase = ropeBase
            self.normEps = normEps
            self.normType = normType
            self.useBias = useBias
            self.ffnType = ffnType
        }

        /// GPT / BERT / TinyGPT style: LayerNorm + biases + GELU.
        /// This is the shape of models from before ~2022. MHA by default.
        public static func gptStyle(
            vocabSize: Int, modelDim: Int, numHeads: Int, ffnDim: Int,
            numLayers: Int, maxSeqLen: Int,
            numKVHeads: Int? = nil,
            ropeBase: Float = 10_000,
            normEps: Float = 1e-5
        ) -> Config {
            Config(
                vocabSize: vocabSize, modelDim: modelDim, numHeads: numHeads,
                numKVHeads: numKVHeads,
                ffnDim: ffnDim, numLayers: numLayers, maxSeqLen: maxSeqLen,
                ropeBase: ropeBase, normEps: normEps,
                normType: .layerNorm, useBias: true, ffnType: .gelu
            )
        }

        /// LLaMA / Mistral / Cohere-family style: RMSNorm + no biases
        /// + SwiGLU. Smaller `normEps` (1e-6) is typical. Pass
        /// `numKVHeads` explicitly for GQA models (TinyLlama uses 4,
        /// LLaMA-3-8B uses 8, etc.); omit for MHA.
        public static func llamaStyle(
            vocabSize: Int, modelDim: Int, numHeads: Int, ffnDim: Int,
            numLayers: Int, maxSeqLen: Int,
            numKVHeads: Int? = nil,
            ropeBase: Float = 10_000,
            normEps: Float = 1e-6
        ) -> Config {
            Config(
                vocabSize: vocabSize, modelDim: modelDim, numHeads: numHeads,
                numKVHeads: numKVHeads,
                ffnDim: ffnDim, numLayers: numLayers, maxSeqLen: maxSeqLen,
                ropeBase: ropeBase, normEps: normEps,
                normType: .rmsNorm, useBias: false, ffnType: .swiGLU
            )
        }
    }

    public enum LoadError: Error, CustomStringConvertible {
        case missing(String)
        case unexpectedShape(name: String, expected: [Int], got: [Int])

        public var description: String {
            switch self {
            case .missing(let n):
                return "missing weight: \(n)"
            case .unexpectedShape(let n, let e, let g):
                return "weight '\(n)': expected \(e), got \(g)"
            }
        }
    }

    public let config: Config

    private let embedding: Tensor
    private let finalNormWeight: Tensor
    private let finalNormBias: Tensor?    // nil for RMSNorm
    private let lmHead: Tensor

    private let layers: [Layer]

    // -----------------------------------------------------------------------
    // Layer: one transformer block with its own weights and KV cache.
    // -----------------------------------------------------------------------
    private final class Layer {
        // Norm weights. beta tensors are nil iff the norm is RMSNorm.
        let norm1Weight: Tensor
        let norm1Bias: Tensor?
        let norm2Weight: Tensor
        let norm2Bias: Tensor?

        // Attention. Biases are nil iff useBias is false.
        let wQ, wK, wV, wO: Tensor
        let bQ, bK, bV, bO: Tensor?

        // FFN. gate is present iff ffnType is SwiGLU.
        // up/down biases are nil iff useBias is false (SwiGLU never has gate bias).
        let wGate: Tensor?
        let wUp, wDown: Tensor
        let bUp, bDown: Tensor?

        let cache: KVCache
        let config: Config

        init(weights: WeightSource, prefix: String, config: Config) throws {
            let D = config.modelDim
            let F = config.ffnDim
            // GQA: K/V projections produce `numKVHeads * headDim`
            // features, typically smaller than `D`. For MHA this equals D.
            let Dkv = config.numKVHeads * config.headDim

            func load(_ name: String, expecting shape: [Int]) throws -> Tensor {
                let full = prefix + name
                guard let t = weights[full] else { throw LoadError.missing(full) }
                if t.shape != shape {
                    throw LoadError.unexpectedShape(
                        name: full, expected: shape, got: t.shape)
                }
                return t
            }
            func loadOptional(_ name: String, expecting shape: [Int]) throws -> Tensor? {
                let full = prefix + name
                guard let t = weights[full] else { return nil }
                if t.shape != shape {
                    throw LoadError.unexpectedShape(
                        name: full, expected: shape, got: t.shape)
                }
                return t
            }

            // Norms.
            self.norm1Weight = try load("norm1.weight", expecting: [D])
            self.norm1Bias = config.normType == .layerNorm
                ? try load("norm1.bias", expecting: [D]) : nil

            self.norm2Weight = try load("norm2.weight", expecting: [D])
            self.norm2Bias = config.normType == .layerNorm
                ? try load("norm2.bias", expecting: [D]) : nil

            // Attention. Q and O project full [D, D]; K and V project
            // the (possibly smaller) [D, Dkv] under GQA.
            self.wQ = try load("attn.q_proj.weight", expecting: [D, D])
            self.wK = try load("attn.k_proj.weight", expecting: [D, Dkv])
            self.wV = try load("attn.v_proj.weight", expecting: [D, Dkv])
            self.wO = try load("attn.o_proj.weight", expecting: [D, D])
            if config.useBias {
                self.bQ = try load("attn.q_proj.bias", expecting: [1, 1, D])
                self.bK = try load("attn.k_proj.bias", expecting: [1, 1, Dkv])
                self.bV = try load("attn.v_proj.bias", expecting: [1, 1, Dkv])
                self.bO = try load("attn.o_proj.bias", expecting: [1, 1, D])
            } else {
                self.bQ = nil
                self.bK = nil
                self.bV = nil
                self.bO = nil
            }

            // FFN.
            self.wGate = config.ffnType == .swiGLU
                ? try load("ffn.gate_proj.weight", expecting: [D, F]) : nil
            self.wUp = try load("ffn.up_proj.weight", expecting: [D, F])
            self.wDown = try load("ffn.down_proj.weight", expecting: [F, D])
            if config.useBias {
                self.bUp = try load("ffn.up_proj.bias", expecting: [1, 1, F])
                self.bDown = try load("ffn.down_proj.bias", expecting: [1, 1, D])
            } else {
                self.bUp = nil
                self.bDown = nil
            }
            _ = loadOptional  // silence unused-fn warning; kept for future knobs.

            self.config = config
            // Cache stores the expanded (Hq-wide) K/V, because we
            // expand BEFORE calling appendAndDecode. A smarter future
            // version would store Hkv-wide and expand post-cache to
            // save cache memory by factor G.
            self.cache = KVCache(config: .init(
                batchSize: 1,
                numHeads: config.numHeads,
                headDim: config.headDim,
                maxSeqLen: config.maxSeqLen
            ))
        }

        // Helpers that dispatch on config.

        private func norm(_ x: Tensor, weight: Tensor, bias: Tensor?) -> Tensor {
            switch config.normType {
            case .layerNorm:
                // We verified in init that bias is non-nil for LayerNorm.
                return x.layerNorm(gamma: weight, beta: bias!, eps: config.normEps)
            case .rmsNorm:
                return x.rmsNorm(gamma: weight, eps: config.normEps)
            }
        }

        private func addOpt(_ x: Tensor, _ bias: Tensor?) -> Tensor {
            if let b = bias { return x.add(b) }
            return x
        }

        /// One decoding step for this layer. Input/output shape `[1, 1, D]`.
        /// Appends exactly one entry to this layer's KV cache.
        func step(_ x: Tensor) throws -> Tensor {
            let D = config.modelDim
            let Hq = config.numHeads
            let Hkv = config.numKVHeads
            let G = config.gqaGroupSize     // = Hq / Hkv, 1 for MHA
            let Dh = config.headDim
            let pos = cache.length

            // Pre-norm + Q/K/V projections (biases optional).
            // Q is [1, 1, D]; K, V are [1, 1, Hkv*Dh] (may be smaller under GQA).
            let normed1 = norm(x, weight: norm1Weight, bias: norm1Bias)
            let q = addOpt(normed1.matmul(wQ), bQ)
            let k = addOpt(normed1.matmul(wK), bK)
            let v = addOpt(normed1.matmul(wV), bV)

            // Split heads (different head counts for Q vs K/V under GQA).
            func splitHeads(_ t: Tensor, heads: Int) -> Tensor {
                t.reshape([1, 1, heads, Dh])
                 .permute([0, 2, 1, 3])
                 .contiguous()
            }
            let qH = splitHeads(q, heads: Hq).rope(startPos: pos, base: config.ropeBase)
            let kHkv = splitHeads(k, heads: Hkv).rope(startPos: pos, base: config.ropeBase)
            let vHkv = splitHeads(v, heads: Hkv)

            // For GQA, broadcast each of Hkv K/V heads to G consecutive
            // Q heads so cache.appendAndDecode sees matched shapes.
            // For MHA (G == 1) we skip the (no-op) expansion.
            let kH: Tensor
            let vH: Tensor
            if G == 1 {
                kH = kHkv
                vH = vHkv
            } else {
                kH = kHkv.repeatInterleave(dim: 1, repeats: G)
                vH = vHkv.repeatInterleave(dim: 1, repeats: G)
            }

            let attended = try cache.appendAndDecode(
                query: qH, key: kH, value: vH,
                scale: 1.0 / sqrt(Float(Dh))
            )
            let merged = attended
                .permute([0, 2, 1, 3]).contiguous()
                .reshape([1, 1, D])
            let attnOut = addOpt(merged.matmul(wO), bO)

            let afterAttn = x.add(attnOut)

            // Pre-norm + FFN + residual.
            let normed2 = norm(afterAttn, weight: norm2Weight, bias: norm2Bias)
            let ffnOut: Tensor
            switch config.ffnType {
            case .gelu:
                // GELU(normed @ W_up [+ b_up]) @ W_down [+ b_down]
                let up = addOpt(normed2.matmul(wUp), bUp)
                let act = up.gelu()
                ffnOut = addOpt(act.matmul(wDown), bDown)
            case .swiGLU:
                // (silu(normed @ W_gate) * (normed @ W_up)) @ W_down
                // SwiGLU doesn't use the up_proj bias even when useBias=true;
                // that's a real LLaMA idiosyncrasy. We treat `bUp` as ignored.
                guard let wGate = wGate else {
                    preconditionFailure("SwiGLU: missing gate_proj")
                }
                let gate = normed2.matmul(wGate).silu()
                let up = normed2.matmul(wUp)
                let gated = gate.mul(up)
                ffnOut = addOpt(gated.matmul(wDown), bDown)
            }
            return afterAttn.add(ffnOut)
        }
    }

    public init(weights: WeightSource, config: Config) throws {
        self.config = config
        let D = config.modelDim
        let V = config.vocabSize

        func load(_ name: String, expecting shape: [Int]) throws -> Tensor {
            guard let t = weights[name] else { throw LoadError.missing(name) }
            if t.shape != shape {
                throw LoadError.unexpectedShape(
                    name: name, expected: shape, got: t.shape)
            }
            return t
        }

        self.embedding = try load("embedding", expecting: [V, D])
        self.finalNormWeight = try load("final_norm.weight", expecting: [D])
        self.finalNormBias = config.normType == .layerNorm
            ? try load("final_norm.bias", expecting: [D]) : nil

        // Tied embeddings: if the model file doesn't ship an explicit
        // `lm_head`, derive it from `embedding.transpose()`. GPT-2 and
        // LLaMA both use this trick to halve their parameter count.
        //
        // The transpose creates an owned tensor (its data is a fresh
        // contiguous copy, independent of `embedding`'s borrow), so
        // `lmHead` is safe to keep for this LM's lifetime even if the
        // source weights are dropped.
        if let explicit = weights["lm_head"] {
            guard explicit.shape == [D, V] else {
                throw LoadError.unexpectedShape(
                    name: "lm_head", expected: [D, V], got: explicit.shape)
            }
            self.lmHead = explicit
        } else {
            // [V, D] -> [D, V] via transpose.
            self.lmHead = self.embedding.transpose(0, 1).contiguous()
        }

        var loaded: [Layer] = []
        loaded.reserveCapacity(config.numLayers)
        for i in 0..<config.numLayers {
            loaded.append(try Layer(
                weights: weights, prefix: "layers.\(i).", config: config
            ))
        }
        self.layers = loaded
    }

    public func reset() {
        for layer in layers { layer.cache.reset() }
    }

    public var currentPosition: Int { layers[0].cache.length }

    public func step(tokenId: Int) throws -> Int {
        var h = embedding.embed(tokens: [tokenId])
        for layer in layers {
            h = try layer.step(h)
        }
        let final: Tensor
        switch config.normType {
        case .layerNorm:
            final = h.layerNorm(
                gamma: finalNormWeight, beta: finalNormBias!, eps: config.normEps
            )
        case .rmsNorm:
            final = h.rmsNorm(gamma: finalNormWeight, eps: config.normEps)
        }
        let logits = final.matmul(lmHead)
        let arr = logits.toArray()
        var best = 0
        var bestVal = arr[0]
        for i in 1..<arr.count where arr[i] > bestVal {
            bestVal = arr[i]
            best = i
        }
        return best
    }

    public func generate(prompt: [Int], maxNewTokens: Int) throws -> [Int] {
        reset()
        var lastPrediction = 0
        for token in prompt {
            lastPrediction = try step(tokenId: token)
        }
        var generated: [Int] = []
        var nextToken = lastPrediction
        for _ in 0..<maxNewTokens {
            generated.append(nextToken)
            nextToken = try step(tokenId: nextToken)
        }
        return generated
    }
}
