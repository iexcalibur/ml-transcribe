import Foundation

/// A canonical pre-norm transformer block:
///
///   x' = x + MultiHeadAttention(LayerNorm(x))
///   y  = x' + FFN(LayerNorm(x'))
///
/// Where:
///   MHA(z) = Concat_h( softmax(Q_h K_h^T / √d) V_h ) @ W_o + b_o,
///            with Q_h/K_h/V_h produced by linear projections of z
///   FFN(z) = GELU(z @ W_up + b_up) @ W_down + b_down
///
/// Weight naming (safetensors keys) follows a light simplification of
/// the HuggingFace convention:
///
///   ln1.gamma        [D]
///   ln1.beta         [D]
///   attn.q_proj.weight   [D, D]     (in our [in, out] convention)
///   attn.q_proj.bias     [1, 1, D]
///   attn.k_proj.weight   [D, D]
///   attn.k_proj.bias     [1, 1, D]
///   attn.v_proj.weight   [D, D]
///   attn.v_proj.bias     [1, 1, D]
///   attn.o_proj.weight   [D, D]
///   attn.o_proj.bias     [1, 1, D]
///   ln2.gamma        [D]
///   ln2.beta         [D]
///   ffn.up_proj.weight    [D, F]      (F = ffnDim, typically 4*D)
///   ffn.up_proj.bias      [1, 1, F]
///   ffn.down_proj.weight  [F, D]
///   ffn.down_proj.bias    [1, 1, D]
///
/// Bias shape `[1, 1, D]` broadcasts against activations of shape
/// `[B, S, D]`. LayerNorm's gamma/beta are 1-D and broadcast along the
/// last dim internally (matching PyTorch's `nn.LayerNorm`).
public final class TransformerLayer {

    public struct Config {
        public let modelDim: Int
        public let numHeads: Int
        public let ffnDim: Int
        public let layerNormEps: Float

        public var headDim: Int { modelDim / numHeads }

        public init(modelDim: Int, numHeads: Int, ffnDim: Int, layerNormEps: Float = 1e-5) {
            precondition(modelDim % numHeads == 0,
                "modelDim (\(modelDim)) must be divisible by numHeads (\(numHeads))")
            self.modelDim = modelDim
            self.numHeads = numHeads
            self.ffnDim = ffnDim
            self.layerNormEps = layerNormEps
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

    // Attention weights.
    private let wQ, wK, wV, wO: Tensor
    private let bQ, bK, bV, bO: Tensor

    // FFN weights.
    private let wUp, wDown: Tensor
    private let bUp, bDown: Tensor

    // LayerNorm weights.
    private let ln1Gamma, ln1Beta: Tensor
    private let ln2Gamma, ln2Beta: Tensor

    public init(weights: SafetensorsWeights, config: Config) throws {
        self.config = config
        let D = config.modelDim
        let F = config.ffnDim

        func load(_ name: String, expecting expected: [Int]) throws -> Tensor {
            guard let t = weights[name] else { throw LoadError.missing(name) }
            if t.shape != expected {
                throw LoadError.unexpectedShape(name: name, expected: expected, got: t.shape)
            }
            return t
        }

        self.ln1Gamma = try load("ln1.gamma", expecting: [D])
        self.ln1Beta  = try load("ln1.beta",  expecting: [D])

        self.wQ = try load("attn.q_proj.weight", expecting: [D, D])
        self.bQ = try load("attn.q_proj.bias",   expecting: [1, 1, D])
        self.wK = try load("attn.k_proj.weight", expecting: [D, D])
        self.bK = try load("attn.k_proj.bias",   expecting: [1, 1, D])
        self.wV = try load("attn.v_proj.weight", expecting: [D, D])
        self.bV = try load("attn.v_proj.bias",   expecting: [1, 1, D])
        self.wO = try load("attn.o_proj.weight", expecting: [D, D])
        self.bO = try load("attn.o_proj.bias",   expecting: [1, 1, D])

        self.ln2Gamma = try load("ln2.gamma", expecting: [D])
        self.ln2Beta  = try load("ln2.beta",  expecting: [D])

        self.wUp   = try load("ffn.up_proj.weight",   expecting: [D, F])
        self.bUp   = try load("ffn.up_proj.bias",     expecting: [1, 1, F])
        self.wDown = try load("ffn.down_proj.weight", expecting: [F, D])
        self.bDown = try load("ffn.down_proj.bias",   expecting: [1, 1, D])
    }

    /// Forward pass. Input shape `[B, S, D]`; output shape matches.
    public func forward(_ x: Tensor) -> Tensor {
        let shape = x.shape
        precondition(shape.count == 3,
            "TransformerLayer expects [B, S, D] input; got \(shape)")
        precondition(shape[2] == config.modelDim,
            "input's last dim (\(shape[2])) != modelDim (\(config.modelDim))")

        let B = shape[0]
        let S = shape[1]
        let D = config.modelDim
        let H = config.numHeads
        let Dh = config.headDim

        // --- Sub-layer 1: pre-norm + self-attention + residual ---
        let normed1 = x.layerNorm(gamma: ln1Gamma, beta: ln1Beta, eps: config.layerNormEps)

        // Q, K, V projections: [B, S, D] @ [D, D] + [1, 1, D] → [B, S, D].
        let q = normed1.matmul(wQ).add(bQ)
        let k = normed1.matmul(wK).add(bK)
        let v = normed1.matmul(wV).add(bV)

        // Split heads: [B, S, D] → [B, S, H, Dh] → [B, H, S, Dh] → [B*H, S, Dh].
        func splitHeads(_ t: Tensor) -> Tensor {
            t.reshape([B, S, H, Dh])
             .permute([0, 2, 1, 3])
             .contiguous()
             .reshape([B * H, S, Dh])
        }

        let attended = splitHeads(q).attention(
            key: splitHeads(k),
            value: splitHeads(v),
            scale: 1.0 / sqrt(Float(Dh)),
            causal: false
        )
        // Merge heads: [B*H, S, Dh] → [B, H, S, Dh] → [B, S, H, Dh] → [B, S, D].
        let merged = attended
            .reshape([B, H, S, Dh])
            .permute([0, 2, 1, 3])
            .contiguous()
            .reshape([B, S, D])

        // Output projection and residual connection.
        let attnOut = merged.matmul(wO).add(bO)
        let afterAttn = x.add(attnOut)

        // --- Sub-layer 2: pre-norm + FFN + residual ---
        let normed2 = afterAttn.layerNorm(gamma: ln2Gamma, beta: ln2Beta, eps: config.layerNormEps)
        let ffnOut = normed2
            .matmul(wUp).add(bUp)   // [B, S, F]
            .gelu()
            .matmul(wDown).add(bDown) // [B, S, D]

        return afterAttn.add(ffnOut)
    }
}
