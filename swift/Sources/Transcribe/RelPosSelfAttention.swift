import Foundation

/// Conformer-style relative-positional multi-head self-attention
/// (Transformer-XL formulation).
///
/// Standard self-attention scores are `Q @ K.T / sqrt(d_h)`. With
/// relative positional encoding, the score becomes
///
///     ((Q + u) @ K.T  +  rel_shift((Q + v) @ P.T)) / sqrt(d_h)
///
/// where:
/// - `u` (= `posBiasU`) and `v` (= `posBiasV`) are learnable biases
///   added to Q before the two dot-products. Shape `[H, D_h]`.
/// - `P` is the projected sinusoidal relative-position embedding,
///   covering offsets in `[-(T-1), (T-1)]`. Shape `[1, 2T-1, D]`
///   before the per-head split.
/// - `rel_shift` realigns the position attention matrix from
///   relative-offset coordinates onto absolute key positions.
///
/// The encoder seq_len T is the same on the query and key sides
/// (full self-attention, no causal mask in the Conformer encoder).
///
/// Weights follow Cohere's HuggingFace layout, but already in our
/// `[in, out]` convention — load through `WeightMap` if the source
/// uses PyTorch's `[out, in]` Linear layout.
public final class RelPosSelfAttention {

    // -------------------------------------------------------------
    // Configuration & weights
    // -------------------------------------------------------------

    /// Model dim. Must be divisible by `numHeads`.
    public let dModel: Int
    public let numHeads: Int
    public var headDim: Int { dModel / numHeads }

    // Linear projections. Shapes assume our [in, out] convention.
    private let wQ, bQ: Tensor      // [D, D], [D]
    private let wK, bK: Tensor      // [D, D], [D]   (Cohere has bias on K)
    private let wV, bV: Tensor      // [D, D], [D]
    private let wPos: Tensor        // [D, D]        (no bias by Conformer convention)
    private let wOut, bOut: Tensor  // [D, D], [D]

    // Per-head learnable biases that condition the two dot-products.
    private let posBiasU: Tensor    // [H, D_h]
    private let posBiasV: Tensor    // [H, D_h]

    // -------------------------------------------------------------
    // Init
    // -------------------------------------------------------------

    public init(
        dModel: Int,
        numHeads: Int,
        wQ: Tensor, bQ: Tensor,
        wK: Tensor, bK: Tensor,
        wV: Tensor, bV: Tensor,
        wPos: Tensor,
        wOut: Tensor, bOut: Tensor,
        posBiasU: Tensor,
        posBiasV: Tensor
    ) {
        precondition(dModel % numHeads == 0,
            "dModel (\(dModel)) must be divisible by numHeads (\(numHeads))")
        self.dModel = dModel
        self.numHeads = numHeads
        self.wQ = wQ; self.bQ = bQ
        self.wK = wK; self.bK = bK
        self.wV = wV; self.bV = bV
        self.wPos = wPos
        self.wOut = wOut; self.bOut = bOut
        self.posBiasU = posBiasU
        self.posBiasV = posBiasV
    }

    // -------------------------------------------------------------
    // Forward
    // -------------------------------------------------------------

    /// Forward pass.
    ///
    /// - `x`: `[1, T, D]` (we don't currently support batches > 1
    ///   here; Cohere encoder is single-utterance per call).
    /// - `posEmb`: `[1, 2T-1, D]` — sinusoidal absolute encoding of
    ///   the relative offset axis. Use `Self.makeSinusoidalPosEmb(T:D:)`
    ///   to build it.
    ///
    /// Returns the attended hidden state `[1, T, D]`.
    public func forward(_ x: Tensor, posEmb: Tensor) -> Tensor {
        let D = dModel
        let H = numHeads
        let Dh = headDim
        precondition(x.shape.count == 3 && x.shape[0] == 1,
            "RelPosSelfAttention currently expects [1, T, D]; got \(x.shape)")
        precondition(x.shape[2] == D,
            "input last dim \(x.shape[2]) != dModel \(D)")
        let T = x.shape[1]
        precondition(posEmb.shape == [1, 2 * T - 1, D],
            "posEmb shape \(posEmb.shape) != [1, \(2*T-1), \(D)]")

        // 1. Project Q, K, V from x; P from posEmb.
        let bQ3 = bQ.reshape([1, 1, D])
        let bK3 = bK.reshape([1, 1, D])
        let bV3 = bV.reshape([1, 1, D])
        let q = x.matmul(wQ).add(bQ3)            // [1, T, D]
        let k = x.matmul(wK).add(bK3)            // [1, T, D]
        let v = x.matmul(wV).add(bV3)            // [1, T, D]
        let p = posEmb.matmul(wPos)              // [1, 2T-1, D] (no bias)

        // 2. Split into heads.
        // Q, K, V: [1, T, D] → [1, T, H, Dh] → [1, H, T, Dh]
        func splitHeads(_ t: Tensor, seqLen: Int) -> Tensor {
            t.reshape([1, seqLen, H, Dh])
             .permute([0, 2, 1, 3]).contiguous()
        }
        let qH = splitHeads(q, seqLen: T)        // [1, H, T, Dh]
        let kH = splitHeads(k, seqLen: T)        // [1, H, T, Dh]
        let vH = splitHeads(v, seqLen: T)        // [1, H, T, Dh]
        // P: [1, 2T-1, D] → [1, 2T-1, H, Dh] → [1, H, 2T-1, Dh]
        let pH = p.reshape([1, 2 * T - 1, H, Dh])
            .permute([0, 2, 1, 3]).contiguous()

        // 3. Add per-head position biases to Q.
        // posBiasU/V are [H, Dh]; reshape to [1, H, 1, Dh] for
        // broadcast over the time axis.
        let uBias = posBiasU.reshape([1, H, 1, Dh])
        let vBias = posBiasV.reshape([1, H, 1, Dh])
        let qWithU = qH.add(uBias)               // [1, H, T, Dh]
        let qWithV = qH.add(vBias)               // [1, H, T, Dh]

        // 4. Compute the two attention matrices.
        // Standard "AC" content attention: (Q+u) @ K.T → [1, H, T, T]
        // Use matmul on flattened batch×heads = [BH, T, Dh] / [BH, Dh, T].
        let kT_ = kH.permute([0, 1, 3, 2]).contiguous()  // [1, H, Dh, T]
        let pT_ = pH.permute([0, 1, 3, 2]).contiguous()  // [1, H, Dh, 2T-1]

        // Treat the leading batch+head dim as a single "BH" axis for
        // matmul, which is 3-D batched.
        let qWithU3 = qWithU.reshape([H, T, Dh])
        let qWithV3 = qWithV.reshape([H, T, Dh])
        let kT_3 = kT_.reshape([H, Dh, T])
        let pT_3 = pT_.reshape([H, Dh, 2 * T - 1])

        let matrixAC = qWithU3.matmul(kT_3)          // [H, T, T]
        let matrixBD = qWithV3.matmul(pT_3)          // [H, T, 2T-1]

        // Re-add the batch dim before rel_shift (the FFI expects
        // exactly 4-D [B, H, T, 2T-1]).
        let matrixBD4 = matrixBD.reshape([1, H, T, 2 * T - 1])
        let matrixBDShifted = matrixBD4.relShift()    // [1, H, T, T]
        let matrixBDShifted3 = matrixBDShifted.reshape([H, T, T])

        // 5. Combine, scale, softmax.
        let scaled = matrixAC.add(matrixBDShifted3)
        let scale = 1.0 / sqrtf(Float(Dh))
        // Multiply by scalar via broadcast: build a 1-element tensor.
        let scaleTensor: Tensor
        do {
            scaleTensor = try! Tensor.from(data: [scale], shape: [1])
        }
        let scaledNorm = scaled.mul(scaleTensor)
        let attn = scaledNorm.softmax(dim: -1)        // [H, T, T]

        // 6. Apply attention to V.
        let vH3 = vH.reshape([H, T, Dh])
        let attended = attn.matmul(vH3)               // [H, T, Dh]

        // 7. Merge heads and project out.
        let merged = attended
            .reshape([1, H, T, Dh])
            .permute([0, 2, 1, 3]).contiguous()
            .reshape([1, T, D])
        let bOut3 = bOut.reshape([1, 1, D])
        return merged.matmul(wOut).add(bOut3)
    }

    // -------------------------------------------------------------
    // Sinusoidal relative-position embedding
    // -------------------------------------------------------------

    /// Build the relative-position sinusoidal table for a given length.
    ///
    /// Output shape `[1, 2T-1, D]`. Layout:
    ///
    ///     row 0      = encoding of relative offset (T-1)   (largest positive)
    ///     row T-1    = encoding of relative offset 0
    ///     row 2T-2   = encoding of relative offset -(T-1)  (largest negative)
    ///
    /// The encoding for offset `p` and dim index `2k` (resp. `2k+1`)
    /// follows the standard transformer formula:
    ///
    ///     pe[p, 2k]   = sin(p / 10000^{2k/D})
    ///     pe[p, 2k+1] = cos(p / 10000^{2k/D})
    ///
    /// `D` must be even.
    public static func makeSinusoidalPosEmb(seqLen T: Int, dModel D: Int) -> Tensor {
        precondition(D % 2 == 0, "dModel must be even, got \(D)")
        let total = 2 * T - 1
        var data = [Float](repeating: 0, count: total * D)
        // Positions: (T-1), (T-2), ..., 0, -1, ..., -(T-1)
        for i in 0..<total {
            let pos = Float((T - 1) - i)
            for k in 0..<(D / 2) {
                let exponent = Float(2 * k) / Float(D)
                let invFreq = expf(-logf(10_000) * exponent)
                let theta = pos * invFreq
                data[i * D + 2 * k]     = sinf(theta)
                data[i * D + 2 * k + 1] = cosf(theta)
            }
        }
        return try! Tensor.from(data: data, shape: [1, total, D])
    }
}
