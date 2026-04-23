import Foundation

/// A minimal 2-layer multilayer perceptron — the simplest trainable
/// architecture that exercises the full Swift / Rust pipeline:
///
///   x -> Linear(inDim -> hiddenDim) -> ReLU -> Linear(hiddenDim -> outDim) -> logits
///
/// Weight convention (safetensors keys):
///
///   "fc1.weight"   shape [inDim,    hiddenDim]
///   "fc1.bias"     shape [1,        hiddenDim]
///   "fc2.weight"   shape [hiddenDim, outDim]
///   "fc2.bias"     shape [1,        outDim]
///
/// Weight layout is `[in, out]` so forward is literally `x @ W + b` with
/// no transpose. This is the opposite of PyTorch's `Linear` (which
/// stores `[out, in]` and uses `x @ W.T`); we chose the simpler
/// convention because no transpose op is exposed yet.
///
/// Biases are `[1, out]` (not `[out]`) to match the row-major output
/// shape exactly, since the current `add` op requires matching shapes.
public final class MLP2 {
    public let inDim: Int
    public let hiddenDim: Int
    public let outDim: Int

    private let w1: Tensor
    private let b1: Tensor
    private let w2: Tensor
    private let b2: Tensor

    public enum LoadError: Error, CustomStringConvertible {
        case missing(String)
        case unexpectedShape(name: String, expected: String, got: [Int])

        public var description: String {
            switch self {
            case .missing(let n): return "missing weight: \(n)"
            case .unexpectedShape(let n, let e, let g):
                return "weight '\(n)': expected \(e), got \(g)"
            }
        }
    }

    /// Build the MLP by looking up the four named tensors in `weights`.
    /// Dims are inferred from weight shapes.
    public init(weights: SafetensorsWeights) throws {
        guard let w1 = weights["fc1.weight"] else { throw LoadError.missing("fc1.weight") }
        guard let b1 = weights["fc1.bias"]   else { throw LoadError.missing("fc1.bias") }
        guard let w2 = weights["fc2.weight"] else { throw LoadError.missing("fc2.weight") }
        guard let b2 = weights["fc2.bias"]   else { throw LoadError.missing("fc2.bias") }

        guard w1.shape.count == 2 else {
            throw LoadError.unexpectedShape(
                name: "fc1.weight", expected: "[inDim, hiddenDim]", got: w1.shape)
        }
        self.inDim = w1.shape[0]
        self.hiddenDim = w1.shape[1]

        guard b1.shape == [1, hiddenDim] else {
            throw LoadError.unexpectedShape(
                name: "fc1.bias", expected: "[1, \(hiddenDim)]", got: b1.shape)
        }
        guard w2.shape.count == 2, w2.shape[0] == hiddenDim else {
            throw LoadError.unexpectedShape(
                name: "fc2.weight", expected: "[\(hiddenDim), outDim]", got: w2.shape)
        }
        self.outDim = w2.shape[1]

        guard b2.shape == [1, outDim] else {
            throw LoadError.unexpectedShape(
                name: "fc2.bias", expected: "[1, \(outDim)]", got: b2.shape)
        }

        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
    }

    /// Full forward pass: returns raw logits of shape `[batch, outDim]`.
    ///
    /// Intermediate tensors (matmul result, hidden pre-activation,
    /// post-ReLU) are owned and freed automatically as the chain
    /// collapses — Swift's ARC drops each intermediate's `Tensor` as
    /// soon as the next op consumes it.
    public func forward(_ input: Tensor) -> Tensor {
        return input
            .matmul(w1)   // [batch, hiddenDim]
            .add(b1)
            .relu()
            .matmul(w2)   // [batch, outDim]
            .add(b2)
    }

    /// Classify a single input (flat 1-D array of length `inDim`).
    /// Returns the argmax class index in `0..<outDim`.
    public func classify(_ input: [Float]) throws -> Int {
        guard input.count == inDim else {
            throw LoadError.unexpectedShape(
                name: "input", expected: "[\(inDim)]", got: [input.count])
        }
        let x = try Tensor.from(data: input, shape: [1, inDim])
        let logits = forward(x).toArray()
        // Argmax in Swift — no need for an FFI op at this scale.
        return logits.enumerated().max(by: { $0.element < $1.element })!.offset
    }
}

// ---------------------------------------------------------------------------
// Test-only helpers
// ---------------------------------------------------------------------------

/// Deterministic linear-congruential RNG so tests can generate
/// reproducible "pretrained" weights without needing a network fetch
/// or a fixture file in version control.
///
/// Exposed as `internal` so tests in the same module can reach it,
/// but not part of the public API.
struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed &+ 0x9E37_79B9_7F4A_7C15 }
    mutating func next() -> UInt64 {
        // SplitMix64 — small, fast, good statistical quality.
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
