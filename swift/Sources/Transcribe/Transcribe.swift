import Foundation
import TranscribeFFIC

/// Sanity-check namespace. Pure FFI helpers unrelated to the engine.
public enum Transcribe {
    /// FFI smoke test: `a + b` computed in Rust.
    public static func add(_ a: Int32, _ b: Int32) -> Int32 {
        return ml_add(a, b)
    }
}

// ---------------------------------------------------------------------------
// Storage dtype
// ---------------------------------------------------------------------------

/// On-device storage dtype for a tensor in the engine. F16 halves the
/// resident memory of a tensor compared to F32; ops still operate on
/// F32 and up-convert F16 transiently when consumed.
///
/// Distinct from `SafetensorsWeights.Dtype`, which describes a tensor's
/// dtype when *serialized to disk*. The two coincide in name but are
/// different types in different scopes.
public enum TensorDtype: Equatable {
    case f32
    case f16

    fileprivate var rawValue: UInt32 {
        switch self {
        case .f32: return 0
        case .f16: return 1
        }
    }

    fileprivate static func decode(_ raw: UInt32) -> TensorDtype? {
        switch raw {
        case 0: return .f32
        case 1: return .f16
        default: return nil
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

/// Errors surfaced by `Tensor` constructors.
public enum TensorError: Error {
    /// `data.count` did not match the product of `shape`.
    case shapeMismatch(expected: Int, got: Int)
    /// Rust returned the invalid-id sentinel for some other reason.
    case engineError
}

/// A handle to a tensor owned by the Rust engine (`TensorStore`).
///
/// Internally this is a single `UInt32` id. The Rust side owns the
/// actual buffer. `deinit` calls `ml_engine_release` so the store can
/// reclaim the slot. Must be a `class` (reference type) — a `struct`
/// would double-release on copy.
///
/// Thread safety: the engine is protected by a Mutex on the Rust side,
/// so individual calls are safe from multiple threads. Do not drop a
/// `Tensor` on one thread while another is reading it.
public final class Tensor {
    /// Rust-side `TensorId` (a `u32`). `UInt32.max` is reserved as an
    /// error sentinel — we never construct a `Tensor` holding it.
    fileprivate let id: UInt32

    /// If non-nil, this Tensor is a *borrowed view* of a tensor owned
    /// by `owner` (e.g. a `SafetensorsWeights`). Deinit does NOT call
    /// `ml_engine_release` — the owner will handle cleanup. The strong
    /// reference also keeps the owner alive while any borrow exists.
    private let owner: AnyObject?

    // MARK: - Constructors

    /// Allocate a tensor of the given shape filled with zeros.
    public static func zeros(shape: [Int]) -> Tensor {
        let dims = shape.map { UInt64($0) }
        let id = dims.withUnsafeBufferPointer { buf in
            ml_engine_zeros(buf.baseAddress, UInt64(buf.count))
        }
        precondition(id != UInt32.max, "engine returned invalid id for zeros")
        return Tensor(id: id)
    }

    /// Convenience for the common 2-D zeros case.
    public static func zeros(rows: UInt32, cols: UInt32) -> Tensor {
        return zeros(shape: [Int(rows), Int(cols)])
    }

    /// 1-D iota tensor: `[0.0, 1.0, ..., n-1]`.
    public static func iota(_ n: UInt32) -> Tensor {
        return Tensor(id: ml_engine_iota(n))
    }

    /// Build a tensor of the given shape by copying `data` into Rust.
    /// Throws `TensorError.shapeMismatch` if `data.count != product(shape)`.
    public static func from(data: [Float], shape: [Int]) throws -> Tensor {
        let expected = shape.reduce(1, *)
        guard data.count == expected else {
            throw TensorError.shapeMismatch(expected: expected, got: data.count)
        }
        let dims = shape.map { UInt64($0) }
        let id: UInt32 = data.withUnsafeBufferPointer { dataBuf in
            dims.withUnsafeBufferPointer { shapeBuf in
                ml_engine_from_data(
                    dataBuf.baseAddress,
                    UInt64(dataBuf.count),
                    shapeBuf.baseAddress,
                    UInt64(shapeBuf.count)
                )
            }
        }
        guard id != UInt32.max else { throw TensorError.engineError }
        return Tensor(id: id)
    }

    /// Convenience for the common 2-D `from(data:)` case.
    public static func from(data: [Float], rows: UInt32, cols: UInt32) -> Tensor? {
        return try? from(data: data, shape: [Int(rows), Int(cols)])
    }

    /// Private designated initializer. Every constructor funnels
    /// through here. `owner == nil` means we own the id; otherwise
    /// we're a borrowed view and the owner is responsible for release.
    fileprivate init(id: UInt32, ownedBy owner: AnyObject? = nil) {
        self.id = id
        self.owner = owner
    }

    // MARK: - Shape / size

    /// Total number of elements (product of shape dims).
    public var numel: UInt64 {
        ml_engine_numel(id)
    }

    /// Storage dtype. F16 tensors take half the memory of F32 but are
    /// up-converted to F32 transiently when consumed by ops.
    public var dtype: TensorDtype {
        let raw = ml_engine_dtype(id)
        return TensorDtype.decode(raw) ?? .f32
    }

    /// Convert this tensor's storage to the requested dtype, returning
    /// a new Tensor (or `self` if already in `target`). The original
    /// tensor is unchanged.
    public func cast(to target: TensorDtype) -> Tensor {
        let newId = ml_engine_cast_dtype(id, target.rawValue)
        if newId == self.id { return self }
        return Tensor(id: newId)
    }

    /// Full shape as a Swift array.
    public var shape: [Int] {
        let ndim = Int(ml_engine_shape_len(id))
        var dims = [UInt64](repeating: 0, count: ndim)
        let written = dims.withUnsafeMutableBufferPointer { buf in
            Int(ml_engine_shape_copy(id, buf.baseAddress, UInt64(ndim)))
        }
        return dims.prefix(written).map { Int($0) }
    }

    // MARK: - Data movement

    /// Sum of all elements. Reads the buffer directly; no autograd.
    public var sum: Float {
        ml_engine_sum(id)
    }

    /// Copy the tensor's row-major data into a fresh `[Float]`.
    public func toArray() -> [Float] {
        let count = Int(numel)
        return [Float](unsafeUninitializedCapacity: count) { buffer, initializedCount in
            let written = ml_engine_copy_into(
                id,
                buffer.baseAddress,
                UInt64(count)
            )
            initializedCount = Int(written)
        }
    }

    // MARK: - Ops

    /// Matrix multiply: `self @ other`. Both operands must be 2-D with
    /// compatible inner dims. Returns a new `Tensor` whose shape is
    /// `[self.rows, other.cols]`.
    public func matmul(_ other: Tensor) -> Tensor {
        Tensor(id: ml_engine_matmul(self.id, other.id))
    }

    /// Elementwise add: `self + other`.
    public func add(_ other: Tensor) -> Tensor {
        Tensor(id: ml_engine_add(self.id, other.id))
    }

    /// Elementwise multiply: `self * other`.
    public func mul(_ other: Tensor) -> Tensor {
        Tensor(id: ml_engine_mul(self.id, other.id))
    }

    /// ReLU: `max(0, x)` elementwise.
    public func relu() -> Tensor {
        Tensor(id: ml_engine_relu(self.id))
    }

    /// Softmax along `dim` (negative values count from the end;
    /// `-1` = last dim, matching PyTorch).
    public func softmax(dim: Int32) -> Tensor {
        Tensor(id: ml_engine_softmax(self.id, dim))
    }

    /// GELU activation — smooth ReLU variant used by most transformer FFNs.
    public func gelu() -> Tensor {
        Tensor(id: ml_engine_gelu(self.id))
    }

    /// Sigmoid: `1 / (1 + exp(-x))`.
    public func sigmoid() -> Tensor {
        Tensor(id: ml_engine_sigmoid(self.id))
    }

    /// SiLU / Swish: `x * sigmoid(x)`. Composed in Swift from `mul` +
    /// `sigmoid` — no dedicated Rust op needed.
    public func silu() -> Tensor {
        self.mul(self.sigmoid())
    }

    /// LayerNorm over the last dim. `gamma` and `beta` must have shape
    /// `[C]` where `C = shape.last`. `eps` is typically `1e-5`.
    public func layerNorm(gamma: Tensor, beta: Tensor, eps: Float = 1e-5) -> Tensor {
        Tensor(id: ml_engine_layernorm(self.id, gamma.id, beta.id, eps))
    }

    /// RMSNorm over the last dim: `y = gamma * x / sqrt(mean(x²) + eps)`.
    /// Unlike LayerNorm, doesn't subtract the mean and has no bias.
    /// `gamma` must have shape `[C]` where `C = shape.last`.
    /// Used by LLaMA, Mistral, and most modern decoder-only LMs.
    public func rmsNorm(gamma: Tensor, eps: Float = 1e-5) -> Tensor {
        let newId = ml_engine_rmsnorm(self.id, gamma.id, eps)
        precondition(newId != UInt32.max,
            "rmsNorm: expected gamma shape [\(shape.last ?? -1)], got \(gamma.shape)")
        return Tensor(id: newId)
    }

    /// Scaled dot-product attention. `self` is the query; all three
    /// must be `[BH, S, D]` (batched heads × seq_len × head_dim).
    /// Output shape matches. `scale` is typically `1 / sqrt(Float(D))`.
    /// When `causal = true`, position `i` can only attend to positions
    /// `j <= i` (decoder self-attention).
    public func attention(
        key: Tensor,
        value: Tensor,
        scale: Float,
        causal: Bool = false
    ) -> Tensor {
        Tensor(id: ml_engine_attention(
            self.id, key.id, value.id, scale, causal ? 1 : 0
        ))
    }

    // MARK: - Layout (reshape / transpose / contiguous)

    /// Return a view of the tensor with a new shape. Total element
    /// count must match `numel`. After a `permute`, prefer calling
    /// `contiguous()` first so the view reflects the post-permute
    /// memory order rather than the original.
    public func reshape(_ newShape: [Int]) -> Tensor {
        let expected = newShape.reduce(1, *)
        precondition(
            expected == Int(numel),
            "reshape: new shape \(newShape) implies \(expected) elements, tensor has \(numel)"
        )
        let dims = newShape.map { UInt64($0) }
        let newId = dims.withUnsafeBufferPointer { buf in
            ml_engine_reshape(id, buf.baseAddress, UInt64(buf.count))
        }
        return Tensor(id: newId)
    }

    /// Permute dimensions according to `dims`. `dims` must be a
    /// permutation of `0..<shape.count`. Example: swap the last two
    /// dims of a 4-D tensor with `permute([0, 1, 3, 2])`.
    ///
    /// Result is usually non-contiguous — follow with `.contiguous()`
    /// before any op that requires row-major layout.
    public func permute(_ dims: [Int]) -> Tensor {
        let ndim = shape.count
        precondition(dims.count == ndim,
            "permute: \(dims.count) dims given for \(ndim)-D tensor")
        precondition(Set(dims) == Set(0..<ndim),
            "permute: \(dims) is not a permutation of 0..<\(ndim)")
        let raw = dims.map { UInt64($0) }
        let newId = raw.withUnsafeBufferPointer { buf in
            ml_engine_permute(id, buf.baseAddress, UInt64(buf.count))
        }
        return Tensor(id: newId)
    }

    /// Convenience: swap two dims. `transpose(d0, d1)` is equivalent
    /// to `permute(...)` with the two dims swapped in the identity
    /// permutation. Matches PyTorch's signature.
    public func transpose(_ d0: Int, _ d1: Int) -> Tensor {
        var dims = Array(0..<shape.count)
        dims.swapAt(d0, d1)
        return permute(dims)
    }

    /// Force row-major memory layout. No-op if already contiguous.
    /// Required between `permute` and `reshape` if you want the reshape
    /// to see the permuted order rather than the original strides.
    ///
    /// When the input is already contiguous, the Rust `ensure_contiguous`
    /// returns the same id — we must `return self` in that case so ARC
    /// sees a single owner. Wrapping the same id in a fresh Tensor
    /// would double-release it.
    public func contiguous() -> Tensor {
        let newId = ml_engine_contiguous(id)
        if newId == self.id {
            return self
        }
        return Tensor(id: newId)
    }

    /// PyTorch's `repeat_interleave`: duplicates each slice along `dim`
    /// `repeats` times consecutively. For input shape `[..., d_k, ...]`
    /// with `dim = k`, output shape is `[..., d_k * repeats, ...]`.
    ///
    /// Layout: element `i` along `dim` in input → elements
    /// `[i*repeats, (i+1)*repeats - 1]` in output.
    ///
    /// Negative `dim` counts from the end (matches PyTorch).
    ///
    /// Main use case: GQA expansion. K/V tensors shaped
    /// `[B, H_kv, S, D_h]` become `[B, H_q, S, D_h]` via
    /// `.repeatInterleave(dim: 1, repeats: H_q/H_kv)`, broadcasting
    /// each KV head to the G consecutive Q heads that share it.
    public func repeatInterleave(dim: Int, repeats: Int) -> Tensor {
        precondition(repeats >= 1, "repeats must be >= 1")
        let newId = ml_engine_repeat_interleave(
            id, Int32(dim), UInt32(repeats)
        )
        precondition(newId != UInt32.max,
            "repeatInterleave: invalid dim \(dim) for shape \(shape)")
        return Tensor(id: newId)
    }

    // MARK: - Embedding lookup

    /// Look up token ids in this tensor (treated as an embedding table
    /// of shape `[vocab_size, embed_dim]`). `tokens` is a 2-D array of
    /// shape `[batch][seq_len]`; output shape is
    /// `[batch, seq_len, embed_dim]`.
    ///
    /// Each id must be in `0..<vocab_size`. Out-of-bounds ids cause a
    /// panic in the Rust core; we precondition here with a clearer
    /// message.
    public func embed(tokens: [[Int]]) -> Tensor {
        let batch = tokens.count
        let seqLen = tokens.first?.count ?? 0
        precondition(tokens.allSatisfy { $0.count == seqLen },
            "embed: inner arrays must all have the same length")

        // Table is [vocab_size, embed_dim].
        let tableShape = shape
        precondition(tableShape.count == 2,
            "embed: table must be 2-D [vocab, dim], got \(tableShape)")
        let vocabSize = tableShape[0]
        for row in tokens {
            for id in row {
                precondition(id >= 0 && id < vocabSize,
                    "embed: token id \(id) out of range 0..<\(vocabSize)")
            }
        }

        let flat: [UInt64] = tokens.flatMap { $0.map { UInt64($0) } }
        let newId: UInt32 = flat.withUnsafeBufferPointer { buf in
            ml_engine_embedding(
                id,
                buf.baseAddress,
                UInt64(buf.count),
                UInt64(batch),
                UInt64(seqLen)
            )
        }
        precondition(newId != UInt32.max,
            "embed: Rust engine rejected the inputs")
        return Tensor(id: newId)
    }

    /// Convenience for a single sequence (batch = 1).
    /// Output shape is `[1, tokens.count, embed_dim]`.
    public func embed(tokens: [Int]) -> Tensor {
        embed(tokens: [tokens])
    }

    // MARK: - Positional encoding

    /// Apply RoPE (rotary position embedding) to the last two dims
    /// (`[..., S, D]`). `D` must be even.
    ///
    /// Half-split convention: pairs feature `i` with `i + D/2` (LLaMA /
    /// HuggingFace). Apply this to both Q and K before attention; V is
    /// left unrotated.
    ///
    /// - `startPos`: position of the first token along the sequence
    ///   axis. Use 0 for a fresh forward pass; during incremental
    ///   decoding with a KV cache, pass the length of the cache so the
    ///   new token gets the right absolute position.
    /// - `base`: frequency base. 10,000 is the LLaMA default; some
    ///   long-context models (e.g. extended LLaMA) use 500,000 or more.
    public func rope(startPos: Int = 0, base: Float = 10_000.0) -> Tensor {
        let newId = ml_engine_rope(id, UInt64(startPos), base)
        precondition(newId != UInt32.max,
            "rope: expected shape [..., S, D] with D even; got \(shape)")
        return Tensor(id: newId)
    }

    // MARK: - Lifetime

    deinit {
        // Only owned Tensors release the id. Borrowed views let the
        // owner (e.g. a SafetensorsWeights handle) clean up en masse.
        if owner == nil {
            ml_engine_release(id)
        }
    }
}

// ---------------------------------------------------------------------------
// KV cache — for incremental autoregressive decoding
// ---------------------------------------------------------------------------

/// Errors from KV cache operations.
public enum KVCacheError: Error {
    case unknownHandle
    case shapeMismatch
    case appendOverflow
    case engineError
}

/// A state-holding cache of past keys and values for one attention
/// layer, used during autoregressive decoding.
///
/// Typical use inside one decode step:
///
///   let out = try cache.appendAndDecode(
///       query: q, key: k, value: v,
///       scale: 1.0 / sqrt(Float(headDim))
///   )
///
/// Each call appends one new step (`k` and `v` at position = length)
/// and returns attention over ALL past keys/values — O(length) instead
/// of O(length²). Compare to re-running full attention from scratch
/// every step.
///
/// Create one cache per attention layer in the model. Caches are
/// freed automatically on deinit.
public final class KVCache {
    public struct Config {
        public let batchSize: Int
        public let numHeads: Int
        public let headDim: Int
        public let maxSeqLen: Int
        public let quantized: Bool

        public init(
            batchSize: Int = 1,
            numHeads: Int,
            headDim: Int,
            maxSeqLen: Int,
            quantized: Bool = false
        ) {
            self.batchSize = batchSize
            self.numHeads = numHeads
            self.headDim = headDim
            self.maxSeqLen = maxSeqLen
            self.quantized = quantized
        }
    }

    public let config: Config
    private let handle: UInt32

    public init(config: Config) {
        self.config = config
        self.handle = ml_engine_kvcache_new(
            UInt64(config.batchSize),
            UInt64(config.numHeads),
            UInt64(config.headDim),
            UInt64(config.maxSeqLen),
            config.quantized ? 1 : 0
        )
        precondition(handle != 0, "KVCache: allocation failed")
    }

    /// Current fill level (how many tokens have been appended).
    public var length: Int {
        Int(ml_engine_kvcache_len(handle))
    }

    /// Zero out the cache; length returns to 0.
    public func reset() {
        ml_engine_kvcache_reset(handle)
    }

    /// Append one step of K and V without computing attention.
    /// Useful when you want to prefill the cache with a batch of
    /// tokens processed by a separate (non-cached) forward pass.
    ///
    /// Shapes: `k` and `v` must each be `[B, H, 1, D]`.
    public func append(key: Tensor, value: Tensor) throws {
        let rc = ml_engine_kvcache_append(handle, key.id, value.id)
        switch rc {
        case 0:  return
        case -1: throw KVCacheError.unknownHandle
        case -2: throw KVCacheError.shapeMismatch
        default: throw KVCacheError.engineError
        }
    }

    /// Append one step and compute attention against all cached
    /// keys/values. Returns the attention output for the single query
    /// step. Shapes: all three inputs `[B, H, 1, D]`; output `[B, H, 1, D]`.
    public func appendAndDecode(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: Float
    ) throws -> Tensor {
        let outId = ml_engine_kvcache_append_and_decode(
            handle, query.id, key.id, value.id, scale
        )
        guard outId != UInt32.max else {
            throw KVCacheError.engineError
        }
        return Tensor(id: outId)
    }

    deinit {
        ml_engine_kvcache_free(handle)
    }
}

// ---------------------------------------------------------------------------
// Safetensors
// ---------------------------------------------------------------------------

public enum SafetensorsError: Error {
    /// Load failed (file not found, bad format, unsupported dtype…).
    case loadFailed(path: String)
    /// Save failed.
    case saveFailed(path: String)
    /// Asked to save a Tensor that is a borrowed view — we don't own
    /// its lifecycle, so this would be surprising. Explicitly disallow.
    case cannotSaveBorrowedTensor
    /// Sharded load failed: the index JSON was missing or malformed.
    case indexMalformed(path: String)
    /// A shard referenced by the index was missing on disk.
    case missingShard(path: String)
}

// ---------------------------------------------------------------------------
// WeightSource — a name-indexed lookup of weights.
// ---------------------------------------------------------------------------

/// Anything that exposes a name-keyed view of tensors. Both
/// `SafetensorsWeights` (raw files) and `WeightMap` (renamed /
/// transposed views) conform, so `DecoderLM` and friends can accept
/// either transparently.
public protocol WeightSource: AnyObject {
    var count: Int { get }
    var keys: [String] { get }
    subscript(name: String) -> Tensor? { get }
}

/// The minimum shape of a `model.safetensors.index.json` we care about.
/// The real file may have extra keys like `metadata`; Codable ignores them.
private struct SafetensorsIndex: Decodable {
    let weight_map: [String: String]
}

/// A read-only, name-indexed collection of tensors loaded from a
/// `.safetensors` file.
///
/// Tensors returned by `subscript(name:)` are *borrowed views* — they
/// share an id with the backing store but do not call
/// `ml_engine_release` on deinit. Instead, this class's `deinit` frees
/// every loaded tensor at once. Each borrowed `Tensor` retains a
/// strong reference to this `SafetensorsWeights`, so the collection
/// stays alive as long as any view exists.
public final class SafetensorsWeights: WeightSource {
    /// One `LoadedWeightsHandle` per underlying file. For a single-file
    /// model this is a one-element array; for a sharded HuggingFace
    /// model (e.g. `model-00001-of-00003.safetensors` + friends) it
    /// holds one handle per shard, in load order.
    ///
    /// Tensor lookups iterate handles and return the first hit. Names
    /// are expected to be unique across shards (the HF convention);
    /// if a name were duplicated, the first shard loaded wins.
    private let handles: [OpaquePointer]

    // MARK: - Single-file load

    /// Load a single `.safetensors` file into the engine.
    public convenience init(path: URL) throws {
        try self.init(path: path.path)
    }

    public convenience init(path: String) throws {
        try self.init(path: path, keepF16: false)
    }

    /// Single-file load with explicit dtype handling. When
    /// `keepF16: true`, F16 source tensors are stored as F16 in the
    /// engine (halving resident memory of weights) instead of being
    /// up-converted to F32. F32 and BF16 source dtypes are unaffected.
    public init(path: String, keepF16: Bool) throws {
        guard let h = Self.openHandle(path: path, keepF16: keepF16) else {
            throw SafetensorsError.loadFailed(path: path)
        }
        self.handles = [h]
    }

    /// URL convenience that takes the same `keepF16` knob.
    public convenience init(path: URL, keepF16: Bool) throws {
        try self.init(path: path.path, keepF16: keepF16)
    }

    // MARK: - Sharded load

    /// Load a sharded HuggingFace model described by a
    /// `model.safetensors.index.json` file at `indexPath`. The index's
    /// `weight_map` is parsed to discover which shard files to load;
    /// each shard is resolved relative to the index file's directory.
    ///
    /// Throws `SafetensorsError.indexMalformed` or `.missingShard` if
    /// the index is bad or a referenced shard doesn't exist.
    public convenience init(shardedIndex indexPath: URL) throws {
        try self.init(shardedIndex: indexPath.path)
    }

    public init(shardedIndex indexPath: String) throws {
        let url = URL(fileURLWithPath: indexPath)
        guard let data = try? Data(contentsOf: url) else {
            throw SafetensorsError.indexMalformed(path: indexPath)
        }
        let index: SafetensorsIndex
        do {
            index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
        } catch {
            throw SafetensorsError.indexMalformed(path: indexPath)
        }

        // Collect unique shard filenames, preserving discovery order so
        // load order is deterministic (makes "first shard wins" on
        // duplicate names predictable).
        var seen = Set<String>()
        var shardNames: [String] = []
        for (_, file) in index.weight_map {
            if seen.insert(file).inserted {
                shardNames.append(file)
            }
        }

        let baseDir = url.deletingLastPathComponent()
        var opened: [OpaquePointer] = []
        for name in shardNames {
            let shardURL = baseDir.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: shardURL.path) else {
                // Clean up anything we already opened.
                for h in opened { ml_engine_safetensors_free(h) }
                throw SafetensorsError.missingShard(path: shardURL.path)
            }
            guard let h = Self.openHandle(path: shardURL.path) else {
                for h in opened { ml_engine_safetensors_free(h) }
                throw SafetensorsError.loadFailed(path: shardURL.path)
            }
            opened.append(h)
        }
        self.handles = opened
    }

    /// Convenience: point at a directory and expect the standard
    /// `model.safetensors.index.json` filename inside it.
    public convenience init(shardedDirectory: URL) throws {
        let indexURL = shardedDirectory
            .appendingPathComponent("model.safetensors.index.json")
        try self.init(shardedIndex: indexURL)
    }

    // MARK: - Query

    /// Total tensor count across all shards.
    public var count: Int {
        handles.reduce(0) { $0 + Int(ml_engine_safetensors_count($1)) }
    }

    /// All tensor names, sorted. Duplicates across shards (shouldn't
    /// happen in well-formed HF files) are de-duped.
    public var keys: [String] {
        var all = Set<String>()
        var scratch = [UInt8](repeating: 0, count: 256)
        for handle in handles {
            let n = Int(ml_engine_safetensors_count(handle))
            for i in 0..<n {
                let nameLen = Int(ml_engine_safetensors_name_len(handle, UInt64(i)))
                if nameLen > scratch.count {
                    scratch = [UInt8](repeating: 0, count: nameLen)
                }
                let written = scratch.withUnsafeMutableBufferPointer { buf in
                    Int(ml_engine_safetensors_name_at(
                        handle, UInt64(i), buf.baseAddress, UInt64(nameLen)))
                }
                all.insert(String(decoding: scratch.prefix(written), as: UTF8.self))
            }
        }
        return all.sorted()
    }

    /// Borrowed view of the tensor with the given name, or nil. When
    /// sharded, returns the tensor from the first shard that contains
    /// it (load order).
    public subscript(name: String) -> Tensor? {
        for handle in handles {
            let id: UInt32 = name.withCString { cstr in
                let bytes = UnsafeRawPointer(cstr).assumingMemoryBound(to: UInt8.self)
                return ml_engine_safetensors_get(handle, bytes, UInt64(strlen(cstr)))
            }
            if id != UInt32.max {
                return Tensor(id: id, ownedBy: self)
            }
        }
        return nil
    }

    // MARK: - Helpers

    private static func openHandle(path: String, keepF16: Bool = false) -> OpaquePointer? {
        return path.withCString { cstr in
            let bytes = UnsafeRawPointer(cstr).assumingMemoryBound(to: UInt8.self)
            let len = UInt64(strlen(cstr))
            return keepF16
                ? ml_engine_safetensors_load_keep_f16(bytes, len)
                : ml_engine_safetensors_load(bytes, len)
        }
    }

    deinit {
        for handle in handles {
            ml_engine_safetensors_free(handle)
        }
    }

    // MARK: - Saving (mostly for tests / debugging)

    /// On-disk tensor dtype for `save(_:to:dtype:)`.
    public enum Dtype {
        /// F32 (single-precision, 4 bytes per element). Exact.
        case f32
        /// F16 (half-precision, 2 bytes per element). Halves file size;
        /// loses precision for values that aren't exactly representable.
        /// Matches how most HuggingFace transformer weights ship.
        case f16
    }

    /// Serialize a set of `(name, Tensor)` pairs to `path` at the given
    /// `dtype`. Tensors live as F32 in the store; for `.f16`, Rust
    /// converts on the way out. Throws `SafetensorsError.saveFailed`
    /// on I/O error.
    public static func save(_ tensors: [(name: String, tensor: Tensor)],
                            to path: String,
                            dtype: Dtype = .f32) throws {
        let plan = ml_engine_safetensors_save_open()!
        for (name, tensor) in tensors {
            let rc: Int32 = name.withCString { cstr in
                let bytes = UnsafeRawPointer(cstr).assumingMemoryBound(to: UInt8.self)
                return ml_engine_safetensors_save_add(
                    plan, bytes, UInt64(strlen(cstr)), tensor.id)
            }
            if rc != 0 { throw SafetensorsError.saveFailed(path: path) }
        }
        let rc: Int32 = path.withCString { cstr in
            let bytes = UnsafeRawPointer(cstr).assumingMemoryBound(to: UInt8.self)
            switch dtype {
            case .f32:
                return ml_engine_safetensors_save_finish(
                    plan, bytes, UInt64(strlen(cstr)))
            case .f16:
                return ml_engine_safetensors_save_finish_f16(
                    plan, bytes, UInt64(strlen(cstr)))
            }
        }
        if rc != 0 { throw SafetensorsError.saveFailed(path: path) }
    }
}
