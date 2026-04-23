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
public final class SafetensorsWeights {
    private let handle: OpaquePointer

    /// Load a `.safetensors` file into the engine.
    /// Only F32 tensors are supported for now.
    public convenience init(path: URL) throws {
        try self.init(path: path.path)
    }

    public init(path: String) throws {
        let maybeHandle: OpaquePointer? = path.withCString { cstr in
            let bytes = UnsafeRawPointer(cstr).assumingMemoryBound(to: UInt8.self)
            return ml_engine_safetensors_load(bytes, UInt64(strlen(cstr)))
        }
        guard let h = maybeHandle else {
            throw SafetensorsError.loadFailed(path: path)
        }
        self.handle = h
    }

    /// Number of tensors in the file.
    public var count: Int {
        Int(ml_engine_safetensors_count(handle))
    }

    /// All tensor names, in sorted order.
    public var keys: [String] {
        let n = count
        var result: [String] = []
        result.reserveCapacity(n)
        var scratch = [UInt8](repeating: 0, count: 256)
        for i in 0..<n {
            let nameLen = Int(ml_engine_safetensors_name_len(handle, UInt64(i)))
            if nameLen > scratch.count { scratch = [UInt8](repeating: 0, count: nameLen) }
            let written = scratch.withUnsafeMutableBufferPointer { buf in
                Int(ml_engine_safetensors_name_at(
                    handle, UInt64(i), buf.baseAddress, UInt64(nameLen)))
            }
            let name = String(decoding: scratch.prefix(written), as: UTF8.self)
            result.append(name)
        }
        return result
    }

    /// Borrowed view of the tensor with the given name, or nil.
    public subscript(name: String) -> Tensor? {
        let id: UInt32 = name.withCString { cstr in
            let bytes = UnsafeRawPointer(cstr).assumingMemoryBound(to: UInt8.self)
            return ml_engine_safetensors_get(handle, bytes, UInt64(strlen(cstr)))
        }
        guard id != UInt32.max else { return nil }
        return Tensor(id: id, ownedBy: self)
    }

    deinit {
        ml_engine_safetensors_free(handle)
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
