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
    private let id: UInt32

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

    private init(id: UInt32) {
        self.id = id
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
        let resultId = ml_engine_matmul(self.id, other.id)
        return Tensor(id: resultId)
    }

    // MARK: - Lifetime

    deinit {
        ml_engine_release(id)
    }
}
