import TranscribeFFIC

/// Top-level namespace for simple value-returning helpers.
///
/// For stateful objects, see `Tensor`.
public enum Transcribe {
    /// FFI sanity check: `a + b` computed in Rust.
    public static func add(_ a: Int32, _ b: Int32) -> Int32 {
        return ml_add(a, b)
    }

    /// One-shot: allocates a zero tensor in Rust, sums it, drops it.
    public static func tensorZerosSum(rows: UInt32, cols: UInt32) -> Float {
        return ml_tensor_zeros_sum(rows, cols)
    }

    /// One-shot: allocates an iota tensor in Rust, sums it, drops it.
    public static func tensorIotaSum(n: UInt32) -> Float {
        return ml_tensor_iota_sum(n)
    }
}

/// A handle to a Rust-owned tensor.
///
/// This is a Swift `class` (reference type) so the `deinit` below fires
/// exactly once, when the last Swift reference drops, and releases the
/// underlying Rust allocation. A `struct` would be copied by value on
/// assignment and each copy would try to free the same pointer — a
/// classic double-free.
///
/// Threading: reads (`sum`, `numel`) are safe from multiple threads on
/// the same Tensor; do not drop a Tensor on one thread while another
/// thread is reading it. In practice: don't share a Tensor across
/// threads unless you serialize access externally.
public final class Tensor {
    /// Opaque pointer to the underlying Rust `Tensor`. Never dereferenced
    /// in Swift — only passed back to C FFI functions.
    private let handle: OpaquePointer

    // MARK: - Constructors

    /// `rows x cols` tensor of zeros.
    public static func zeros(rows: UInt32, cols: UInt32) -> Tensor {
        // Because `MLTensor` is forward-declared (incomplete) in the C
        // header, Swift imports `MLTensor*` as `OpaquePointer` — not
        // `UnsafePointer<MLTensor>`. So constructors already return the
        // right type and no cast is needed.
        //
        // `!` is safe here because our Rust constructor never returns null.
        let handle = ml_tensor_zeros(rows, cols)!
        return Tensor(handle: handle)
    }

    /// `n`-element iota tensor: [0.0, 1.0, ..., n-1].
    public static func iota(_ n: UInt32) -> Tensor {
        let handle = ml_tensor_iota(n)!
        return Tensor(handle: handle)
    }

    /// Build a `rows x cols` tensor by copying a Swift array into Rust.
    /// Returns `nil` if `data.count != rows * cols`.
    public static func from(data: [Float], rows: UInt32, cols: UInt32) -> Tensor? {
        guard data.count == Int(rows) * Int(cols) else { return nil }
        // `withUnsafeBufferPointer` gives us a temporary C-compatible pointer
        // to the array's storage. Rust copies out of it before we return;
        // the pointer is not retained.
        let maybeHandle: OpaquePointer? = data.withUnsafeBufferPointer { buf in
            ml_tensor_from_data(buf.baseAddress, UInt64(buf.count), rows, cols)
        }
        guard let handle = maybeHandle else { return nil }
        return Tensor(handle: handle)
    }

    private init(handle: OpaquePointer) {
        self.handle = handle
    }

    // MARK: - Accessors

    /// Sum of all elements.
    public var sum: Float {
        ml_tensor_sum(handle)
    }

    /// Total number of elements (product of shape dims).
    public var numel: UInt64 {
        ml_tensor_numel(handle)
    }

    // MARK: - Data movement

    /// Copy the tensor's row-major data into a fresh `[Float]`.
    /// Safe (Swift owns the returned array) but O(numel) memory + copy.
    /// For large tensors or hot paths, prefer `withUnsafeData`.
    public func toArray() -> [Float] {
        let count = Int(numel)
        // Allocate uninitialized storage, then fill it via FFI.
        return [Float](unsafeUninitializedCapacity: count) { buffer, initializedCount in
            let written = ml_tensor_copy_into(
                handle,
                buffer.baseAddress,
                UInt64(count)
            )
            initializedCount = Int(written)
        }
    }

    /// Zero-copy read access to the tensor's buffer.
    ///
    /// The closure receives a pointer that is valid ONLY for the
    /// duration of the call. Do not store it, do not pass it to async
    /// work, do not return anything that captures it.
    ///
    /// Typical use: read a few elements, compute a summary, return it.
    public func withUnsafeData<R>(
        _ body: (UnsafeBufferPointer<Float>) throws -> R
    ) rethrows -> R {
        let ptr = ml_tensor_data_ptr(handle)
        let buffer = UnsafeBufferPointer(start: ptr, count: Int(numel))
        return try body(buffer)
    }

    // MARK: - Lifetime

    deinit {
        ml_tensor_free(handle)
    }
}
