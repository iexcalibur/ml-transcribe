import XCTest
import Foundation
@testable import Transcribe

/// Phase 1 of F16 support: storage halved, ops still operate on F32
/// (transient up-conversion happens inside `to_host`).
///
/// What this proves:
///   - Cast F32 → F16 → F32 round-trips with expected precision loss.
///   - F16-loaded weights survive a real forward pass (composing with
///     existing F32 ops).
///   - dtype is reportable from Swift.
final class F16StorageTests: XCTestCase {

    private func tmp(_ name: String) -> String {
        (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("ml_transcribe_f16_\(name)_\(UUID().uuidString).safetensors")
    }

    // MARK: - Cast

    /// New tensors default to F32.
    func testFreshTensorIsF32() throws {
        let t = try Tensor.from(data: [1, 2, 3, 4], shape: [2, 2])
        XCTAssertEqual(t.dtype, .f32)
    }

    /// Cast to the same dtype is a no-op (returns the same Tensor instance).
    func testCastToSameDtypeIsIdentity() throws {
        let t = try Tensor.from(data: [1, 2, 3, 4], shape: [2, 2])
        let same = t.cast(to: .f32)
        XCTAssertTrue(t === same, "cast to same dtype should return self")
    }

    /// F32 → F16 → F32 round-trip preserves integer values exactly.
    /// (Half-precision can represent any integer in [-2048, 2048].)
    func testF32ToF16RoundTripExactForIntegers() throws {
        let original: [Float] = [-3, -2, -1, 0, 1, 2, 3, 4]
        let t = try Tensor.from(data: original, shape: [8])

        let asF16 = t.cast(to: .f16)
        XCTAssertEqual(asF16.dtype, .f16)
        XCTAssertEqual(asF16.shape, [8])
        XCTAssertEqual(asF16.numel, 8)

        let backToF32 = asF16.cast(to: .f32)
        XCTAssertEqual(backToF32.dtype, .f32)
        XCTAssertEqual(backToF32.toArray(), original)
    }

    /// F16 has ~3 decimal digits of precision; relative error scales
    /// with magnitude. For |x| ~ 6 we expect ~3e-3 absolute error.
    func testF32ToF16RoundTripIsLossyForIrrational() throws {
        let original: [Float] = [Float.pi, Float.pi * 2, -Float.pi]
        let t = try Tensor.from(data: original, shape: [3])
        let recovered = t.cast(to: .f16).cast(to: .f32).toArray()
        for (a, b) in zip(original, recovered) {
            XCTAssertEqual(a, b, accuracy: 5e-3)
        }
    }

    /// `toArray()` on an F16 tensor materializes F32 values on the fly.
    /// The Swift array is plain Float, no special API needed.
    func testToArrayWorksOnF16() throws {
        let t = try Tensor.from(data: [10, 20, 30, 40], shape: [4]).cast(to: .f16)
        XCTAssertEqual(t.toArray(), [10, 20, 30, 40])
    }

    /// F16 storage uses half the bytes — the resident size of the
    /// underlying TensorStore drops accordingly. We can't measure
    /// memory directly from Swift, but we CAN verify the dtype
    /// reports F16 and that ops still produce correct outputs.
    func testF16TensorComposesWithOps() throws {
        // Two F32 tensors → cast to F16 → sum via existing ops:
        //   sum is computed by reading the F16 store via to_host (which
        //   materializes F32), so ops "just work" without changes.
        let a = try Tensor.from(data: [1, 2, 3, 4], shape: [4]).cast(to: .f16)
        let b = try Tensor.from(data: [10, 20, 30, 40], shape: [4]).cast(to: .f16)
        XCTAssertEqual(a.dtype, .f16)
        XCTAssertEqual(b.dtype, .f16)

        XCTAssertEqual(a.sum, 10.0, accuracy: 1e-3)
        XCTAssertEqual(b.sum, 100.0, accuracy: 1e-3)
    }

    // MARK: - Safetensors keepF16 load path

    /// Save an F16 file, load with `keepF16: true`, verify the loaded
    /// tensors are stored as F16.
    func testSafetensorsLoadKeepsF16() throws {
        let path = tmp("keep_f16")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let w = try Tensor.from(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
        let b = try Tensor.from(data: [10, 20, 30], shape: [3])
        try SafetensorsWeights.save(
            [("w", w), ("b", b)],
            to: path,
            dtype: .f16   // file-side dtype: write as F16
        )

        let loaded = try SafetensorsWeights(path: path, keepF16: true)
        let wLoaded = loaded["w"]
        let bLoaded = loaded["b"]
        XCTAssertNotNil(wLoaded)
        XCTAssertEqual(wLoaded?.dtype, .f16,
            "with keepF16=true, F16 source tensors stay as F16")
        XCTAssertEqual(bLoaded?.dtype, .f16)

        // Values still readable (via on-the-fly up-conversion).
        XCTAssertEqual(wLoaded?.toArray(), [1, 2, 3, 4, 5, 6])
        XCTAssertEqual(bLoaded?.toArray(), [10, 20, 30])
    }

    /// Without keepF16, F16 source tensors are still up-converted to
    /// F32 in the store (the existing default behavior — no regression).
    func testSafetensorsLoadDefaultUpconvertsToF32() throws {
        let path = tmp("default_upconvert")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let w = try Tensor.from(data: [1, 2, 3, 4], shape: [2, 2])
        try SafetensorsWeights.save(
            [("w", w)], to: path, dtype: .f16   // file-side F16
        )

        // Default load: up-converts to F32 in the engine.
        let loaded = try SafetensorsWeights(path: path)
        XCTAssertEqual(loaded["w"]?.dtype, .f32,
            "default load (keepF16=false) up-converts F16 source to F32 storage")
    }

    // MARK: - F16-aware matmul (Phase 2)

    /// Direct F16 matmul: cast a weight tensor to F16 and matmul with
    /// an F32 input. The result should match the F32 baseline within
    /// F16 precision (~1e-2 relative for small accumulators).
    ///
    /// This exercises the F16-aware fast path in `ops::matmul::matmul`,
    /// which reads F16 storage element-by-element instead of cloning a
    /// full F32 weight buffer.
    func testF16AwareMatmulMatchesF32WithinPrecision() throws {
        // Activation x: F32. Weight w: small enough that exact F16
        // representation is possible for integer values.
        let x = try Tensor.from(data: [1, 2, 3, 4], shape: [1, 4])
        let wF32 = try Tensor.from(
            data: [1, 0, 1, 0, 0, 1, 0, 1,
                   1, 0, 1, 0, 0, 1, 0, 1],
            shape: [4, 4]
        )
        let baseline = x.matmul(wF32).toArray()

        let wF16 = wF32.cast(to: .f16)
        XCTAssertEqual(wF16.dtype, .f16)
        let viaF16 = x.matmul(wF16).toArray()

        // For exact-representable F16 weights, the result matches
        // bit-for-bit (the inner-loop conversion `f16.to_f32()` is
        // exact for these values).
        XCTAssertEqual(baseline, viaF16)
    }

    /// Larger random matmul: F16 weight should match F32 result within
    /// F16 precision. Tolerance scales with the inner-product
    /// accumulator size (k=64 here, expected error ~k * eps_f16 ~ 0.06
    /// per output element).
    func testF16MatmulRandomCloseToF32() throws {
        let M = 4, K = 64, N = 8
        var rng = SeededRNG(seed: 12345)
        let xData: [Float] = (0..<(M * K)).map { _ in Float.random(in: -1...1, using: &rng) }
        let wData: [Float] = (0..<(K * N)).map { _ in Float.random(in: -1...1, using: &rng) }

        let x = try Tensor.from(data: xData, shape: [M, K])
        let wF32 = try Tensor.from(data: wData, shape: [K, N])
        let baseline = x.matmul(wF32).toArray()
        let viaF16 = x.matmul(wF32.cast(to: .f16)).toArray()
        XCTAssertEqual(baseline.count, viaF16.count)
        for (a, b) in zip(baseline, viaF16) {
            XCTAssertEqual(a, b, accuracy: 0.1,
                "F16-aware matmul should track F32 within ~k * eps_f16")
        }
    }

    /// F16-loaded weights flow through a downstream op. The op reads
    /// F16 storage transiently as F32 — output is correct.
    func testF16LoadedWeightFeedsMatmul() throws {
        let path = tmp("f16_matmul")
        defer { try? FileManager.default.removeItem(atPath: path) }

        // 2x3 weight saved as F16; load keepF16; matmul against an F32
        // input. Verify the result.
        let w = try Tensor.from(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
        try SafetensorsWeights.save([("W", w)], to: path, dtype: .f16)

        let loaded = try SafetensorsWeights(path: path, keepF16: true)
        let wLoaded = loaded["W"]!
        XCTAssertEqual(wLoaded.dtype, .f16)

        // x [1, 2] @ W [2, 3] → [1, 3].
        // x = [10, 20], W = [[1,2,3],[4,5,6]]
        // result = [10*1 + 20*4, 10*2 + 20*5, 10*3 + 20*6] = [90, 120, 150]
        let x = try Tensor.from(data: [10, 20], shape: [1, 2])
        let y = x.matmul(wLoaded)
        XCTAssertEqual(y.toArray(), [90, 120, 150],
            "matmul with F16 weight should produce same result as F32 weight (within F16 precision; integer values are exact)")
    }
}
