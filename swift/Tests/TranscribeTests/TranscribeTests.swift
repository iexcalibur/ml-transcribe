import XCTest
@testable import Transcribe

final class TranscribeTests: XCTestCase {

    // MARK: - Value-returning helpers (Milestone 1 + 2a)

    func testAddBridgesToRust() {
        XCTAssertEqual(Transcribe.add(2, 3), 5)
        XCTAssertEqual(Transcribe.add(-10, 4), -6)
        XCTAssertEqual(Transcribe.add(0, 0), 0)
    }

    func testTensorZerosSum() {
        XCTAssertEqual(Transcribe.tensorZerosSum(rows: 3, cols: 4), 0.0)
        XCTAssertEqual(Transcribe.tensorZerosSum(rows: 100, cols: 100), 0.0)
    }

    func testTensorIotaSum() {
        XCTAssertEqual(Transcribe.tensorIotaSum(n: 10), 45.0)
        XCTAssertEqual(Transcribe.tensorIotaSum(n: 100), 4950.0)
    }

    // MARK: - Handle-based Tensor (Milestone 2b)

    func testTensorZerosHandle() {
        let t = Tensor.zeros(rows: 3, cols: 4)
        XCTAssertEqual(t.numel, 12)
        XCTAssertEqual(t.sum, 0.0)
        // `t` is freed at end of scope via deinit.
    }

    func testTensorIotaHandle() {
        let t = Tensor.iota(10)
        XCTAssertEqual(t.numel, 10)
        XCTAssertEqual(t.sum, 45.0)
    }

    func testMultipleIndependentTensors() {
        // Each tensor owns its own Rust allocation; freeing one must not
        // affect the others.
        let a = Tensor.zeros(rows: 2, cols: 3)
        let b = Tensor.iota(5)
        let c = Tensor.iota(100)

        XCTAssertEqual(a.numel, 6)
        XCTAssertEqual(a.sum, 0.0)

        XCTAssertEqual(b.numel, 5)
        XCTAssertEqual(b.sum, 10.0) // 0+1+2+3+4

        XCTAssertEqual(c.numel, 100)
        XCTAssertEqual(c.sum, 4950.0)
    }

    func testReferenceSemantics() {
        // Two Swift references to the same underlying Rust tensor.
        // Only one free() should fire — when both refs drop.
        let a = Tensor.iota(10)
        let b = a  // reference copy, same handle
        XCTAssertEqual(a.sum, 45.0)
        XCTAssertEqual(b.sum, 45.0)
        // When this function returns, both `a` and `b` go out of scope,
        // but ARC ensures deinit runs exactly once on the last release.
    }

    func testManyTensorsNoLeak() {
        // Allocate and drop 10k tensors; if free is wired up correctly,
        // memory is bounded. A leak would OOM under Instruments; here we
        // just confirm we don't crash and the sums are consistent.
        for _ in 0..<10_000 {
            let t = Tensor.iota(128)
            XCTAssertEqual(t.numel, 128)
            XCTAssertEqual(t.sum, 8128.0) // 127*128/2
        }
    }

    // MARK: - Data movement (Milestone 3)

    func testFromDataRoundTrip() {
        let source: [Float] = [1, 2, 3, 4, 5, 6]
        guard let t = Tensor.from(data: source, rows: 2, cols: 3) else {
            return XCTFail("from(data:) returned nil")
        }
        XCTAssertEqual(t.numel, 6)
        XCTAssertEqual(t.sum, 21.0)
        // Swift -> Rust -> Swift should preserve every element exactly.
        XCTAssertEqual(t.toArray(), source)
    }

    func testFromDataRejectsShapeMismatch() {
        // 5 elements can't fit a 2x3 shape.
        let t = Tensor.from(data: [1, 2, 3, 4, 5], rows: 2, cols: 3)
        XCTAssertNil(t, "Rust should reject size mismatch; Swift returns nil")
    }

    func testToArrayOnIota() {
        let t = Tensor.iota(5)
        XCTAssertEqual(t.toArray(), [0, 1, 2, 3, 4])
    }

    func testToArrayOnZeros() {
        let t = Tensor.zeros(rows: 3, cols: 4)
        XCTAssertEqual(t.toArray(), Array(repeating: Float(0), count: 12))
    }

    func testWithUnsafeDataBorrow() {
        let t = Tensor.iota(100)
        // Zero-copy max via the borrow API.
        let maxValue = t.withUnsafeData { buffer in
            buffer.max() ?? .nan
        }
        XCTAssertEqual(maxValue, 99.0)
    }

    func testWithUnsafeDataRethrows() {
        // The closure-based borrow API rethrows errors from the body.
        struct Boom: Error {}
        let t = Tensor.iota(10)
        XCTAssertThrowsError(
            try t.withUnsafeData { _ in throw Boom() }
        )
        // After the throw, the tensor should still be usable (no leaks,
        // no corrupted state).
        XCTAssertEqual(t.sum, 45.0)
    }

    func testRoundTripLargeBuffer() {
        // 1M elements: verifies copy_into doesn't truncate and the
        // unsafeUninitializedCapacity pattern respects initializedCount.
        let n: UInt32 = 1_000_000
        let t = Tensor.iota(n)
        let arr = t.toArray()
        XCTAssertEqual(arr.count, Int(n))
        XCTAssertEqual(arr.first, 0)
        XCTAssertEqual(arr.last, Float(n - 1))
    }
}
