import XCTest
@testable import Transcribe

final class TranscribeTests: XCTestCase {

    // MARK: - FFI sanity

    func testAddBridgesToRust() {
        XCTAssertEqual(Transcribe.add(2, 3), 5)
        XCTAssertEqual(Transcribe.add(-10, 4), -6)
        XCTAssertEqual(Transcribe.add(0, 0), 0)
    }

    // MARK: - Tensor construction

    func testTensorZerosHandle() {
        let t = Tensor.zeros(rows: 3, cols: 4)
        XCTAssertEqual(t.numel, 12)
        XCTAssertEqual(t.sum, 0.0)
        XCTAssertEqual(t.shape, [3, 4])
    }

    func testTensorIotaHandle() {
        let t = Tensor.iota(10)
        XCTAssertEqual(t.numel, 10)
        XCTAssertEqual(t.sum, 45.0)
        XCTAssertEqual(t.shape, [10])
    }

    func testMultipleIndependentTensors() {
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
        // Two Swift references share the same underlying Rust id; release
        // must happen exactly once, when the last ref drops.
        let a = Tensor.iota(10)
        let b = a
        XCTAssertEqual(a.sum, 45.0)
        XCTAssertEqual(b.sum, 45.0)
    }

    func testManyTensorsNoLeak() {
        // 10k alloc/release cycles. The store recycles freed ids via its
        // free_ids list, so memory should stay bounded.
        for _ in 0..<10_000 {
            let t = Tensor.iota(128)
            XCTAssertEqual(t.numel, 128)
            XCTAssertEqual(t.sum, 8128.0) // 127*128/2
        }
    }

    // MARK: - Data movement

    func testFromDataRoundTrip() throws {
        let source: [Float] = [1, 2, 3, 4, 5, 6]
        let t = try Tensor.from(data: source, shape: [2, 3])
        XCTAssertEqual(t.numel, 6)
        XCTAssertEqual(t.sum, 21.0)
        XCTAssertEqual(t.shape, [2, 3])
        XCTAssertEqual(t.toArray(), source)
    }

    func testFromDataRejectsShapeMismatch() {
        XCTAssertThrowsError(try Tensor.from(data: [1, 2, 3, 4, 5], shape: [2, 3])) { error in
            guard case TensorError.shapeMismatch(let expected, let got) = error else {
                return XCTFail("expected .shapeMismatch")
            }
            XCTAssertEqual(expected, 6)
            XCTAssertEqual(got, 5)
        }
    }

    func testFromDataConvenienceInit() {
        // The non-throwing 2-D variant returns nil on mismatch.
        XCTAssertNil(Tensor.from(data: [1, 2, 3, 4, 5], rows: 2, cols: 3))
        XCTAssertNotNil(Tensor.from(data: [1, 2, 3, 4, 5, 6], rows: 2, cols: 3))
    }

    func testToArrayOnIota() {
        let t = Tensor.iota(5)
        XCTAssertEqual(t.toArray(), [0, 1, 2, 3, 4])
    }

    func testToArrayOnZeros() {
        let t = Tensor.zeros(rows: 3, cols: 4)
        XCTAssertEqual(t.toArray(), Array(repeating: Float(0), count: 12))
    }

    func testRoundTripLargeBuffer() {
        let n: UInt32 = 1_000_000
        let t = Tensor.iota(n)
        let arr = t.toArray()
        XCTAssertEqual(arr.count, Int(n))
        XCTAssertEqual(arr.first, 0)
        XCTAssertEqual(arr.last, Float(n - 1))
    }

    // MARK: - Real ops

    func testMatmul2x2() throws {
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = try Tensor.from(data: [1, 2, 3, 4], shape: [2, 2])
        let b = try Tensor.from(data: [5, 6, 7, 8], shape: [2, 2])
        let c = a.matmul(b)

        XCTAssertEqual(c.shape, [2, 2])
        XCTAssertEqual(c.numel, 4)
        XCTAssertEqual(c.toArray(), [19, 22, 43, 50])
    }

    func testMatmul2x3_3x2() throws {
        // [[1,2,3],[4,5,6]] (2x3) @ [[7,8],[9,10],[11,12]] (3x2)
        // = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // = [[58, 64], [139, 154]]
        let a = try Tensor.from(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
        let b = try Tensor.from(data: [7, 8, 9, 10, 11, 12], shape: [3, 2])
        let c = a.matmul(b)

        XCTAssertEqual(c.shape, [2, 2])
        XCTAssertEqual(c.toArray(), [58, 64, 139, 154])
    }

    func testMatmulIdentity() throws {
        // A @ I = A (for any conforming A)
        let a = try Tensor.from(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
        let i = try Tensor.from(data: [1, 0, 0, 0, 1, 0, 0, 0, 1], shape: [3, 3])
        let c = a.matmul(i)
        XCTAssertEqual(c.toArray(), [1, 2, 3, 4, 5, 6])
    }
}
