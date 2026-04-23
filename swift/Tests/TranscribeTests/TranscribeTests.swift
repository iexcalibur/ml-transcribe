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

    // MARK: - Elementwise + activations

    func testAdd() throws {
        let a = try Tensor.from(data: [1, 2, 3, 4], shape: [2, 2])
        let b = try Tensor.from(data: [10, 20, 30, 40], shape: [2, 2])
        XCTAssertEqual(a.add(b).toArray(), [11, 22, 33, 44])
    }

    func testMul() throws {
        let a = try Tensor.from(data: [1, 2, 3, 4], shape: [2, 2])
        let b = try Tensor.from(data: [10, 20, 30, 40], shape: [2, 2])
        XCTAssertEqual(a.mul(b).toArray(), [10, 40, 90, 160])
    }

    func testRelu() throws {
        let a = try Tensor.from(data: [-1, 0, 1, -2, 3, -4], shape: [2, 3])
        XCTAssertEqual(a.relu().toArray(), [0, 0, 1, 0, 3, 0])
    }

    func testSoftmaxRowsSumToOne() throws {
        // softmax along last dim → each row should sum to 1.0
        // and, because input is strictly increasing per row, output
        // must also be strictly increasing per row.
        let a = try Tensor.from(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
        let s = a.softmax(dim: -1)
        let arr = s.toArray()
        XCTAssertEqual(s.shape, [2, 3])

        let row0Sum = arr[0] + arr[1] + arr[2]
        let row1Sum = arr[3] + arr[4] + arr[5]
        XCTAssertEqual(row0Sum, 1.0, accuracy: 1e-5)
        XCTAssertEqual(row1Sum, 1.0, accuracy: 1e-5)

        XCTAssertLessThan(arr[0], arr[1])
        XCTAssertLessThan(arr[1], arr[2])
        XCTAssertLessThan(arr[3], arr[4])
        XCTAssertLessThan(arr[4], arr[5])
    }

    func testSoftmaxKnownValues() throws {
        // softmax([1,2,3]) has a closed form. Check the well-known
        // triplet (rounded): [0.09003, 0.24473, 0.66524].
        let a = try Tensor.from(data: [1, 2, 3], shape: [3])
        let s = a.softmax(dim: 0).toArray()
        XCTAssertEqual(s[0], 0.09003, accuracy: 1e-4)
        XCTAssertEqual(s[1], 0.24473, accuracy: 1e-4)
        XCTAssertEqual(s[2], 0.66524, accuracy: 1e-4)
    }

    // MARK: - Composed ops (sanity check: chaining works)

    func testMlpStyleChain() throws {
        // Mini "layer": relu(x @ W + b)  — the classic MLP building block.
        let x = try Tensor.from(data: [1, -2, 3, -4], shape: [1, 4])
        let w = try Tensor.from(
            data: [1, 0, 1, 1, 1, 0, 1, 1],
            shape: [4, 2]
        )
        let b = try Tensor.from(data: [-1, 1], shape: [1, 2])

        // Manual calculation:
        //   x @ W = [[1*1 + -2*1 + 3*1 + -4*1, 1*0 + -2*1 + 3*0 + -4*1]]
        //         = [[-2, -6]]
        //   + b   = [[-3, -5]]
        //   relu  = [[0, 0]]
        let out = x.matmul(w).add(b).relu()
        XCTAssertEqual(out.shape, [1, 2])
        XCTAssertEqual(out.toArray(), [0, 0])
    }

    // MARK: - Safetensors

    /// Returns a unique tmp file path for this test run.
    private func tmpSafetensorsPath(_ name: String) -> String {
        let dir = NSTemporaryDirectory()
        return (dir as NSString).appendingPathComponent("ml_transcribe_\(name).safetensors")
    }

    func testSafetensorsRoundTrip() throws {
        let path = tmpSafetensorsPath("roundtrip")
        defer { try? FileManager.default.removeItem(atPath: path) }

        // Write a tiny fixture through Swift's save API.
        let w = try Tensor.from(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
        let b = try Tensor.from(data: [10, 20, 30], shape: [3])
        try SafetensorsWeights.save(
            [("weight", w), ("bias", b)],
            to: path
        )

        // Load back through SafetensorsWeights.
        let loaded = try SafetensorsWeights(path: path)
        XCTAssertEqual(loaded.count, 2)
        XCTAssertEqual(loaded.keys, ["bias", "weight"]) // sorted

        let wRead = loaded["weight"]
        XCTAssertNotNil(wRead)
        XCTAssertEqual(wRead!.shape, [2, 3])
        XCTAssertEqual(wRead!.toArray(), [1, 2, 3, 4, 5, 6])

        let bRead = loaded["bias"]
        XCTAssertNotNil(bRead)
        XCTAssertEqual(bRead!.shape, [3])
        XCTAssertEqual(bRead!.toArray(), [10, 20, 30])

        // Missing key returns nil.
        XCTAssertNil(loaded["does_not_exist"])
    }

    func testSafetensorsLoadFailure() {
        XCTAssertThrowsError(try SafetensorsWeights(path: "/does/not/exist.safetensors"))
    }

    func testSafetensorsBorrowedTensorsUsableInOps() throws {
        // Real-world usage: load weights, use them in forward pass.
        let path = tmpSafetensorsPath("forward")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let w = try Tensor.from(data: [1, 0, 0, 1], shape: [2, 2]) // identity
        let b = try Tensor.from(data: [10, 20], shape: [1, 2])
        try SafetensorsWeights.save(
            [("W", w), ("b", b)],
            to: path
        )

        let weights = try SafetensorsWeights(path: path)
        let W = weights["W"]!
        let B = weights["b"]!

        // x @ I + b = x + b, where x = [[3, 4]].
        let x = try Tensor.from(data: [3, 4], shape: [1, 2])
        let out = x.matmul(W).add(B)
        XCTAssertEqual(out.toArray(), [13, 24])
    }

    func testSafetensorsTensorsOutliveHandle() throws {
        // The Tensor view must keep the SafetensorsWeights alive via
        // its strong `owner` ref. Dropping `loaded` here while still
        // holding `w` must NOT cause a use-after-free when w is read.
        let path = tmpSafetensorsPath("outlive")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let w0 = try Tensor.from(data: [7, 8, 9], shape: [3])
        try SafetensorsWeights.save([("w", w0)], to: path)

        var loaded: SafetensorsWeights? = try SafetensorsWeights(path: path)
        let borrowed = loaded!["w"]!
        loaded = nil // Release our explicit ref to the collection.

        // `borrowed` retains `loaded` internally — the file's data must
        // still be readable.
        XCTAssertEqual(borrowed.toArray(), [7, 8, 9])
    }
}
