import XCTest
import Foundation
@testable import Transcribe

final class KVCacheTests: XCTestCase {

    // MARK: - Lifecycle

    func testNewCacheStartsEmpty() {
        let cache = KVCache(config: .init(
            batchSize: 1, numHeads: 2, headDim: 4, maxSeqLen: 16
        ))
        XCTAssertEqual(cache.length, 0)
    }

    func testMultipleCachesAreIndependent() {
        let c1 = KVCache(config: .init(numHeads: 1, headDim: 2, maxSeqLen: 4))
        let c2 = KVCache(config: .init(numHeads: 1, headDim: 2, maxSeqLen: 4))

        XCTAssertEqual(c1.length, 0)
        XCTAssertEqual(c2.length, 0)

        do {
            let k = try Tensor.from(data: [1, 2], shape: [1, 1, 1, 2])
            let v = try Tensor.from(data: [3, 4], shape: [1, 1, 1, 2])
            try c1.append(key: k, value: v)
        } catch {
            XCTFail("append failed: \(error)")
        }

        XCTAssertEqual(c1.length, 1)
        XCTAssertEqual(c2.length, 0, "appending to c1 must not affect c2")
    }

    func testResetClearsCache() throws {
        let cache = KVCache(config: .init(numHeads: 1, headDim: 2, maxSeqLen: 4))
        let k = try Tensor.from(data: [1, 2], shape: [1, 1, 1, 2])
        let v = try Tensor.from(data: [3, 4], shape: [1, 1, 1, 2])
        try cache.append(key: k, value: v)
        try cache.append(key: k, value: v)
        XCTAssertEqual(cache.length, 2)
        cache.reset()
        XCTAssertEqual(cache.length, 0)
    }

    // MARK: - append

    func testAppendGrowsLength() throws {
        let cache = KVCache(config: .init(numHeads: 1, headDim: 2, maxSeqLen: 8))
        let k = try Tensor.from(data: [1, 2], shape: [1, 1, 1, 2])
        let v = try Tensor.from(data: [3, 4], shape: [1, 1, 1, 2])
        for expected in 1...5 {
            try cache.append(key: k, value: v)
            XCTAssertEqual(cache.length, expected)
        }
    }

    func testAppendWrongShapeThrows() throws {
        let cache = KVCache(config: .init(numHeads: 2, headDim: 4, maxSeqLen: 4))
        // Expected shape is [1, 2, 1, 4]; give it [1, 1, 1, 4].
        let k = try Tensor.from(data: [0, 0, 0, 0], shape: [1, 1, 1, 4])
        let v = try Tensor.from(data: [0, 0, 0, 0], shape: [1, 1, 1, 4])
        XCTAssertThrowsError(try cache.append(key: k, value: v)) { error in
            guard case KVCacheError.shapeMismatch = error else {
                return XCTFail("expected .shapeMismatch, got \(error)")
            }
        }
    }

    func testAppendBeyondMaxSeqLenThrows() throws {
        let cache = KVCache(config: .init(numHeads: 1, headDim: 2, maxSeqLen: 3))
        let k = try Tensor.from(data: [1, 2], shape: [1, 1, 1, 2])
        let v = try Tensor.from(data: [3, 4], shape: [1, 1, 1, 2])
        try cache.append(key: k, value: v) // 1
        try cache.append(key: k, value: v) // 2
        try cache.append(key: k, value: v) // 3 (at capacity)
        XCTAssertThrowsError(try cache.append(key: k, value: v))  // 4th -> error
    }

    // MARK: - append_and_decode

    func testAppendAndDecodeReturnsExpectedShape() throws {
        let cfg = KVCache.Config(numHeads: 2, headDim: 4, maxSeqLen: 8)
        let cache = KVCache(config: cfg)

        // Single decode step: q, k, v all [1, 2, 1, 4].
        let q = try Tensor.from(data: Array(repeating: Float(1), count: 8),
                                shape: [1, 2, 1, 4])
        let k = try Tensor.from(data: Array(repeating: Float(1), count: 8),
                                shape: [1, 2, 1, 4])
        let v = try Tensor.from(data: Array(repeating: Float(2), count: 8),
                                shape: [1, 2, 1, 4])

        let out = try cache.appendAndDecode(
            query: q, key: k, value: v, scale: 1.0 / 2.0
        )
        XCTAssertEqual(out.shape, [1, 2, 1, 4])
        XCTAssertEqual(cache.length, 1)

        // With a cache of length 1, attention weights are softmax([qk])=1,
        // so output = 1 * V = V = [2, 2, 2, 2, ...].
        for e in out.toArray() {
            XCTAssertEqual(e, 2.0, accuracy: 1e-5)
        }
    }

    func testDecodeStepsAccumulatePastValues() throws {
        // After two steps, the attention output should be a softmax
        // blend of the two cached V values (identical K keys → uniform
        // softmax weights ~0.5 each → mean of the Vs).
        let cfg = KVCache.Config(numHeads: 1, headDim: 2, maxSeqLen: 4)
        let cache = KVCache(config: cfg)
        let q = try Tensor.from(data: [1, 0], shape: [1, 1, 1, 2])
        let k = try Tensor.from(data: [1, 0], shape: [1, 1, 1, 2])

        // Step 1: V = [10, 10]
        let v1 = try Tensor.from(data: [10, 10], shape: [1, 1, 1, 2])
        _ = try cache.appendAndDecode(query: q, key: k, value: v1, scale: 1.0)

        // Step 2: V = [20, 20]. Past K equals current K, so softmax is
        // [0.5, 0.5]; output should be 0.5*10 + 0.5*20 = 15.
        let v2 = try Tensor.from(data: [20, 20], shape: [1, 1, 1, 2])
        let out = try cache.appendAndDecode(query: q, key: k, value: v2, scale: 1.0)

        XCTAssertEqual(cache.length, 2)
        for e in out.toArray() {
            XCTAssertEqual(e, 15.0, accuracy: 1e-4)
        }
    }

    // MARK: - Integration: 3-step autoregressive loop

    /// A mini autoregressive decoding loop: generate three "steps"
    /// and verify the KV cache length grows to 3 and each step
    /// produces well-formed output.
    func testAutoregressiveLoopSmokeTest() throws {
        let numHeads = 2, headDim = 4
        let cache = KVCache(config: .init(
            numHeads: numHeads, headDim: headDim, maxSeqLen: 8
        ))

        var rng = SeededRNG(seed: 1234)
        let scale = 1.0 / sqrt(Float(headDim))
        for step in 1...3 {
            let count = numHeads * headDim
            let q = try Tensor.from(
                data: (0..<count).map { _ in Float.random(in: -1...1, using: &rng) },
                shape: [1, numHeads, 1, headDim]
            )
            let k = try Tensor.from(
                data: (0..<count).map { _ in Float.random(in: -1...1, using: &rng) },
                shape: [1, numHeads, 1, headDim]
            )
            let v = try Tensor.from(
                data: (0..<count).map { _ in Float.random(in: -1...1, using: &rng) },
                shape: [1, numHeads, 1, headDim]
            )

            let out = try cache.appendAndDecode(
                query: q, key: k, value: v, scale: scale
            )
            XCTAssertEqual(out.shape, [1, numHeads, 1, headDim])
            XCTAssertEqual(cache.length, step)
            for f in out.toArray() {
                XCTAssertTrue(f.isFinite)
            }
        }
    }
}
