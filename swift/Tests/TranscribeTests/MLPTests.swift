import XCTest
import Foundation
@testable import Transcribe

/// End-to-end tests: safetensors -> SafetensorsWeights -> MLP2 -> classify.
final class MLPTests: XCTestCase {

    private func tmp(_ name: String) -> String {
        (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("ml_transcribe_mlp_\(name).safetensors")
    }

    /// Save a set of Tensors (owned, freshly built) to a safetensors file.
    private func writeFixture(
        _ pairs: [(String, [Float], [Int])],
        to path: String
    ) throws {
        let tensors: [(String, Tensor)] = try pairs.map { (name, data, shape) in
            (name, try Tensor.from(data: data, shape: shape))
        }
        try SafetensorsWeights.save(tensors, to: path)
    }

    // MARK: - Crafted identity network (exact correctness)

    /// A 4-in / 4-hidden / 4-out MLP with identity weights should act
    /// as the identity function on strictly positive inputs (ReLU is a
    /// no-op for those), so `argmax(x)` must equal the index of the
    /// largest element of `x`.
    func testIdentityNetworkArgmax() throws {
        let path = tmp("identity")
        defer { try? FileManager.default.removeItem(atPath: path) }

        // 4x4 identity (row-major)
        let I: [Float] = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        let zeros4: [Float] = [0, 0, 0, 0]

        try writeFixture([
            ("fc1.weight", I,      [4, 4]),
            ("fc1.bias",   zeros4, [1, 4]),
            ("fc2.weight", I,      [4, 4]),
            ("fc2.bias",   zeros4, [1, 4]),
        ], to: path)

        let weights = try SafetensorsWeights(path: path)
        let mlp = try MLP2(weights: weights)
        XCTAssertEqual(mlp.inDim, 4)
        XCTAssertEqual(mlp.hiddenDim, 4)
        XCTAssertEqual(mlp.outDim, 4)

        // Each test input: the largest element's index == expected class.
        XCTAssertEqual(try mlp.classify([10, 5, 3, 1]),   0)
        XCTAssertEqual(try mlp.classify([0, 10, 0, 0]),   1)
        XCTAssertEqual(try mlp.classify([2, 3, 7, 4]),    2)
        XCTAssertEqual(try mlp.classify([1, 1, 1, 10]),   3)
    }

    /// Same architecture, but now the second layer scrambles class
    /// indices via a permutation matrix. This proves the second matmul
    /// is actually being applied — not just passing activations through.
    func testPermutationSecondLayer() throws {
        let path = tmp("permute")
        defer { try? FileManager.default.removeItem(atPath: path) }

        // Permutation P: class i <- input feature π(i), where
        // π = [2, 0, 3, 1]. Row j of P picks column π(j) of input.
        //
        // P in row-major [in=4, out=4] means P[i][j] = 1 iff j == π⁻¹(i),
        // i.e. (input @ P)[0][j] == input[0][π(j)].
        //
        // For π = [2, 0, 3, 1], the inverse is π⁻¹ = [1, 3, 0, 2].
        // So P[i][j] = 1 iff j == π⁻¹(i):
        //   row 0 (i=0): j=1 -> [0,1,0,0]
        //   row 1 (i=1): j=3 -> [0,0,0,1]
        //   row 2 (i=2): j=0 -> [1,0,0,0]
        //   row 3 (i=3): j=2 -> [0,0,1,0]
        let P: [Float] = [
            0,1,0,0,
            0,0,0,1,
            1,0,0,0,
            0,0,1,0,
        ]
        let I: [Float] = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        let zeros4: [Float] = [0, 0, 0, 0]

        try writeFixture([
            ("fc1.weight", I,      [4, 4]), // identity first layer
            ("fc1.bias",   zeros4, [1, 4]),
            ("fc2.weight", P,      [4, 4]), // permutation second layer
            ("fc2.bias",   zeros4, [1, 4]),
        ], to: path)

        let mlp = try MLP2(weights: try SafetensorsWeights(path: path))

        // If input's biggest element is at position π(j), the output's
        // biggest element is at position j.
        //
        // Input [10, 1, 1, 1]: biggest at position 0 = π(1), so output's
        // argmax should be 1.
        XCTAssertEqual(try mlp.classify([10, 1, 1, 1]), 1)
        // Biggest at position 1 = π(3) → output argmax = 3.
        XCTAssertEqual(try mlp.classify([1, 10, 1, 1]), 3)
        // Biggest at position 2 = π(0) → output argmax = 0.
        XCTAssertEqual(try mlp.classify([1, 1, 10, 1]), 0)
        // Biggest at position 3 = π(2) → output argmax = 2.
        XCTAssertEqual(try mlp.classify([1, 1, 1, 10]), 2)
    }

    /// ReLU actually fires: negative inputs should be zeroed out.
    func testReluKillsNegatives() throws {
        let path = tmp("relu")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let I: [Float] = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        let zeros4: [Float] = [0, 0, 0, 0]

        try writeFixture([
            ("fc1.weight", I,      [4, 4]),
            ("fc1.bias",   zeros4, [1, 4]),
            ("fc2.weight", I,      [4, 4]),
            ("fc2.bias",   zeros4, [1, 4]),
        ], to: path)

        let mlp = try MLP2(weights: try SafetensorsWeights(path: path))

        // Input [-10, -5, 3, -1]: ReLU zeros the first, second, fourth.
        // Post-ReLU hidden: [0, 0, 3, 0]. Class 2 is the largest (3),
        // but ties exist (0, 0, 0). Swift's `max(by:)` with `<` picks
        // the first maximum, so with ties we expect index 0 — UNLESS
        // some non-zero element wins, which is 3 at index 2.
        XCTAssertEqual(try mlp.classify([-10, -5, 3, -1]), 2)
    }

    // MARK: - MNIST-scale (real dims, random weights)

    /// A 784-input / 128-hidden / 10-output MLP with seeded random
    /// weights. We don't care about accuracy (untrained), only that:
    ///   - every shape checks out,
    ///   - forward runs without crashing at this scale,
    ///   - argmax returns a valid class in 0..<10.
    func testMNISTScaleForwardIsValid() throws {
        let path = tmp("mnist_scale")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let inDim = 784
        let hiddenDim = 128
        let outDim = 10

        var rng = SeededRNG(seed: 42)
        let w1: [Float] = (0..<(inDim * hiddenDim)).map { _ in
            Float.random(in: -0.1...0.1, using: &rng)
        }
        let b1: [Float] = Array(repeating: 0, count: hiddenDim)
        let w2: [Float] = (0..<(hiddenDim * outDim)).map { _ in
            Float.random(in: -0.1...0.1, using: &rng)
        }
        let b2: [Float] = Array(repeating: 0, count: outDim)

        try writeFixture([
            ("fc1.weight", w1, [inDim, hiddenDim]),
            ("fc1.bias",   b1, [1, hiddenDim]),
            ("fc2.weight", w2, [hiddenDim, outDim]),
            ("fc2.bias",   b2, [1, outDim]),
        ], to: path)

        let weights = try SafetensorsWeights(path: path)
        let mlp = try MLP2(weights: weights)
        XCTAssertEqual(mlp.inDim, 784)
        XCTAssertEqual(mlp.hiddenDim, 128)
        XCTAssertEqual(mlp.outDim, 10)

        // "Image": a plausible normalized MNIST-ish input.
        let input: [Float] = (0..<784).map { _ in Float.random(in: 0...1, using: &rng) }
        let predicted = try mlp.classify(input)
        XCTAssertTrue((0..<10).contains(predicted), "got class \(predicted)")

        // Determinism: same input should always give the same class.
        let again = try mlp.classify(input)
        XCTAssertEqual(predicted, again)
    }

    /// Shape validation surfaces a useful error when a weight is the
    /// wrong shape, so debugging a wrong safetensors file is sane.
    func testShapeMismatchSurfacesLoadError() throws {
        let path = tmp("wrong_shape")
        defer { try? FileManager.default.removeItem(atPath: path) }

        // fc1.bias should be [1, 4], not [4].
        try writeFixture([
            ("fc1.weight", [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1], [4, 4]),
            ("fc1.bias",   [0,0,0,0],                           [4]),       // wrong
            ("fc2.weight", [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1], [4, 4]),
            ("fc2.bias",   [0,0,0,0],                           [1, 4]),
        ], to: path)

        let weights = try SafetensorsWeights(path: path)
        XCTAssertThrowsError(try MLP2(weights: weights)) { error in
            guard let err = error as? MLP2.LoadError else {
                return XCTFail("expected MLP2.LoadError, got \(error)")
            }
            if case .unexpectedShape(let name, _, _) = err {
                XCTAssertEqual(name, "fc1.bias")
            } else {
                XCTFail("expected .unexpectedShape, got \(err)")
            }
        }
    }
}
