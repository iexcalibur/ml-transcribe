import XCTest
import Foundation
@testable import Transcribe

/// End-to-end test with a *real* MNIST MLP trained in PyTorch.
///
/// Fixtures are produced by `scripts/train_mnist.py` and committed to
/// `swift/Tests/Fixtures/`:
///
///   mnist_mlp.safetensors    — trained weights (F32)
///   mnist_samples.json       — 10 test images (one per digit class)
///
/// The model used in training is a 784 -> 128 -> 10 MLP identical in
/// shape to `MLP2`. After 5 epochs it reaches ~97.7% test accuracy,
/// so the 10 samples here should nearly always classify correctly.
final class RealMNISTTests: XCTestCase {

    /// Resolve a fixture path at runtime using `#filePath`, which gives
    /// the compile-time absolute path of this source file. Works for
    /// both `swift test` on macOS and `xcodebuild test` on iOS Sim —
    /// the test bundle can still read files at the host-side path.
    private func fixturePath(_ name: String) -> String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
            .appendingPathComponent(name)
            .path
    }

    private struct Sample: Decodable {
        let label: Int
        let pixels: [Float]
    }

    func testRealTrainedModelClassifiesRealDigits() throws {
        let weightsPath = fixturePath("mnist_mlp.safetensors")
        let samplesPath = fixturePath("mnist_samples.json")

        // Sanity check: make sure the fixtures are actually present
        // (if someone deletes them without rerunning train_mnist.py).
        XCTAssertTrue(FileManager.default.fileExists(atPath: weightsPath),
                      "missing fixture: run scripts/train_mnist.py")
        XCTAssertTrue(FileManager.default.fileExists(atPath: samplesPath),
                      "missing fixture: run scripts/train_mnist.py")

        // Load: safetensors file -> TensorStore -> MLP2 wrapper.
        let weights = try SafetensorsWeights(path: weightsPath)
        XCTAssertEqual(Set(weights.keys),
                       ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"])
        let mlp = try MLP2(weights: weights)
        XCTAssertEqual(mlp.inDim, 784)
        XCTAssertEqual(mlp.hiddenDim, 128)
        XCTAssertEqual(mlp.outDim, 10)

        // Decode the per-class test digits from the JSON fixture.
        let samplesData = try Data(contentsOf: URL(fileURLWithPath: samplesPath))
        let samples = try JSONDecoder().decode([Sample].self, from: samplesData)
        XCTAssertEqual(samples.count, 10, "fixture should have one sample per class")

        // Classify each. A 97.7%-accurate model on 10 pre-selected
        // per-class digits almost certainly gets all 10 right; we set
        // the threshold to 9/10 to survive the rare edge case.
        var correct = 0
        var predictions: [(expected: Int, got: Int)] = []
        for sample in samples {
            let predicted = try mlp.classify(sample.pixels)
            predictions.append((sample.label, predicted))
            if predicted == sample.label { correct += 1 }
        }

        // Diagnostic output on failure (Xcode shows this in the log).
        if correct < 10 {
            for (expected, got) in predictions where expected != got {
                print("misclassified: expected=\(expected) got=\(got)")
            }
        }
        XCTAssertGreaterThanOrEqual(correct, 9,
            "only \(correct)/10 correct — the FFI / ops / loader pipeline likely drifted")
    }

    /// Running classification twice on the same input should give the
    /// same prediction — no hidden global state, no RNG in the forward
    /// path.
    func testClassificationIsDeterministic() throws {
        let weights = try SafetensorsWeights(path: fixturePath("mnist_mlp.safetensors"))
        let mlp = try MLP2(weights: weights)
        let samplesData = try Data(contentsOf: URL(fileURLWithPath: fixturePath("mnist_samples.json")))
        let samples = try JSONDecoder().decode([Sample].self, from: samplesData)

        let first = try mlp.classify(samples[0].pixels)
        for _ in 0..<5 {
            XCTAssertEqual(try mlp.classify(samples[0].pixels), first)
        }
    }
}
