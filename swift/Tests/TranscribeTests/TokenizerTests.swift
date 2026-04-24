import XCTest
import Foundation
@testable import Transcribe

final class TokenizerTests: XCTestCase {

    /// Write a minimal WordLevel tokenizer.json — just enough to
    /// exercise load + encode + decode without relying on any external
    /// fixture or network. Vocab: hello=0, world=1, foo=2, [UNK]=3.
    private func makeMinimalWordLevel() throws -> String {
        let json = """
        {
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
            "model": {
                "type": "WordLevel",
                "vocab": {"hello": 0, "world": 1, "foo": 2, "[UNK]": 3},
                "unk_token": "[UNK]"
            }
        }
        """
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("ml_transcribe_tokenizer_\(UUID().uuidString).json")
        try json.write(toFile: path, atomically: true, encoding: .utf8)
        return path
    }

    // MARK: - Correctness

    func testEncodeKnownVocab() throws {
        let path = try makeMinimalWordLevel()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let tok = try Tokenizer(path: path)
        let ids = try tok.encode("hello world", addSpecialTokens: false)
        XCTAssertEqual(ids, [0, 1])
    }

    func testEncodeMapsUnknownToUnkToken() throws {
        let path = try makeMinimalWordLevel()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let tok = try Tokenizer(path: path)
        // "bar" is not in vocab → UNK (id 3).
        let ids = try tok.encode("hello bar world", addSpecialTokens: false)
        XCTAssertEqual(ids, [0, 3, 1])
    }

    func testDecodeRoundTripsKnownIds() throws {
        let path = try makeMinimalWordLevel()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let tok = try Tokenizer(path: path)
        let text = try tok.decode([0, 1], skipSpecialTokens: false)
        XCTAssertTrue(text.contains("hello"))
        XCTAssertTrue(text.contains("world"))
    }

    func testEncodeDecodeRoundTrip() throws {
        let path = try makeMinimalWordLevel()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let tok = try Tokenizer(path: path)
        let input = "hello world foo"
        let ids = try tok.encode(input, addSpecialTokens: false)
        let decoded = try tok.decode(ids, skipSpecialTokens: false)

        // WordLevel tokenizers don't preserve whitespace exactly, but
        // the tokens themselves should round-trip.
        for word in ["hello", "world", "foo"] {
            XCTAssertTrue(decoded.contains(word),
                "decoded text '\(decoded)' missing '\(word)'")
        }
    }

    func testEmptyInputProducesEmptyOutput() throws {
        let path = try makeMinimalWordLevel()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let tok = try Tokenizer(path: path)
        let ids = try tok.encode("", addSpecialTokens: false)
        XCTAssertEqual(ids, [])
    }

    func testDecodeEmptyArrayIsEmptyString() throws {
        let path = try makeMinimalWordLevel()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let tok = try Tokenizer(path: path)
        let text = try tok.decode([], skipSpecialTokens: false)
        XCTAssertEqual(text, "")
    }

    // MARK: - Composition with embedding

    /// The canonical LM input path: text -> tokenizer -> embedding
    /// table -> tensor. Verifies the two FFI surfaces compose without
    /// ownership bugs across the boundary.
    func testTokenizerFeedsEmbedding() throws {
        let path = try makeMinimalWordLevel()
        defer { try? FileManager.default.removeItem(atPath: path) }

        // Vocab size = 4 (hello, world, foo, [UNK]); embed dim = 3.
        // Row i = [i*10 + 0, i*10 + 1, i*10 + 2].
        let vocab = 4, dim = 3
        var tableData: [Float] = []
        for i in 0..<vocab {
            for j in 0..<dim {
                tableData.append(Float(i * 10 + j))
            }
        }
        let table = try Tensor.from(data: tableData, shape: [vocab, dim])

        let tok = try Tokenizer(path: path)
        let rawIds = try tok.encode("hello world", addSpecialTokens: false)
        XCTAssertEqual(rawIds, [0, 1])

        let tokens = rawIds.map { Int($0) }
        let emb = table.embed(tokens: tokens)
        XCTAssertEqual(emb.shape, [1, 2, dim])

        // Expected: [[row 0], [row 1]] = [[0, 1, 2], [10, 11, 12]].
        XCTAssertEqual(emb.toArray(), [0, 1, 2, 10, 11, 12])
    }

    // MARK: - Lifecycle

    func testLoadFailureOnMissingFile() {
        let path = "/does/not/exist/tokenizer.json"
        XCTAssertThrowsError(try Tokenizer(path: path)) {
            guard case TokenizerError.loadFailed = $0 else {
                return XCTFail("expected .loadFailed, got \($0)")
            }
        }
    }

    func testRepeatedEncodeDoesNotLeak() throws {
        // Each encode allocates+frees a u32 buffer in Rust. Running a
        // lot of them in a tight loop shouldn't crash or grow memory
        // unbounded (we can't easily assert memory, but a crash or
        // double-free would surface here).
        let path = try makeMinimalWordLevel()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let tok = try Tokenizer(path: path)
        for _ in 0..<1000 {
            let ids = try tok.encode("hello world foo", addSpecialTokens: false)
            XCTAssertEqual(ids.count, 3)
        }
    }
}
