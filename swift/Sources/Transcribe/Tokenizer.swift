import Foundation
import TranscribeFFIC

/// Errors from tokenizer operations.
public enum TokenizerError: Error, CustomStringConvertible {
    case loadFailed(path: String)
    case encodeFailed(text: String)
    case decodeFailed(tokenCount: Int)

    public var description: String {
        switch self {
        case .loadFailed(let p):      return "tokenizer load failed: \(p)"
        case .encodeFailed(let t):    return "tokenizer encode failed (text length \(t.count))"
        case .decodeFailed(let n):    return "tokenizer decode failed (\(n) tokens)"
        }
    }
}

/// Thin wrapper over the HuggingFace `tokenizers` Rust crate.
///
/// Supports any tokenizer serialized to the standard `tokenizer.json`
/// format — BPE (GPT-style), WordPiece (BERT), Unigram/SentencePiece
/// (LLaMA, T5, Cohere), WordLevel, etc. The format encodes the model
/// type, vocab, merges, pre/post-processing, and special tokens, so
/// the Swift API doesn't need to know which kind you're using.
public final class Tokenizer {
    /// `*mut Tokenizer` from the Rust FFI.
    private let handle: OpaquePointer

    /// Load a tokenizer from a `tokenizer.json` file on disk.
    public convenience init(path: URL) throws {
        try self.init(path: path.path)
    }

    public init(path: String) throws {
        let maybeHandle: OpaquePointer? = path.withCString { cstr in
            let bytes = UnsafeRawPointer(cstr).assumingMemoryBound(to: UInt8.self)
            return ml_engine_tokenizer_load(bytes, UInt64(strlen(cstr)))
        }
        guard let h = maybeHandle else {
            throw TokenizerError.loadFailed(path: path)
        }
        self.handle = h
    }

    /// Encode a string to an array of token ids.
    ///
    /// - `addSpecialTokens`: whether to add model-specific prefix/suffix
    ///   tokens (e.g. BERT's `[CLS]`/`[SEP]`, LLaMA's BOS). Defaults
    ///   to `true`, matching `tokenizers`' default.
    public func encode(_ text: String, addSpecialTokens: Bool = true) throws -> [UInt32] {
        var outLen: UInt64 = 0
        let ptr: UnsafeMutablePointer<UInt32>? = text.withCString { cstr in
            let bytes = UnsafeRawPointer(cstr).assumingMemoryBound(to: UInt8.self)
            return ml_engine_tokenizer_encode(
                handle,
                bytes,
                UInt64(strlen(cstr)),
                addSpecialTokens ? 1 : 0,
                &outLen
            )
        }
        guard let p = ptr else {
            throw TokenizerError.encodeFailed(text: text)
        }
        let ids = Array(UnsafeBufferPointer(start: p, count: Int(outLen)))
        ml_engine_tokenizer_free_u32_buffer(p, outLen)
        return ids
    }

    /// Decode an array of token ids back to a string.
    ///
    /// - `skipSpecialTokens`: if true, the tokenizer's special tokens
    ///   (BOS, EOS, PAD, etc.) are stripped from the output. Usually
    ///   what you want when rendering model output to a user.
    public func decode(_ ids: [UInt32], skipSpecialTokens: Bool = false) throws -> String {
        var outLen: UInt64 = 0
        let ptr: UnsafeMutablePointer<UInt8>? = ids.withUnsafeBufferPointer { buf in
            ml_engine_tokenizer_decode(
                handle,
                buf.baseAddress,
                UInt64(buf.count),
                skipSpecialTokens ? 1 : 0,
                &outLen
            )
        }
        guard let p = ptr else {
            throw TokenizerError.decodeFailed(tokenCount: ids.count)
        }
        // Build a Swift String from the bytes, then free the Rust buffer.
        let bytes = UnsafeBufferPointer(start: p, count: Int(outLen))
        let text = String(decoding: bytes, as: UTF8.self)
        ml_engine_tokenizer_free_u8_buffer(p, outLen)
        return text
    }

    deinit {
        // `handle` is already `OpaquePointer`, matching Swift's import
        // of the forward-declared `Tokenizer*` C type.
        ml_engine_tokenizer_free(handle)
    }
}
