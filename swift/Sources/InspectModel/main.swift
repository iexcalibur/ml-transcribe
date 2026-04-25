import Foundation
import Transcribe

/// `swift run inspect-model <path>`
///
/// Prints a summary of the tensors in a safetensors file (or a
/// directory containing a sharded `model.safetensors.index.json`).
/// Useful when picking up a new HuggingFace model to see what names
/// it uses and what shapes the projections have, so a `WeightMap`
/// can be assembled.

func usage() -> Never {
    let me = CommandLine.arguments.first ?? "inspect-model"
    print("""
    usage: \(me) <path>

    <path> can be either:
      - a single .safetensors file, or
      - a directory containing model.safetensors.index.json (sharded)
    """)
    exit(1)
}

guard CommandLine.arguments.count >= 2 else { usage() }
let path = CommandLine.arguments[1]

var isDirectory: ObjCBool = false
guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) else {
    fputs("error: path does not exist: \(path)\n", stderr)
    exit(1)
}

// Look for `--keep-f16` anywhere in the args; if present, F16 source
// tensors stay as F16 in storage (halving resident memory).
let keepF16 = CommandLine.arguments.contains("--keep-f16")

let weights: SafetensorsWeights
do {
    if isDirectory.boolValue {
        weights = try SafetensorsWeights(
            shardedDirectory: URL(fileURLWithPath: path)
        )
    } else {
        weights = try SafetensorsWeights(path: path, keepF16: keepF16)
    }
} catch {
    fputs("error: failed to load: \(error)\n", stderr)
    exit(1)
}
if keepF16 { print("(loaded with --keep-f16: F16 source tensors kept as F16 in store)") }

print("===== \(path) =====")
print("Total tensors: \(weights.count)\n")

let sortedKeys = weights.keys
let preview = 40
print("Tensor list (showing up to \(preview) of \(sortedKeys.count)):")
for key in sortedKeys.prefix(preview) {
    guard let t = weights[key] else { continue }
    let shapeStr = t.shape.map(String.init).joined(separator: ", ")
    // Pad key to a nice width without using %s (which crashes on
    // Swift Strings inside String(format:)).
    let keyCol = key.padding(toLength: 70, withPad: " ", startingAt: 0)
    print("  \(keyCol) [\(shapeStr)]  numel=\(t.numel)")
}
if sortedKeys.count > preview {
    print("  ... and \(sortedKeys.count - preview) more")
}

// ---------------------------------------------------------------------------
// Top-level name grouping: first dotted segment, with counts.
// ---------------------------------------------------------------------------
var groups: [String: Int] = [:]
for key in sortedKeys {
    let head = key.split(separator: ".", maxSplits: 1)
        .first.map(String.init) ?? key
    groups[head, default: 0] += 1
}
print("\nTop-level prefixes:")
for (prefix, count) in groups.sorted(by: { $0.value > $1.value }) {
    print("  \(prefix): \(count)")
}

// ---------------------------------------------------------------------------
// Layer-count heuristic: find the largest `layers.{N}` index we see.
// ---------------------------------------------------------------------------
var maxLayerIdx = -1
for key in sortedKeys {
    // Match patterns like "model.layers.17.self_attn.q_proj.weight" or
    // "layers.3.attn.q_proj.weight".
    let parts = key.split(separator: ".")
    for i in 0..<(parts.count - 1) where parts[i] == "layers" {
        if let n = Int(parts[i + 1]) {
            maxLayerIdx = max(maxLayerIdx, n)
        }
    }
}
if maxLayerIdx >= 0 {
    print("\nInferred layer count: \(maxLayerIdx + 1)")
}

// ---------------------------------------------------------------------------
// Shape fingerprints: the three most common shapes give a sense of
// (D, D), (D, F), (F, D), and 1-D norms.
// ---------------------------------------------------------------------------
var shapeCounts: [[Int]: Int] = [:]
for key in sortedKeys {
    if let t = weights[key] {
        shapeCounts[t.shape, default: 0] += 1
    }
}
print("\nMost common shapes:")
for (shape, count) in shapeCounts.sorted(by: { $0.value > $1.value }).prefix(8) {
    print("  \(shape): \(count) tensors")
}

// ---------------------------------------------------------------------------
// Optional: `--load-llama D F H L` attempts to load as a LLaMA-style
// DecoderLM with the given (modelDim, ffnDim, numHeads, numLayers).
// If the rename/transpose machinery is correct, construction succeeds
// and we can run one forward step.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// `--gen-llama D F H L Hkv tokenizer.json "<prompt>" maxTokens`
//
// Full proof loop: tokenize text → embed → run all N transformer
// layers with KV cache → argmax → decode → print.
//
// On CPU this is ~1–2 sec per token for a 1B-class model, so a 30-
// token demo runs in 30–60 seconds.
// ---------------------------------------------------------------------------
if CommandLine.arguments.count >= 11,
   CommandLine.arguments[2] == "--gen-llama",
   let D = Int(CommandLine.arguments[3]),
   let F = Int(CommandLine.arguments[4]),
   let H = Int(CommandLine.arguments[5]),
   let L = Int(CommandLine.arguments[6]),
   let Hkv = Int(CommandLine.arguments[7]),
   let maxTokens = Int(CommandLine.arguments[10])
{
    let tokenizerPath = CommandLine.arguments[8]
    let prompt        = CommandLine.arguments[9]
    let embedName = "model.embed_tokens.weight"
    let vocab = weights[embedName]?.shape.first ?? 0
    print("\n--- Generating: D=\(D) F=\(F) H=\(H) Hkv=\(Hkv) L=\(L) V=\(vocab) ---")
    print("    prompt: \"\(prompt)\"")
    print("    max new tokens: \(maxTokens)")

    let mapped = WeightMap(
        source: weights,
        renames: WeightMap.huggingFaceLlamaRenames(numLayers: L),
        transpose: WeightMap.huggingFaceLlamaTransposeSet(numLayers: L)
    )
    let config = DecoderLM.Config.llamaStyle(
        vocabSize: vocab, modelDim: D, numHeads: H, ffnDim: F,
        numLayers: L, maxSeqLen: 256, numKVHeads: Hkv
    )

    do {
        let lm = try DecoderLM(weights: mapped, config: config)
        let tokenizer = try Tokenizer(path: tokenizerPath)

        let promptIds = try tokenizer.encode(prompt, addSpecialTokens: true)
        print("    prompt tokens (\(promptIds.count)): \(promptIds)")

        let start = Date()
        let generated = try lm.generate(
            prompt: promptIds.map { Int($0) },
            maxNewTokens: maxTokens
        )
        let elapsed = Date().timeIntervalSince(start)

        print("    generated tokens: \(generated)")
        let decoded = try tokenizer.decode(
            generated.map { UInt32($0) },
            skipSpecialTokens: true
        )
        let full = try tokenizer.decode(
            (promptIds.map { Int($0) } + generated).map { UInt32($0) },
            skipSpecialTokens: true
        )
        print("\n--- result ---")
        print("continuation: \"\(decoded)\"")
        print("\nfull: \"\(full)\"")
        print(String(
            format: "\n(%.1f sec total, %.2f sec/token)",
            elapsed, elapsed / Double(maxTokens)
        ))
    } catch {
        print("✗ Generation failed: \(error)")
    }
    exit(0)
}

if CommandLine.arguments.count >= 7,
   CommandLine.arguments[2] == "--load-llama",
   let D = Int(CommandLine.arguments[3]),
   let F = Int(CommandLine.arguments[4]),
   let H = Int(CommandLine.arguments[5]),
   let L = Int(CommandLine.arguments[6])
{
    // Optional 5th positional: num_key_value_heads (for GQA).
    // If omitted, defaults to H (= plain MHA).
    let Hkv: Int? = CommandLine.arguments.count >= 8
        ? Int(CommandLine.arguments[7])
        : nil
    let embedName = "model.embed_tokens.weight"
    let vocab = weights[embedName]?.shape.first ?? 0
    let kvDescr = Hkv.map { " Hkv=\($0)" } ?? ""
    print("\n--- Attempting LLaMA load: D=\(D) F=\(F) H=\(H)\(kvDescr) L=\(L) V=\(vocab) ---")

    let mapped = WeightMap(
        source: weights,
        renames: WeightMap.huggingFaceLlamaRenames(numLayers: L),
        transpose: WeightMap.huggingFaceLlamaTransposeSet(numLayers: L)
    )
    let config = DecoderLM.Config.llamaStyle(
        vocabSize: vocab, modelDim: D, numHeads: H, ffnDim: F,
        numLayers: L, maxSeqLen: 128, numKVHeads: Hkv
    )
    do {
        let lm = try DecoderLM(weights: mapped, config: config)
        print("✓ DecoderLM constructed.")
        let pred = try lm.step(tokenId: 0)
        print("✓ Forward step: step(0) -> token \(pred) (in 0..<\(vocab))")
    } catch {
        print("✗ Load failed: \(error)")
    }
}
