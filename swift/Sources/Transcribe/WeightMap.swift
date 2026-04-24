import Foundation

/// A `WeightSource` that wraps another `WeightSource` and renames or
/// transposes tensors on the way through.
///
/// # Use case
///
/// HuggingFace safetensors files ship with names like
/// `model.layers.0.self_attn.q_proj.weight`, and PyTorch `Linear`
/// weights are stored as `[out, in]`. Our `DecoderLM` expects
/// `layers.0.attn.q_proj.weight` in `[in, out]` layout.
///
/// `WeightMap` bridges the two without modifying the underlying file:
///
///   let raw = try SafetensorsWeights(shardedDirectory: hfModelDir)
///   let weights = WeightMap(
///       source: raw,
///       renames: [
///           "embedding":         "model.embed_tokens.weight",
///           "final_norm.weight": "model.norm.weight",
///           "lm_head":           "lm_head.weight",
///           // ...per-layer rules built with a for loop...
///       ],
///       transpose: [
///           "layers.0.attn.q_proj.weight",
///           "layers.0.attn.k_proj.weight",
///           // ...all 2-D Linear weights...
///       ]
///   )
///   let lm = try DecoderLM(weights: weights, config: .llamaStyle(...))
///
/// # Semantics
///
/// - `renames[ourName] = sourceName`: looking up `ourName` returns the
///   tensor stored at `sourceName` in the source. Unrenamed names
///   pass through unchanged.
/// - `transpose` lists the our-names whose tensors should be
///   transposed (swap last two dims) on load. The transposed tensors
///   are materialized eagerly at init time and cached, so subscript
///   access stays O(1) and side-effect-free.
/// - `keys` reports consumer-facing names: rename targets plus any
///   source keys not referenced by a rename.
public final class WeightMap: WeightSource {
    private let source: WeightSource
    private let renames: [String: String]
    private let transposeKeys: Set<String>

    /// Source names that appear on the right-hand side of a rename —
    /// i.e. names that have been "renamed away." Looking them up by
    /// their original source name must return nil; they're only
    /// reachable via their new target name. Pre-computed for O(1)
    /// lookup in the subscript.
    private let renamedSourceNames: Set<String>

    /// Tensors produced by the eager transpose phase. Each entry is an
    /// owned tensor (`.transpose(0, 1).contiguous()` makes a real copy
    /// in the store, independent of the source view), so they survive
    /// as long as this WeightMap does.
    private let transposedCache: [String: Tensor]

    public init(
        source: WeightSource,
        renames: [String: String] = [:],
        transpose: Set<String> = []
    ) {
        self.source = source
        self.renames = renames
        self.transposeKeys = transpose
        self.renamedSourceNames = Set(renames.values)

        // Eager transpose: for each target name in `transpose`, find
        // the underlying source tensor (via renames if needed) and
        // produce a transposed copy. Skip silently if the source key
        // doesn't exist — subscript will return nil and the caller's
        // error path kicks in.
        var cache: [String: Tensor] = [:]
        for targetName in transpose {
            let sourceName = renames[targetName] ?? targetName
            guard let raw = source[sourceName] else { continue }
            cache[targetName] = raw.transpose(0, 1).contiguous()
        }
        self.transposedCache = cache
    }

    public subscript(name: String) -> Tensor? {
        // Transposed tensors live in our own cache.
        if let cached = transposedCache[name] {
            return cached
        }
        // A source name that's been renamed away is NOT accessible by
        // its original name — only by its new target name. This keeps
        // the name-space consistent with what `keys` reports.
        if renamedSourceNames.contains(name) {
            return nil
        }
        let sourceName = renames[name] ?? name
        return source[sourceName]
    }

    /// The union of:
    ///   - all rename targets (our-names exposed by the map), and
    ///   - source keys that aren't being renamed away.
    /// Sorted for determinism.
    public var keys: [String] {
        let passThrough = source.keys.filter { !renamedSourceNames.contains($0) }
        return (Array(renames.keys) + passThrough).sorted()
    }

    /// Same logic as `keys.count`, but cheaper to compute.
    public var count: Int {
        let passThroughCount = source.keys.reduce(0) {
            renamedSourceNames.contains($1) ? $0 : $0 + 1
        }
        return renames.count + passThroughCount
    }
}

// ---------------------------------------------------------------------------
// Convenience: rename map builders for common HF conventions.
// ---------------------------------------------------------------------------

public extension WeightMap {
    /// Build a rename dictionary mapping our LLaMA-convention names
    /// onto HuggingFace's LLaMA convention. Useful starting point;
    /// callers can merge/override entries for other model families.
    ///
    /// HF LLaMA layout (what's on disk):
    ///
    ///   model.embed_tokens.weight
    ///   model.norm.weight
    ///   lm_head.weight
    ///   model.layers.{i}.input_layernorm.weight
    ///   model.layers.{i}.post_attention_layernorm.weight
    ///   model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    ///   model.layers.{i}.mlp.{gate,up,down}_proj.weight
    ///
    /// Ours:
    ///
    ///   embedding
    ///   final_norm.weight
    ///   lm_head
    ///   layers.{i}.norm1.weight
    ///   layers.{i}.norm2.weight
    ///   layers.{i}.attn.{q,k,v,o}_proj.weight
    ///   layers.{i}.ffn.{gate,up,down}_proj.weight
    static func huggingFaceLlamaRenames(
        numLayers: Int
    ) -> [String: String] {
        var r: [String: String] = [
            "embedding":         "model.embed_tokens.weight",
            "final_norm.weight": "model.norm.weight",
            "lm_head":           "lm_head.weight",
        ]
        for i in 0..<numLayers {
            let ours = "layers.\(i)."
            let hf   = "model.layers.\(i)."
            r[ours + "norm1.weight"] = hf + "input_layernorm.weight"
            r[ours + "norm2.weight"] = hf + "post_attention_layernorm.weight"
            for proj in ["q", "k", "v", "o"] {
                r[ours + "attn.\(proj)_proj.weight"]
                    = hf + "self_attn.\(proj)_proj.weight"
            }
            for proj in ["gate", "up", "down"] {
                r[ours + "ffn.\(proj)_proj.weight"]
                    = hf + "mlp.\(proj)_proj.weight"
            }
        }
        return r
    }

    /// All 2-D projection weights used by a LLaMA-style DecoderLM —
    /// the ones that need transposing because PyTorch stores `Linear`
    /// as `[out, in]` and we use `[in, out]`.
    static func huggingFaceLlamaTransposeSet(
        numLayers: Int
    ) -> Set<String> {
        var s: Set<String> = ["lm_head"]
        for i in 0..<numLayers {
            let base = "layers.\(i)."
            for proj in ["q", "k", "v", "o"] {
                s.insert(base + "attn.\(proj)_proj.weight")
            }
            for proj in ["gate", "up", "down"] {
                s.insert(base + "ffn.\(proj)_proj.weight")
            }
        }
        return s
    }
}
