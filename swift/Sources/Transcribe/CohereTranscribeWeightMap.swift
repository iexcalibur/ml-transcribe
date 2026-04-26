import Foundation

/// Map a HuggingFace `CohereLabs/cohere-transcribe-03-2026`
/// safetensors checkpoint onto a `CohereTranscribe.Weights` bundle.
///
/// The Cohere checkpoint is exported by NeMo's PyTorch state_dict
/// serializer, so every Linear weight uses the PyTorch `[out, in]`
/// layout (`weight @ X.T`-style). Our matmul uses `X @ weight` with
/// `weight` shaped `[in, out]`, so every Linear weight needs a
/// transpose on load. Conv weights are already in PyTorch
/// `[C_out, C_in/groups, K]` / `[C_out, C_in/groups, kH, kW]` layout
/// which matches our convention exactly — those pass through.
///
/// 1-D parameters (LayerNorm gamma/beta, BatchNorm gamma/beta + running
/// stats, biases, pos_bias_u/v) also pass through.
public enum CohereTranscribeWeightMap {

    /// Load a full Cohere Transcribe model's `Weights` bundle from a
    /// HuggingFace safetensors file.
    ///
    /// `keepF16: true` saves ~2 GiB of resident memory by keeping
    /// F16-source tensors as F16 in the engine store. Cohere's
    /// safetensors file is BF16, which always up-converts to F32, so
    /// the `keepF16` flag has no effect on the actual on-device
    /// memory footprint for THIS model — but the option is here for
    /// consistency with `WhisperTiny` / `TinyLlama` loaders.
    public static func load(
        from path: String,
        config: CohereTranscribe.Config = .cohereTranscribe2B,
        keepF16: Bool = true
    ) throws -> CohereTranscribe.Weights {
        let raw = try SafetensorsWeights(path: path, keepF16: keepF16)
        let mapped = WeightMap(
            source: raw,
            transpose: cohereTransposeSet(
                encoderLayers: config.encoder.nLayers,
                decoderLayers: config.decoderLayers
            )
        )
        return try buildBundle(from: mapped, config: config)
    }

    /// Construct the `Weights` bundle from a `WeightSource` whose
    /// names match Cohere's PyTorch state_dict layout. Caller is
    /// responsible for any rename / transpose machinery; the typical
    /// path runs through `load(from:)` which supplies the right
    /// `WeightMap` automatically.
    public static func buildBundle(
        from src: WeightSource,
        config: CohereTranscribe.Config
    ) throws -> CohereTranscribe.Weights {
        // ----- ConvSubsampling -----
        let subWeights = try ConvSubsampling.Weights(
            conv1Weight: try fetch(src, "encoder.pre_encode.conv.0.weight"),
            conv1Bias:   try fetch(src, "encoder.pre_encode.conv.0.bias"),
            conv2Weight: try fetch(src, "encoder.pre_encode.conv.2.weight"),
            conv2Bias:   try fetch(src, "encoder.pre_encode.conv.2.bias"),
            conv3Weight: try fetch(src, "encoder.pre_encode.conv.3.weight"),
            conv3Bias:   try fetch(src, "encoder.pre_encode.conv.3.bias"),
            conv4Weight: try fetch(src, "encoder.pre_encode.conv.5.weight"),
            conv4Bias:   try fetch(src, "encoder.pre_encode.conv.5.bias"),
            conv5Weight: try fetch(src, "encoder.pre_encode.conv.6.weight"),
            conv5Bias:   try fetch(src, "encoder.pre_encode.conv.6.bias"),
            outProjWeight: try fetch(src, "encoder.pre_encode.out.weight"),
            outProjBias:   try fetch(src, "encoder.pre_encode.out.bias")
        )

        // ----- Conformer encoder layers -----
        var encLayerWeights: [ConformerLayer.Weights] = []
        encLayerWeights.reserveCapacity(config.encoder.nLayers)
        for i in 0..<config.encoder.nLayers {
            let p = "encoder.layers.\(i)."
            encLayerWeights.append(ConformerLayer.Weights(
                ffn1NormGamma: try fetch(src, p + "norm_feed_forward1.weight"),
                ffn1NormBeta:  try fetch(src, p + "norm_feed_forward1.bias"),
                ffn1UpW:   try fetch(src, p + "feed_forward1.linear1.weight"),
                ffn1UpB:   try fetch(src, p + "feed_forward1.linear1.bias"),
                ffn1DownW: try fetch(src, p + "feed_forward1.linear2.weight"),
                ffn1DownB: try fetch(src, p + "feed_forward1.linear2.bias"),
                attnNormGamma: try fetch(src, p + "norm_self_att.weight"),
                attnNormBeta:  try fetch(src, p + "norm_self_att.bias"),
                attnWQ: try fetch(src, p + "self_attn.linear_q.weight"),
                attnBQ: try fetch(src, p + "self_attn.linear_q.bias"),
                attnWK: try fetch(src, p + "self_attn.linear_k.weight"),
                attnBK: try fetch(src, p + "self_attn.linear_k.bias"),
                attnWV: try fetch(src, p + "self_attn.linear_v.weight"),
                attnBV: try fetch(src, p + "self_attn.linear_v.bias"),
                attnWPos: try fetch(src, p + "self_attn.linear_pos.weight"),
                attnWOut: try fetch(src, p + "self_attn.linear_out.weight"),
                attnBOut: try fetch(src, p + "self_attn.linear_out.bias"),
                attnPosBiasU: try fetch(src, p + "self_attn.pos_bias_u"),
                attnPosBiasV: try fetch(src, p + "self_attn.pos_bias_v"),
                convNormGamma: try fetch(src, p + "norm_conv.weight"),
                convNormBeta:  try fetch(src, p + "norm_conv.bias"),
                convPointwise1Weight: try fetch(src, p + "conv.pointwise_conv1.weight"),
                convPointwise1Bias:   try fetch(src, p + "conv.pointwise_conv1.bias"),
                convDepthwiseWeight:  try fetch(src, p + "conv.depthwise_conv.weight"),
                convDepthwiseBias:    try fetch(src, p + "conv.depthwise_conv.bias"),
                convBNRunningMean: try fetch(src, p + "conv.batch_norm.running_mean"),
                convBNRunningVar:  try fetch(src, p + "conv.batch_norm.running_var"),
                convBNGamma: try fetch(src, p + "conv.batch_norm.weight"),
                convBNBeta:  try fetch(src, p + "conv.batch_norm.bias"),
                convPointwise2Weight: try fetch(src, p + "conv.pointwise_conv2.weight"),
                convPointwise2Bias:   try fetch(src, p + "conv.pointwise_conv2.bias"),
                ffn2NormGamma: try fetch(src, p + "norm_feed_forward2.weight"),
                ffn2NormBeta:  try fetch(src, p + "norm_feed_forward2.bias"),
                ffn2UpW:   try fetch(src, p + "feed_forward2.linear1.weight"),
                ffn2UpB:   try fetch(src, p + "feed_forward2.linear1.bias"),
                ffn2DownW: try fetch(src, p + "feed_forward2.linear2.weight"),
                ffn2DownB: try fetch(src, p + "feed_forward2.linear2.bias"),
                normOutGamma: try fetch(src, p + "norm_out.weight"),
                normOutBeta:  try fetch(src, p + "norm_out.bias")
            ))
        }

        // ----- Decoder layers -----
        var decLayerWeights: [CohereTranscribe.DecoderLayer.Weights] = []
        decLayerWeights.reserveCapacity(config.decoderLayers)
        for i in 0..<config.decoderLayers {
            let p = "transf_decoder._decoder.layers.\(i)."
            decLayerWeights.append(CohereTranscribe.DecoderLayer.Weights(
                saNormGamma: try fetch(src, p + "layer_norm_1.weight"),
                saNormBeta:  try fetch(src, p + "layer_norm_1.bias"),
                saWQ: try fetch(src, p + "first_sub_layer.query_net.weight"),
                saBQ: try fetch(src, p + "first_sub_layer.query_net.bias"),
                saWK: try fetch(src, p + "first_sub_layer.key_net.weight"),
                saBK: try fetch(src, p + "first_sub_layer.key_net.bias"),
                saWV: try fetch(src, p + "first_sub_layer.value_net.weight"),
                saBV: try fetch(src, p + "first_sub_layer.value_net.bias"),
                saWO: try fetch(src, p + "first_sub_layer.out_projection.weight"),
                saBO: try fetch(src, p + "first_sub_layer.out_projection.bias"),
                caNormGamma: try fetch(src, p + "layer_norm_2.weight"),
                caNormBeta:  try fetch(src, p + "layer_norm_2.bias"),
                caWQ: try fetch(src, p + "second_sub_layer.query_net.weight"),
                caBQ: try fetch(src, p + "second_sub_layer.query_net.bias"),
                caWK: try fetch(src, p + "second_sub_layer.key_net.weight"),
                caBK: try fetch(src, p + "second_sub_layer.key_net.bias"),
                caWV: try fetch(src, p + "second_sub_layer.value_net.weight"),
                caBV: try fetch(src, p + "second_sub_layer.value_net.bias"),
                caWO: try fetch(src, p + "second_sub_layer.out_projection.weight"),
                caBO: try fetch(src, p + "second_sub_layer.out_projection.bias"),
                ffnNormGamma: try fetch(src, p + "layer_norm_3.weight"),
                ffnNormBeta:  try fetch(src, p + "layer_norm_3.bias"),
                ffnUpW:   try fetch(src, p + "third_sub_layer.dense_in.weight"),
                ffnUpB:   try fetch(src, p + "third_sub_layer.dense_in.bias"),
                ffnDownW: try fetch(src, p + "third_sub_layer.dense_out.weight"),
                ffnDownB: try fetch(src, p + "third_sub_layer.dense_out.bias")
            ))
        }

        // ----- Top-level pieces -----
        let bundle = CohereTranscribe.Weights(
            encoderSubsampling: subWeights,
            encoderLayers: encLayerWeights,
            encDecProjW: try fetch(src, "encoder_decoder_proj.weight"),
            encDecProjB: try fetch(src, "encoder_decoder_proj.bias"),
            embedTokens: try fetch(src, "transf_decoder._embedding.token_embedding.weight"),
            embedNormGamma: try fetch(src, "transf_decoder._embedding.layer_norm.weight"),
            embedNormBeta:  try fetch(src, "transf_decoder._embedding.layer_norm.bias"),
            decoderLayers: decLayerWeights,
            decoderFinalNormGamma: try fetch(src, "transf_decoder._decoder.final_layer_norm.weight"),
            decoderFinalNormBeta:  try fetch(src, "transf_decoder._decoder.final_layer_norm.bias"),
            headWeight: try fetch(src, "log_softmax.mlp.layer0.weight"),
            headBias:   try fetch(src, "log_softmax.mlp.layer0.bias")
        )
        return bundle
    }

    /// Set of state-dict names whose tensors need a `[out, in] →
    /// [in, out]` transpose on load. Every PyTorch Linear weight ends
    /// up here; conv weights and 1-D biases / norms / running stats
    /// pass through unchanged.
    public static func cohereTransposeSet(
        encoderLayers: Int,
        decoderLayers: Int
    ) -> Set<String> {
        var s: Set<String> = []

        // ConvSubsampling final Linear.
        s.insert("encoder.pre_encode.out.weight")

        // Encoder layers: every Linear.
        for i in 0..<encoderLayers {
            let p = "encoder.layers.\(i)."
            s.insert(p + "feed_forward1.linear1.weight")
            s.insert(p + "feed_forward1.linear2.weight")
            s.insert(p + "feed_forward2.linear1.weight")
            s.insert(p + "feed_forward2.linear2.weight")
            s.insert(p + "self_attn.linear_q.weight")
            s.insert(p + "self_attn.linear_k.weight")
            s.insert(p + "self_attn.linear_v.weight")
            s.insert(p + "self_attn.linear_pos.weight")
            s.insert(p + "self_attn.linear_out.weight")
        }

        // Decoder layers: every Linear.
        for i in 0..<decoderLayers {
            let p = "transf_decoder._decoder.layers.\(i)."
            s.insert(p + "first_sub_layer.query_net.weight")
            s.insert(p + "first_sub_layer.key_net.weight")
            s.insert(p + "first_sub_layer.value_net.weight")
            s.insert(p + "first_sub_layer.out_projection.weight")
            s.insert(p + "second_sub_layer.query_net.weight")
            s.insert(p + "second_sub_layer.key_net.weight")
            s.insert(p + "second_sub_layer.value_net.weight")
            s.insert(p + "second_sub_layer.out_projection.weight")
            s.insert(p + "third_sub_layer.dense_in.weight")
            s.insert(p + "third_sub_layer.dense_out.weight")
        }

        // Encoder→decoder projection + classifier head.
        s.insert("encoder_decoder_proj.weight")
        s.insert("log_softmax.mlp.layer0.weight")

        return s
    }

    /// Lookup helper that produces a more useful error than the raw
    /// `nil` from `WeightSource[name]`.
    public enum LoadError: Error, CustomStringConvertible {
        case missing(String)
        public var description: String {
            switch self {
            case .missing(let n): return "Cohere weight '\(n)' not in safetensors"
            }
        }
    }

    private static func fetch(_ src: WeightSource, _ name: String) throws -> Tensor {
        guard let t = src[name] else { throw LoadError.missing(name) }
        return t
    }
}
