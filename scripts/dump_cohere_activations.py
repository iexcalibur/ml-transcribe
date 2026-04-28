#!/usr/bin/env python3
"""
Dump intermediate activations from the Cohere Transcribe-2B PyTorch
reference model so the Swift port can diff against ground truth
layer-by-layer.

This is the "ground truth" half of the layer-diff workflow:

    PyTorch reference forward → activations dumped to .npz
                                         │
                                         ▼
                                Swift comparator test
                                (loads .npz, runs same audio
                                 through our pipeline, computes
                                 cosine sim + L2 distance per
                                 checkpoint, finds first divergence)

Usage:

    pip install transformers sentencepiece torch numpy soundfile
    python scripts/dump_cohere_activations.py \\
        --model-dir /tmp/ml-inspect/cohere-transcribe \\
        --audio /tmp/ml-inspect/cohere-transcribe/tts_sample.wav \\
        --output /tmp/cohere_pytorch_activations.npz

What gets dumped (~15-20 tensors, ~50 MB total):
    input_features                  — mel spectrogram going into encoder
    encoder.pre_encode              — output of ConvSubsampling
    encoder.layers.{0,11,23,35,47}  — Conformer layer outputs (sampled)
    encoder_final                   — full encoder output [1, T', 1280]
    encoder_decoder_proj            — projected encoder context [1, T', 1024]
    decoder.embedding               — token + pos + LN, [1, prompt_len, 1024]
    decoder.layer0.self_attn        — output of self-attn sub-layer
    decoder.layer0.cross_attn       — output of cross-attn sub-layer
    decoder.layer0.ffn              — output of FFN sub-layer
    decoder.layer0                  — full decoder layer 0 output
    decoder.layer7                  — full decoder layer 7 (last) output
    decoder.final_norm              — post-final-LN hidden, [1, prompt_len, 1024]
    head_logits                     — [1, prompt_len, 16384]

Each activation is a contiguous F32 numpy array, one entry per prompt
token (we feed the full 9-token build_prompt as decoder_input_ids in
teacher-forced mode, so you can compare any decoder step you want).

Memory: model loads as F32 (~8 GiB). Forward pass adds ~1-2 GiB
of activations. Run on a Mac with ≥16 GiB RAM, or pass --bf16.

Speed: ~30 sec to load, ~30-60 sec for forward on M-series CPU.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_audio_16k_mono(path: str) -> np.ndarray:
    """Match Swift's `loadWav16kMono` helper: 16 kHz mono float32 in [-1, 1]."""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"expected 16 kHz, got {sr}")
    return audio.astype(np.float32)


def to_f32_numpy(t):
    """Pull a tensor (or tuple's first tensor) to a F32 numpy array."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().to(torch.float32).numpy()
    if isinstance(t, (tuple, list)) and len(t) > 0 and isinstance(t[0], torch.Tensor):
        return t[0].detach().cpu().to(torch.float32).numpy()
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="/tmp/ml-inspect/cohere-transcribe",
                    help="path to the Cohere safetensors directory")
    ap.add_argument("--audio", default="/tmp/ml-inspect/cohere-transcribe/tts_sample.wav",
                    help="16 kHz mono WAV to feed through the model")
    ap.add_argument("--output", default="/tmp/cohere_pytorch_activations.npz",
                    help="where to write the .npz of all dumped activations")
    ap.add_argument("--bf16", action="store_true",
                    help="load model in bf16 instead of f32 (halves memory)")
    args = ap.parse_args()

    # The model dir's modeling_cohere_asr.py uses relative imports
    # (`from .configuration_cohere_asr import ...`), so we can't just
    # `sys.path.insert + import`. Use HF's dynamic module loader,
    # which sets up a proper package context.
    from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: E402

    cohere_cls = get_class_from_dynamic_module(
        "modeling_cohere_asr.CohereAsrForConditionalGeneration",
        os.path.abspath(args.model_dir),
    )
    # Critical: AutoProcessor / AutoFeatureExtractor don't trigger
    # `CohereAsrFeatureExtractor.from_pretrained`'s override that
    # injects the model's stored mel filterbank + window. We MUST
    # bypass and call the class directly, otherwise the runtime
    # filterbank is whatever librosa.filters.mel returned (zeros if
    # we're using the librosa stub).
    fe_cls = get_class_from_dynamic_module(
        "processing_cohere_asr.CohereAsrFeatureExtractor",
        os.path.abspath(args.model_dir),
    )
    proc_cls = get_class_from_dynamic_module(
        "processing_cohere_asr.CohereAsrProcessor",
        os.path.abspath(args.model_dir),
    )
    tok_cls = get_class_from_dynamic_module(
        "tokenization_cohere_asr.CohereAsrTokenizer",
        os.path.abspath(args.model_dir),
    )

    print(f"=== Cohere PyTorch activation dump ===")
    print(f"  model dir: {args.model_dir}")
    print(f"  audio:     {args.audio}")
    print(f"  output:    {args.output}")
    print(f"  precision: {'bf16' if args.bf16 else 'f32'}")
    print()

    print("Loading processor (via class.from_pretrained, not Auto*) ...")
    feature_extractor = fe_cls.from_pretrained(args.model_dir)
    tokenizer = tok_cls.from_pretrained(args.model_dir)
    processor = proc_cls(feature_extractor=feature_extractor, tokenizer=tokenizer)
    fbm = processor.feature_extractor.filterbank
    fb_max = fbm.fb.abs().max().item()
    print(f"  fbm.fb max value: {fb_max}  ({'OK' if fb_max > 0.001 else 'BUG: filterbank is zeros'})")

    print("Loading model... (~30 sec)")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = cohere_cls.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} params loaded")

    # ---- audio ----
    audio = load_audio_16k_mono(args.audio)
    print(f"\nAudio: {len(audio)} samples = {len(audio) / 16000:.2f} sec")

    # The processor handles mel + per-feature norm.
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(dtype=dtype)
    print(f"input_features shape: {tuple(input_features.shape)}")

    # ---- prompt: same as our Swift test ----
    # <|startofcontext|><|startoftranscript|><|emo:undefined|><|en|><|en|>
    # <|pnc|><|noitn|><|notimestamp|><|nodiarize|>
    prompt = [7, 4, 16, 62, 62, 5, 9, 11, 13]
    decoder_input_ids = torch.tensor([prompt], dtype=torch.long)
    seq_len = decoder_input_ids.shape[1]
    positions = torch.arange(seq_len).unsqueeze(0)
    print(f"prompt: {prompt}")
    print(f"positions: {positions.tolist()[0]}")

    # ---- register hooks ----
    activations: dict[str, np.ndarray] = {}
    hook_handles = []

    def make_hook(name):
        def fn(module, input_, output):
            arr = to_f32_numpy(output)
            if arr is not None:
                activations[name] = arr
        return fn

    def hook(name, module):
        hook_handles.append(module.register_forward_hook(make_hook(name)))

    # Save the raw mel as the first checkpoint.
    activations["input_features"] = input_features.detach().cpu().to(torch.float32).numpy()

    # Encoder
    encoder = model.encoder
    hook("encoder.pre_encode", encoder.pre_encode)
    for i in [0, 11, 23, 35, 47]:
        hook(f"encoder.layers.{i}", encoder.layers[i])
    hook("encoder_final", encoder)  # ENCODER overall output (will dump first elem of tuple)

    # Encoder→decoder projection (might be None on this model — only
    # set if encoder.d_model != decoder.hidden_size).
    if model.encoder_decoder_proj is not None:
        hook("encoder_decoder_proj", model.encoder_decoder_proj)
    else:
        print("WARNING: encoder_decoder_proj is None on this model.")

    # Decoder
    decoder = model.transf_decoder
    hook("decoder.embedding", decoder._embedding)

    layer0 = decoder._decoder.layers[0]
    hook("decoder.layer0.self_attn", layer0.first_sub_layer)
    hook("decoder.layer0.cross_attn", layer0.second_sub_layer)
    hook("decoder.layer0.ffn", layer0.third_sub_layer)
    hook("decoder.layer0", layer0)

    hook("decoder.layer7", decoder._decoder.layers[7])
    hook("decoder.final_norm", decoder._decoder.final_layer_norm)

    # Final logits — the head IS named log_softmax in the model code,
    # but it's just `Linear(1024, 16384)` (the log-softmax is optional
    # via use_log_softmax). Hook it.
    hook("head_logits", model.log_softmax)

    # ---- forward ----
    print("\nRunning forward pass... (~30-60 sec)")
    with torch.no_grad():
        # length defaults to input_features.shape[-1] inside the encoder
        # forward when None is passed.
        outputs = model(
            input_features=input_features,
            input_ids=decoder_input_ids,
            positions=positions,
            length=None,
        )

    # Detach hooks before saving so memory clears.
    for h in hook_handles:
        h.remove()

    # ---- summary ----
    print(f"\nCaptured {len(activations)} activations:")
    for name in activations:
        arr = activations[name]
        l2 = float(np.linalg.norm(arr.reshape(-1)))
        print(f"  {name:32s} shape={tuple(arr.shape)} L2={l2:.3f}")

    # Write as both .npz (Python-friendly) and .safetensors (Swift-
    # readable via our existing SafetensorsWeights loader). Activation
    # names with dots are kept verbatim — safetensors permits them.
    print(f"\nWriting to {args.output}...")
    np.savez(args.output, **activations)
    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"  wrote {size_mb:.1f} MB to {args.output}")

    safetensors_path = str(Path(args.output).with_suffix(".safetensors"))
    try:
        from safetensors.numpy import save_file as sf_save
        # safetensors requires contiguous arrays.
        contig = {k: np.ascontiguousarray(v) for k, v in activations.items()}
        sf_save(contig, safetensors_path)
        size_mb = Path(safetensors_path).stat().st_size / (1024 * 1024)
        print(f"  wrote {size_mb:.1f} MB to {safetensors_path}")
    except ImportError:
        print("  (safetensors python package not installed; skipping .safetensors export)")

    print("\nDone. Now run the Swift comparison test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
