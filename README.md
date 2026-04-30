# ml-transcribe

A Swift package + Rust ML framework for running real transformer models
end-to-end on Apple Silicon — no Python, no PyTorch, no third-party ML
library at runtime.

Whisper, TinyLlama, and Cohere Transcribe-2B all run through this stack.
Audio bytes go in, English text comes out.

## What works today

| Model | Size | Use case | Status |
|--|--|--|--|
| **OpenAI Whisper-tiny.en** | 39 M params | Speech-to-text (English) | Bit-perfect transcription |
| **TinyLlama-1.1B-Chat** | 1.1 B params | Text generation | Greedy decoding, GQA |
| **Cohere Transcribe-2B** | 2 B params | Speech-to-text (14 languages) | Pipeline correct, accuracy WIP |

Verified outputs, end-to-end:

```
[whisper-tiny on a TTS sample]
  in:  "The quick brown fox jumps over the lazy dog."
  out: " quick brown fox jumps over the lazy dog."

[tinyllama on a factual prompt]
  in:  "The capital of France is"
  out: "Paris" (within 8 generated tokens)

[cohere-transcribe-2B on a TTS sample]
  in:  "The quick brown fox jumps over the lazy dog."
  out: "the the the the"     ← model collapses to repetition (see below)
```

Whisper and TinyLlama are reproducible end-to-end via the test suite
(see _Quick start_). Cohere Transcribe-2B's full pipeline is in place —
all 2104 weights load, the 48-layer Conformer encoder runs, the
8-layer decoder produces real English with proper capitalization —
but the output suffers from a **repetition collapse**: the encoder's
time-varying features aren't discriminative enough to drive specific
transcription, so the decoder defaults to high-frequency word loops.

This isn't an architectural bug — F16 vs F32 produces bit-identical
output, the stored Cohere mel filterbank produces bit-identical output
to our computed Slaney one, and the rel_shift gather, pre-emphasis,
window-vs-FFT-size, log-zero-guard mode, and Slaney mel scale are all
fixed and verified. The remaining bug needs **layer-by-layer activation
diff against a PyTorch reference run** to localize, which is queued
work but not yet done. See
[`swift/Tests/TranscribeTests/CohereTranscribeIntegrationTests.swift`](swift/Tests/TranscribeTests/CohereTranscribeIntegrationTests.swift)
for the diagnostic test infrastructure already in place.

## Why

Most on-device ML on Apple platforms goes through CoreML, which is fast
(ANE acceleration) but inflexible: model conversion can fail, custom ops
need workarounds, and the converted `.mlmodelc` is opaque. This package
takes the opposite trade-off: every op, every weight loader, and every
numerical decision is in the source tree, written in plain Rust + Swift.

You give up speed (no ANE today; CPU only) for control:

- Models that `coremltools.convert(...)` chokes on just work — write
  the architecture as Swift, load the weights, run.
- Layer-by-layer numerical debugging: dump activations, diff against
  PyTorch, find bugs.
- Same code runs on macOS arm64, macOS Intel, iOS device, iOS Simulator.
- F16 storage with NEON SIMD on aarch64; scalar fallback on x86_64.

Whisper and TinyLlama work in real time. Cohere Transcribe-2B is ~6×
slower than realtime on M-series CPU — fine for batch transcription,
not for live use.

## Quick start

Clone, build the Rust framework as a universal xcframework, run the
test suite:

```bash
git clone https://github.com/iexcalibur/ml-transcribe
cd ml-transcribe
bash scripts/build-swift-ffi.sh        # ~2 min, cross-compiles 5 Apple targets

cd swift
swift test                              # 197 tests, runs in ~15 sec
```

Run an actual transcription with Whisper-tiny:

```bash
# Download Whisper-tiny weights (~150 MB)
hf download openai/whisper-tiny.en --local-dir /tmp/whisper-tiny

# Stage a sample
say -v Samantha -o /tmp/hello.aiff "The quick brown fox jumps over the lazy dog."
afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/hello.aiff /tmp/whisper-tiny/sample.wav

# Run end-to-end transcription
WHISPER_TINY_DIR=/tmp/whisper-tiny swift test \
    --filter WhisperTinyIntegrationTests/testWhisperTranscribeOnRealSpeech
```

### Cohere Transcribe-2B

The Cohere model is gated on Hugging Face — accept the license at
[CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
first, then:

```bash
hf auth login
hf download CohereLabs/cohere-transcribe-03-2026 \
    --local-dir /tmp/cohere-transcribe

COHERE_TRANSCRIBE_DIR=/tmp/cohere-transcribe \
COHERE_RUN_E2E=1 \
    swift test --filter CohereTranscribeIntegrationTests/testCohereTranscribeEndToEnd
```

The model needs ~12 GiB free RAM at peak (BF16→F16 in-memory) and
~4 GiB of disk for weights. Encoder runs in ~28 sec for 5 sec of audio
on M-series.

## Using the package programmatically

```swift
import Transcribe

// 1. Load model weights
let weights = try CohereTranscribeWeightMap.load(
    from: "/path/to/cohere-transcribe/model.safetensors",
    keepF16: true
)
let model = CohereTranscribe(weights: weights)
let tokenizer = try Tokenizer(path: "/path/to/cohere-transcribe/tokenizer.json")

// 2. Read 16 kHz mono PCM samples (any WAV reader works; here using AVFoundation)
let samples: [Float] = ... // your [-1, 1] normalized audio

// 3. Transcribe
let promptTokens = [7, 4, 16, 62, 62, 5, 9, 11, 13]   // Cohere English prompt
let text = try model.transcribe(
    samples: samples,
    tokenizer: tokenizer,
    promptTokens: promptTokens,
    maxNewTokens: 200
)
print(text)   // "Your transcribed audio here."
```

The same pattern works for `WhisperTiny` and `DecoderLM` (the LLM
class). See `swift/Sources/Transcribe/` for full API.

## Architecture

```
┌────────────────────────────────────────┐
│  Swift app                             │
│  ┌──────────────────────────────────┐  │
│  │  Transcribe Swift package        │  │
│  │  WhisperTiny / DecoderLM /       │  │
│  │  CohereTranscribe / Tensor /     │  │
│  │  AudioPreprocessor / Tokenizer   │  │
│  └────────────┬─────────────────────┘  │
└───────────────│────────────────────────┘
                │ FFI
┌───────────────▼────────────────────────┐
│  TranscribeFFI.xcframework (Rust)      │
│  ml_engine_matmul, conv2d, attention,  │
│  cross_attention, kv_cache, embedding, │
│  rope, rel_shift, log_mel, batchnorm,  │
│  glu, softmax, layernorm, rmsnorm, ... │
└────────────────────────────────────────┘
                │ delegates to
┌───────────────▼────────────────────────┐
│  mni_framework_core (Rust crate)       │
│  Tensor store, autograd tape (unused   │
│  at inference), F32/F16 storage, ops   │
└────────────────────────────────────────┘
```

The Rust ML core (`mni_framework_core`) is shared between this Swift
package and the existing TypeScript / N-API binding (see
[`docs/README-typescript-framework.md`](docs/README-typescript-framework.md)
for the Node.js path). The Swift FFI lives in `src/native-swift/` and
exposes a flat C ABI; the Swift `Tensor` class wraps it with a clean
ARC-managed handle.

## Op coverage

The framework implements every primitive a transformer encoder-decoder
needs for inference:

- **Linear algebra**: matmul (with NEON F16 SIMD on aarch64), reshape,
  permute, transpose, contiguous, repeat_interleave
- **Convolution**: 1D + 2D with `stride`, `padding`, `groups`
  (depthwise separable convs are a first-class case)
- **Attention**: causal self-attention, non-causal self-attention,
  cross-attention, KV cache, GQA via `repeat_interleave`,
  Transformer-XL relative-positional self-attention with `rel_shift`
- **Normalization**: LayerNorm, RMSNorm, BatchNorm1d (inference)
- **Activations**: ReLU, GELU, SiLU, sigmoid, GLU, softmax (any dim)
- **Embedding**: token-id lookup
- **Position**: RoPE (half-split LLaMA convention), fixed sinusoidal,
  Transformer-XL relative
- **Audio DSP**: log-mel spectrogram with HTK or Slaney mel scale,
  pre-emphasis, multiple normalization modes, configurable n_fft /
  window / hop, optional pre-loaded filterbank (Cohere-compatible)
- **Tokenizer**: HuggingFace `tokenizer.json` (BPE, SentencePiece,
  byte-fallback)
- **Safetensors**: F32, F16, BF16 (with optional BF16→F16 storage),
  sharded loaders for large models, save with F32 or F16 output

All 197 unit tests pass. End-to-end model verification covered by
opt-in integration tests (gated on the corresponding model directory
env var).

## Limitations and status

| | |
|--|--|
| **Inference only** | No autograd / training support exposed at the Swift API level. The Rust core has an autograd tape but it's CPU-only and not wired through the Swift FFI. |
| **CPU only** | No Metal or ANE backend yet. Whisper runs in real time; Cohere-2B is ~6× slower than realtime. |
| **F16 storage, F32 compute** | Weights live as F16 in memory (saving half the RAM) but matmul up-converts to F32 for accumulation. |
| **No quantization** | INT8 / INT4 support is on the roadmap. Current models ship as F16 (Cohere is 4 GB on disk). |
| **Macros stable Rust 1.94** | Pinned via `rust-toolchain.toml` because that's where `stdarch_neon_f16` stabilized. |
| **Intel Mac**: builds, untested at runtime | The xcframework includes the `x86_64-apple-darwin` slice and the code is portable. End-to-end run on Intel hardware hasn't been verified. |

## Repository layout

```
ml-transcribe/
├── README.md                       # this file
├── LICENSE                         # MIT
├── docs/
│   └── README-typescript-framework.md   # original Node.js path
├── scripts/
│   └── build-swift-ffi.sh          # build the universal xcframework
├── src/
│   ├── native-core/                # shared Rust ML framework (CPU + ops)
│   ├── native-swift/               # Swift FFI + audio DSP + safetensors I/O
│   └── native/                     # legacy N-API binding for Node.js
└── swift/
    ├── Package.swift
    ├── Sources/
    │   ├── Transcribe/             # Swift wrappers, model classes, weight maps
    │   └── InspectModel/           # `swift run inspect-model <path>` CLI
    ├── Tests/TranscribeTests/      # 197 unit + integration tests
    └── artifacts/                  # built xcframework
```

## Building from source

Requires:
- Rust 1.94 (pinned by `rust-toolchain.toml`)
- Xcode 15+ with command-line tools
- macOS arm64 or Intel; iOS Simulator works too

```bash
# All-in-one: build the xcframework, then run Swift tests
bash scripts/build-swift-ffi.sh
cd swift && swift test
```

The build script cross-compiles the Rust crate to 5 targets:
`aarch64-apple-darwin`, `x86_64-apple-darwin`, `aarch64-apple-ios`,
`aarch64-apple-ios-sim`, `x86_64-apple-ios`. The resulting
`TranscribeFFI.xcframework` lipo's the macOS targets into a universal
slice and ships device + simulator slices for iOS.

## Testing

```bash
cd swift

# Default unit tests (no model downloads required)
swift test                          # 197 tests, ~15 sec

# With model integration tests
WHISPER_TINY_DIR=/path/to/whisper-tiny swift test
TINYLLAMA_DIR=/path/to/tinyllama swift test
COHERE_TRANSCRIBE_DIR=/path/to/cohere-transcribe COHERE_RUN_E2E=1 swift test

# iOS Simulator
SIM_ID=$(xcrun simctl list devices available | \
    grep -oE 'iPhone 1[5-9] \([A-F0-9-]+\)' | head -1 | \
    grep -oE '[A-F0-9-]{36}')
xcodebuild test -scheme Transcribe-Package -destination "id=${SIM_ID}"
```

## Roadmap

The framework's op surface is feature-complete for the architectures
it supports; the roadmap is about closing one accuracy gap and then
scaling for production:

1. **Cohere accuracy fix** (in flight) — layer-by-layer activation
   diff against a PyTorch reference run to find where our forward
   diverges. The diagnostic harness in
   `swift/Tests/TranscribeTests/CohereTranscribeIntegrationTests.swift`
   already extracts checkpoint activations, computes per-time-step
   variance, and bisects encoder layers; what's missing is the
   PyTorch dumper script. Estimated fix: 1-2 days once the diff
   scaffolding is in place.
2. **Metal / MPS backend** — bring matmul to GPU; should give 5-10×
   speedup on M-series, making Cohere-2B real-time on macOS.
3. **INT8 / INT4 quantization** — cut Cohere from 4 GB to 1 GB or
   500 MB for shippable iOS apps.
4. **More model families** — Llama-3, Phi-3, Qwen, Mistral all fit
   the same pattern as TinyLlama (write a `WeightMap`, instantiate
   `DecoderLM` with the right config).

## License

MIT. See [`LICENSE`](LICENSE).

## Acknowledgments

- The Rust ML core (`mni_framework_core`) is the same kernel that
  powers the [TypeScript / N-API path](docs/README-typescript-framework.md);
  this Swift package shares its tensor store and op implementations.
- Whisper-tiny.en weights and tokenizer from
  [openai/whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en).
- TinyLlama weights from
  [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).
- Cohere Transcribe-2B weights and reference modeling code from
  [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026).
- HuggingFace `tokenizers` crate for tokenization, `safetensors` crate
  for weight I/O, `rustfft` for STFT, `half` crate for F16 / BF16.
