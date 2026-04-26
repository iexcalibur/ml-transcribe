import Foundation
import TranscribeFFIC

/// Audio preprocessing for speech models.
///
/// Produces a log-mel spectrogram from raw `Float` PCM samples. The
/// returned `Tensor` has shape `[n_mels, n_frames]` and is suitable
/// as input to a speech encoder (Whisper, Conformer, Cohere
/// Transcribe, etc).
///
/// Two presets cover the common cases:
///
/// - `Config.whisper` (the default): sample_rate=16k, n_fft=400,
///   hop=160, n_mels=80, normalization clamps to `[max-8, max]` and
///   scales `(x+4)/4`. Bit-compatible with `openai/whisper`.
///
/// - `Config.cohereTranscribe`: sample_rate=16k, n_fft=512, hop=160,
///   n_mels=128, normalization is per-feature zero-mean / unit-var
///   across frames (matches NeMo's `normalize=per_feature` and
///   Cohere Transcribe's preprocessor config).
public enum AudioPreprocessor {

    /// Post-spectrogram normalization variants. Numbers must match
    /// the FFI's `normalize_mode` argument exactly.
    public enum NormalizeMode: UInt32 {
        case whisper    = 0
        case perFeature = 1
        case none       = 2
    }

    /// Configurable preprocessing parameters.
    public struct Config {
        public let sampleRate: UInt32
        public let nFFT: Int
        public let hopLength: Int
        public let nMels: Int
        /// Length of the Hann window, in samples. When `< nFFT`, the
        /// windowed signal is centered inside an `nFFT`-long zero-
        /// padded buffer before FFT — matches PyTorch / torchaudio's
        /// behavior when `win_length < n_fft`.
        ///
        /// `0` defaults to `nFFT` (Whisper convention, where window
        /// length and FFT length match).
        public let nWindowSize: Int
        /// Pre-emphasis coefficient. NeMo / Cohere Transcribe use
        /// `0.97`; Whisper uses none (`0`). Pre-emphasis is a
        /// first-order high-pass filter applied to audio BEFORE
        /// windowing: `y[i] = x[i] - α * x[i-1]`.
        public let preemph: Float
        public let normalize: NormalizeMode

        public init(
            sampleRate: UInt32 = 16_000,
            nFFT: Int = 400,
            hopLength: Int = 160,
            nMels: Int = 80,
            nWindowSize: Int = 0,
            preemph: Float = 0,
            normalize: NormalizeMode = .whisper
        ) {
            self.sampleRate = sampleRate
            self.nFFT = nFFT
            self.hopLength = hopLength
            self.nMels = nMels
            self.nWindowSize = nWindowSize
            self.preemph = preemph
            self.normalize = normalize
        }

        /// Whisper-compatible: 80 mels, n_fft=400, win=400, no
        /// pre-emphasis, Whisper-style magnitude clamp.
        public static let whisper = Config()

        /// Cohere Transcribe / NeMo Fast-Conformer preprocessing:
        /// 128 mels, n_fft=512, win=400, **pre-emphasis 0.97**,
        /// per-feature normalization. Verified against the model's
        /// `preprocessor_config.json`.
        public static let cohereTranscribe = Config(
            sampleRate: 16_000,
            nFFT: 512,
            hopLength: 160,
            nMels: 128,
            nWindowSize: 400,
            preemph: 0.97,
            normalize: .perFeature
        )
    }

    /// Compute a log-mel spectrogram from `samples`.
    ///
    /// Output shape: `[nMels, nFrames]`.
    public static func logMelSpectrogram(
        samples: [Float],
        config: Config = .whisper
    ) -> Tensor {
        let id: UInt32 = samples.withUnsafeBufferPointer { buf in
            ml_engine_log_mel_spectrogram(
                buf.baseAddress,
                UInt64(buf.count),
                config.sampleRate,
                UInt64(config.nFFT),
                UInt64(config.hopLength),
                UInt64(config.nMels),
                UInt64(config.nWindowSize),
                config.preemph,
                config.normalize.rawValue
            )
        }
        precondition(id != UInt32.max,
            "logMelSpectrogram: empty input or null pointer")
        return Tensor(id: id)
    }
}
