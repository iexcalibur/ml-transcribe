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
        public let normalize: NormalizeMode

        public init(
            sampleRate: UInt32 = 16_000,
            nFFT: Int = 400,
            hopLength: Int = 160,
            nMels: Int = 80,
            normalize: NormalizeMode = .whisper
        ) {
            self.sampleRate = sampleRate
            self.nFFT = nFFT
            self.hopLength = hopLength
            self.nMels = nMels
            self.normalize = normalize
        }

        /// Whisper-compatible: 80 mels, n_fft=400, Whisper-style
        /// magnitude clamp. Used by `WhisperTiny`.
        public static let whisper = Config()

        /// Cohere Transcribe / NeMo Fast-Conformer preprocessing:
        /// 128 mels, n_fft=512, per-feature normalization.
        public static let cohereTranscribe = Config(
            sampleRate: 16_000,
            nFFT: 512,
            hopLength: 160,
            nMels: 128,
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
                config.normalize.rawValue
            )
        }
        precondition(id != UInt32.max,
            "logMelSpectrogram: empty input or null pointer")
        return Tensor(id: id)
    }
}
