import Foundation
import TranscribeFFIC

/// Audio preprocessing for speech models.
///
/// Produces a log-mel spectrogram from raw `Float` PCM samples. The
/// returned `Tensor` has shape `[n_mels, n_frames]` and is suitable
/// as input to a speech encoder (Whisper, Conformer, Cohere
/// Transcribe, etc).
///
/// Defaults match Whisper exactly:
///   - `sampleRate = 16000`
///   - `nFFT = 400` (25 ms window)
///   - `hopLength = 160` (10 ms hop)
///   - `nMels = 80`
///
/// The output is normalized the same way Whisper expects:
/// `log10`-magnitude floored at 1e-10, clamped to `[max - 8, max]`,
/// then scaled `(x + 4) / 4`. So a Whisper safetensors file works
/// with no further normalization on our side.
public enum AudioPreprocessor {

    /// Default Whisper preprocessing config.
    public struct Config {
        public let sampleRate: UInt32
        public let nFFT: Int
        public let hopLength: Int
        public let nMels: Int

        public init(
            sampleRate: UInt32 = 16_000,
            nFFT: Int = 400,
            hopLength: Int = 160,
            nMels: Int = 80
        ) {
            self.sampleRate = sampleRate
            self.nFFT = nFFT
            self.hopLength = hopLength
            self.nMels = nMels
        }

        public static let whisper = Config()
    }

    /// Compute a log-mel spectrogram from `samples`.
    ///
    /// Output shape: `[nMels, nFrames]` where
    /// `nFrames = 1 + samples.count / hopLength` (approximately,
    /// depending on padding).
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
                UInt64(config.nMels)
            )
        }
        precondition(id != UInt32.max,
            "logMelSpectrogram: empty input or null pointer")
        return Tensor(id: id)
    }
}
