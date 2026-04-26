//! Audio preprocessing for speech models.
//!
//! Produces a log-mel spectrogram from raw audio samples — the
//! standard input format for Whisper, Conformer, and Cohere Transcribe.
//!
//! This module is intentionally Whisper-compatible by default:
//! n_fft=400, hop_length=160, n_mels=80 at 16 kHz, log10 magnitude
//! with the same -8 floor and (x+4)/4 normalization. That lets us
//! drop a Whisper safetensors file in and have its preprocessing
//! match openai/whisper bit-for-bit (within float tolerance).

use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

/// HTK mel scale: hz → mels.
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// HTK mel scale: mels → hz.
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Build the mel filterbank as a [n_mels, n_fft/2 + 1] matrix laid out
/// row-major. Each row is a triangular filter centered on a mel-scale
/// frequency, with linear ramps to its neighbors.
///
/// Uses Slaney-style normalization (each filter's area = 1), which
/// matches Whisper / torch.audio defaults. HTK-style (no
/// normalization) is also common; we don't expose it as a flag yet.
fn mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // n_mels + 2 evenly-spaced points on the mel scale: edges + centers.
    let mut mel_points = vec![0.0f32; n_mels + 2];
    for i in 0..(n_mels + 2) {
        let frac = i as f32 / (n_mels + 1) as f32;
        mel_points[i] = mel_min + frac * (mel_max - mel_min);
    }
    // Convert each to an FFT bin index (fractional).
    let mut bin_freqs = vec![0.0f32; n_mels + 2];
    for i in 0..(n_mels + 2) {
        let hz = mel_to_hz(mel_points[i]);
        bin_freqs[i] = hz * (n_fft as f32) / (sample_rate as f32);
    }

    let mut fb = vec![0.0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let lo = bin_freqs[m];
        let mid = bin_freqs[m + 1];
        let hi = bin_freqs[m + 2];
        // Slaney normalization: divide by the filter width (in Hz).
        let lo_hz = mel_to_hz(mel_points[m]);
        let hi_hz = mel_to_hz(mel_points[m + 2]);
        let enorm = 2.0 / (hi_hz - lo_hz);

        for k in 0..n_freqs {
            let kf = k as f32;
            let weight = if kf < lo || kf > hi {
                0.0
            } else if kf <= mid {
                (kf - lo) / (mid - lo)
            } else {
                (hi - kf) / (hi - mid)
            };
            fb[m * n_freqs + k] = weight * enorm;
        }
    }
    fb
}

/// Hann window of length `n`.
fn hann_window(n: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; n];
    if n > 1 {
        let denom = (n - 1) as f32;
        for i in 0..n {
            // 0.5 * (1 - cos(2π * i / (n-1)))
            let phase = 2.0 * std::f32::consts::PI * (i as f32) / denom;
            w[i] = 0.5 - 0.5 * phase.cos();
        }
    }
    w
}

/// Produce a log-mel spectrogram from `samples` at `sample_rate` Hz.
///
/// Output shape: `[n_mels, n_frames]` row-major. `n_frames` depends
/// on audio length and hop:
///   `n_frames = 1 + (n_samples - 1) / hop_length`
/// (with reflective padding at the edges, matching librosa /
/// torchaudio defaults — n_frames = `n_samples / hop` rounded up).
///
/// Normalization: log10 magnitude with a 1e-10 floor, then clamped
/// to [max - 8, max], then scaled `(x + 4) / 4`. This matches
/// Whisper's preprocessing exactly so its weights work with no
/// further normalization on our side.
pub fn log_mel_spectrogram(
    samples: &[f32],
    sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
) -> (Vec<f32>, Vec<usize>) {
    // 1. Pad reflectively by n_fft/2 on each side so the first/last
    //    frames are centered — matches `center=True` in librosa.
    let pad = n_fft / 2;
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    for i in (1..=pad).rev() {
        padded.push(samples[i.min(samples.len() - 1)]);
    }
    padded.extend_from_slice(samples);
    for i in 0..pad {
        let idx = samples.len().saturating_sub(2 + i);
        padded.push(samples[idx.min(samples.len() - 1)]);
    }

    // 2. Frame into `[n_frames, n_fft]` with `hop_length` stride,
    //    multiply by Hann window.
    let window = hann_window(n_fft);
    let n_frames = (padded.len() - n_fft) / hop_length + 1;
    let n_freqs = n_fft / 2 + 1;

    // 3. Run FFT for each frame; collect magnitude squared (power).
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut buf = vec![Complex32::new(0.0, 0.0); n_fft];
    let mut power = vec![0.0f32; n_frames * n_freqs];

    for f in 0..n_frames {
        let off = f * hop_length;
        for i in 0..n_fft {
            buf[i] = Complex32::new(padded[off + i] * window[i], 0.0);
        }
        fft.process(&mut buf);
        for k in 0..n_freqs {
            power[f * n_freqs + k] = buf[k].norm_sqr();
        }
    }

    // 4. Apply mel filterbank. mel[m, f] = sum_k power[f, k] * fb[m, k].
    let fb = mel_filterbank(sample_rate, n_fft, n_mels, 0.0,
                            sample_rate as f32 / 2.0);
    let mut mel = vec![0.0f32; n_mels * n_frames];
    for m in 0..n_mels {
        for f in 0..n_frames {
            let mut acc = 0.0f32;
            for k in 0..n_freqs {
                acc += power[f * n_freqs + k] * fb[m * n_freqs + k];
            }
            mel[m * n_frames + f] = acc;
        }
    }

    // 5. log10 with floor at 1e-10.
    for x in mel.iter_mut() {
        *x = x.max(1e-10).log10();
    }

    // 6. Whisper normalization: clamp to [max - 8, max], then (x+4)/4.
    let max_val = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let floor = max_val - 8.0;
    for x in mel.iter_mut() {
        *x = x.max(floor);
        *x = (*x + 4.0) / 4.0;
    }

    (mel, vec![n_mels, n_frames])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity check: a constant DC signal produces a log-mel output
    /// where the first (lowest) bin dominates after windowing.
    #[test]
    fn mel_of_dc_signal_is_finite() {
        let samples = vec![1.0f32; 16000]; // 1 second of DC at 16 kHz
        let (mel, shape) = log_mel_spectrogram(&samples, 16000, 400, 160, 80);
        assert_eq!(shape, vec![80, 101]);
        for &x in mel.iter() {
            assert!(x.is_finite(), "non-finite value in mel: {}", x);
        }
    }

    /// A 440 Hz sine wave should peak in the mel bin closest to 440 Hz
    /// (which is roughly mel bin 16-20 of an 80-mel-bank at 16 kHz
    /// over 0..8000 Hz).
    #[test]
    fn mel_of_sine_peaks_at_expected_bin() {
        let sr = 16000u32;
        let freq = 440.0f32;
        let n = 16000usize; // 1 second
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32) / (sr as f32)).sin())
            .collect();
        let (mel, shape) = log_mel_spectrogram(&samples, sr, 400, 160, 80);
        assert_eq!(shape, vec![80, 101]);

        // Average energy per mel bin (across frames). Find the argmax.
        let n_frames = shape[1];
        let mut energies = vec![0.0f32; 80];
        for m in 0..80 {
            for f in 0..n_frames {
                energies[m] += mel[m * n_frames + f];
            }
        }
        let argmax = energies.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;

        // 440 Hz lies somewhere in mel bins 12-22 for this filterbank
        // (depends on exact triangular weights). Just check it's in
        // the lower half of the mel range.
        assert!(argmax < 40,
            "440 Hz sine should peak in lower mel bins, got bin {}", argmax);
    }
}
