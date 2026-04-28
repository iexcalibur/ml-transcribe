#!/usr/bin/env python3
"""
Side-by-side comparison of NeMo's mel pipeline (the one Cohere
trained on) versus our Swift implementation, replicated in pure
NumPy. Both consume the same audio and we diff at each stage:

    audio
     │
     ├─ pre-emph                  →  diff
     ├─ STFT power                →  diff
     ├─ mel filterbank @ power    →  diff
     ├─ log                       →  diff
     └─ per-feature normalize     →  diff

The first stage where the two diverge is the bug in our Swift mel.
"""

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file


def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"want 16kHz, got {sr}")
    return audio.astype(np.float32)


def preemph_ours(x, alpha=0.97):
    """Our pre-emph: y[0]=x[0], y[i]=x[i]-α*x[i-1]."""
    y = np.zeros_like(x)
    if len(x) > 0:
        y[0] = x[0]
        y[1:] = x[1:] - alpha * x[:-1]
    return y


def hann_symmetric(n):
    """Symmetric Hann (our impl): w[i] = 0.5 - 0.5*cos(2*pi*i/(n-1))."""
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    i = np.arange(n).astype(np.float32)
    return (0.5 - 0.5 * np.cos(2 * np.pi * i / (n - 1))).astype(np.float32)


def stft_power_ours(samples, n_fft=512, hop=160, win_size=400, preemph=0.97):
    """Replicates audio.rs log_mel_spectrogram steps 0-3.
    Returns power spectrogram of shape [n_frames, n_freqs]."""
    # 0. preemph
    x = preemph_ours(samples, preemph) if preemph else samples
    # 1. center=True reflective pad by n_fft/2
    pad = n_fft // 2
    if pad > 0 and len(x) > 0:
        # Mirror our Rust code's specific reflective padding (which
        # may differ slightly from numpy.pad's default).
        head = np.zeros(pad, dtype=np.float32)
        for i in range(pad, 0, -1):
            head[pad - i] = x[min(i, len(x) - 1)]
        tail = np.zeros(pad, dtype=np.float32)
        for i in range(pad):
            idx = max(0, len(x) - 2 - i)
            tail[i] = x[min(idx, len(x) - 1)]
        x = np.concatenate([head, x, tail])
    # 2-3. windowed STFT
    win = hann_symmetric(win_size)
    n_frames = (len(x) - n_fft) // hop + 1
    n_freqs = n_fft // 2 + 1
    win_offset = (n_fft - win_size) // 2
    power = np.zeros((n_frames, n_freqs), dtype=np.float32)
    for f in range(n_frames):
        off = f * hop
        buf = np.zeros(n_fft, dtype=np.float32)
        buf[win_offset:win_offset + win_size] = x[off + win_offset:off + win_offset + win_size] * win
        spec = np.fft.rfft(buf)
        power[f, :] = np.abs(spec) ** 2
    return power


def stft_power_torch(samples, fb, window, n_fft=512, hop=160, win_size=400, preemph=0.97):
    """Replicate NeMo's STFT path using torch operations."""
    x = torch.tensor(samples, dtype=torch.float32)
    # Pre-emph (matches NeMo's)
    if preemph:
        x = torch.cat([x[:1], x[1:] - preemph * x[:-1]])
    # NeMo uses torch.stft with center=True, periodic Hann. But the
    # stored window is closer to symmetric, so use it.
    spec = torch.stft(
        x, n_fft=n_fft, hop_length=hop,
        win_length=win_size, window=window,
        center=True, return_complex=True,
        pad_mode="reflect",
    )
    # spec shape: [freq, time]. Power = |spec|^2.
    power = torch.abs(spec) ** 2
    return power.numpy().T  # → [time, freq]


def main():
    audio_path = "/tmp/ml-inspect/cohere-transcribe/tts_sample.wav"
    model_safetensors = "/tmp/ml-inspect/cohere-transcribe/model.safetensors"

    print(f"Loading audio: {audio_path}")
    audio = load_audio(audio_path)
    print(f"  {len(audio)} samples = {len(audio)/16000:.2f} sec")

    print("\nLoading model's stored mel filterbank + window...")
    state = load_file(model_safetensors)
    fb_stored = state["preprocessor.featurizer.fb"].to(torch.float32).numpy()
    win_stored = state["preprocessor.featurizer.window"].to(torch.float32).numpy()
    print(f"  fb shape: {fb_stored.shape}")
    print(f"  window shape: {win_stored.shape}, sum={win_stored.sum():.4f}")
    # fb may be [1, n_mels, n_freqs]; squeeze.
    if fb_stored.ndim == 3:
        fb_stored = fb_stored.squeeze(0)

    # ---- Stage 1: pre-emphasized audio ----
    pe_ours = preemph_ours(audio, 0.97)
    # NeMo's pre-emph is identical to ours (verified by reading their code).
    # But run torch version for paranoia:
    x_t = torch.tensor(audio, dtype=torch.float32)
    pe_torch = torch.cat([x_t[:1], x_t[1:] - 0.97 * x_t[:-1]]).numpy()
    diff_pe = np.abs(pe_ours - pe_torch).max()
    print(f"\n[Stage 1] pre-emph audio: max|diff| = {diff_pe:.6e}  ({'OK' if diff_pe < 1e-5 else 'DIFFERS'})")
    print(f"  ours[:10]:  {pe_ours[:10]}")
    print(f"  torch[:10]: {pe_torch[:10]}")

    # ---- Stage 2: STFT power ----
    power_ours = stft_power_ours(audio, n_fft=512, hop=160, win_size=400, preemph=0.97)
    power_torch = stft_power_torch(audio, fb_stored,
                                    torch.tensor(win_stored),
                                    n_fft=512, hop=160, win_size=400, preemph=0.97)
    print(f"\n[Stage 2] STFT power: ours shape={power_ours.shape}, torch shape={power_torch.shape}")
    if power_ours.shape == power_torch.shape:
        diff = np.abs(power_ours - power_torch)
        print(f"  max|diff| = {diff.max():.6e}, mean|diff| = {diff.mean():.6e}")
        print(f"  ours[0,:5]:  {power_ours[0,:5]}")
        print(f"  torch[0,:5]: {power_torch[0,:5]}")
        # Cosine sim across all elements
        a = power_ours.flatten(); b = power_torch.flatten()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"  cosine sim: {cos:.6f}")

    # ---- Stage 3: mel filterbank @ power ----
    # mel shape [n_mels, time]. fb_stored is [n_mels, n_freqs]. power [time, n_freqs].
    # mel = fb @ power.T → [n_mels, time]
    mel_ours = (fb_stored @ power_ours.T).astype(np.float32)
    mel_torch = (fb_stored @ power_torch.T).astype(np.float32)
    print(f"\n[Stage 3] mel pre-log: ours shape={mel_ours.shape}, torch shape={mel_torch.shape}")
    diff = np.abs(mel_ours - mel_torch)
    print(f"  max|diff| = {diff.max():.6e}, mean|diff| = {diff.mean():.6e}")
    print(f"  ours[0,:5]:  {mel_ours[0,:5]}")
    print(f"  torch[0,:5]: {mel_torch[0,:5]}")

    # ---- Stage 4: log ----
    # NeMo: log(mel + 2^-24)
    # Whisper / our default: log10(max(mel, 1e-10))
    # For Cohere we use NeMo's. Let's apply both to OUR mel and theirs.
    log_zero_guard = 2.0 ** -24
    log_ours = np.log(mel_ours + log_zero_guard).astype(np.float32)
    log_torch = np.log(mel_torch + log_zero_guard).astype(np.float32)
    print(f"\n[Stage 4] log_mel: ours shape={log_ours.shape}")
    diff = np.abs(log_ours - log_torch)
    print(f"  max|diff| = {diff.max():.6e}")
    print(f"  ours[0,:5]:  {log_ours[0,:5]}")
    print(f"  torch[0,:5]: {log_torch[0,:5]}")

    # ---- Stage 5: per-feature normalize ----
    # NeMo's: mean across time per row, std across time (N-1 div),
    # add DITHER_CONSTANT to std, then (x - mean) / std.
    DITHER = 1e-5
    def perfeat_norm_nemo(x):
        # x shape [n_mels, time]. mean / std along time axis.
        n = x.shape[1]
        m = x.mean(axis=1, keepdims=True)
        s = np.sqrt(((x - m) ** 2).sum(axis=1, keepdims=True) / (n - 1))
        s = np.nan_to_num(s, nan=0.0)
        s = s + DITHER
        return (x - m) / s

    norm_ours = perfeat_norm_nemo(log_ours)
    norm_torch = perfeat_norm_nemo(log_torch)
    print(f"\n[Stage 5] normalized: ours shape={norm_ours.shape}")
    print(f"  ours[0,:5]:  {norm_ours[0,:5]}")
    print(f"  torch[0,:5]: {norm_torch[0,:5]}")
    print(f"  ours[0,:10]: {norm_ours[0,:10]}")
    diff = np.abs(norm_ours - norm_torch)
    print(f"  max|diff| = {diff.max():.6e}")

    # ---- Stage 6: compare against PyTorch processor's actual output ----
    print("\n[Stage 6] compare normalized vs PyTorch processor input_features:")
    from safetensors.torch import load_file as ld
    py_state = ld("/tmp/cohere_pytorch_activations.safetensors")
    py_input_features = py_state["input_features"].to(torch.float32).numpy().squeeze(0)
    print(f"  py input_features shape: {py_input_features.shape}")
    print(f"  py[0,:10]: {py_input_features[0,:10]}")
    diff_ours = np.abs(norm_ours - py_input_features)
    diff_torch = np.abs(norm_torch - py_input_features)
    print(f"  ours-mel    vs PyTorch input_features:  max|diff| = {diff_ours.max():.6e}")
    print(f"  torch-stft  vs PyTorch input_features:  max|diff| = {diff_torch.max():.6e}")


if __name__ == "__main__":
    main()
