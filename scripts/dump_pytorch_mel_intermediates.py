#!/usr/bin/env python3
"""
Step through PyTorch's `FilterbankFeatures.forward` manually, saving
each intermediate state. This pins down which internal step
makes mel bin 0's first 10 frames produce the constant -0.16013081
value our reproduction can't reproduce.
"""
import sys, os
sys.path.insert(0, '/tmp/ml-inspect/cohere-transcribe')

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor


def main():
    proc = AutoProcessor.from_pretrained(
        '/tmp/ml-inspect/cohere-transcribe', trust_remote_code=True)
    fbm = proc.feature_extractor.filterbank

    audio, sr = sf.read('/tmp/ml-inspect/cohere-transcribe/tts_sample.wav')
    audio = audio.astype(np.float32)
    print(f"Audio: {len(audio)} samples ({len(audio)/sr:.2f} sec)")
    print(f"audio[:10] = {audio[:10]}")

    # Match processor.__call__:
    # 1. processor calls feature_extractor (CohereAsrFeatureExtractor.__call__)
    # 2. which calls filterbank.forward(audio_tensor, seq_len)

    audio_t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    seq_len = torch.tensor([len(audio)], dtype=torch.long)

    print("\n--- stepping through filterbank.forward ---")

    # Walk the same logic as FilterbankFeatures.forward, capturing
    # state at each line.

    x = audio_t.clone()
    print(f"\n[step 0] input: shape={tuple(x.shape)}")
    print(f"  x[0,:10] = {x[0,:10].numpy()}")

    # pad_min_duration check (no-op for us)
    # stft_pad_amount check (no-op)

    # _apply_dither
    seq_len_time = seq_len
    x_dith = fbm._apply_dither(x, seq_len_time)
    print(f"\n[step 1] post-dither:")
    print(f"  x[0,:10] = {x_dith[0,:10].numpy()}")
    print(f"  diff vs input: {(x_dith - x).abs().max().item():.6e}")

    # preemph
    if fbm.preemph is not None:
        timemask = torch.arange(x_dith.shape[1]).unsqueeze(0) < seq_len_time.unsqueeze(1)
        x_pe = torch.cat((x_dith[:, 0].unsqueeze(1),
                          x_dith[:, 1:] - fbm.preemph * x_dith[:, :-1]), dim=1)
        x_pe = x_pe.masked_fill(~timemask, 0.0)
    print(f"\n[step 2] post-preemph:")
    print(f"  x[0,:10] = {x_pe[0,:10].numpy()}")

    # STFT
    x_stft = fbm.stft(x_pe)  # complex
    print(f"\n[step 3] post-STFT (complex spec):")
    print(f"  shape = {tuple(x_stft.shape)}")
    print(f"  x[0,1,:5] (first 5 frames of FFT bin 1):")
    if x_stft.dim() == 4:  # [B, freq, time, complex]
        print(f"    real: {x_stft[0,1,:5,0].numpy()}")
        print(f"    imag: {x_stft[0,1,:5,1].numpy()}")
    else:
        print(f"    {x_stft[0,1,:5].numpy()}")

    # |spec|
    guard = 0 if not fbm.use_grads else 1e-5
    if x_stft.dim() == 4:
        x_mag = torch.sqrt(x_stft.pow(2).sum(-1) + guard)
    else:
        x_mag = torch.sqrt((x_stft.real ** 2 + x_stft.imag ** 2) + guard)
    print(f"\n[step 4] |spec| (magnitude):")
    print(f"  shape = {tuple(x_mag.shape)}")
    print(f"  x[0,1,:5] (FFT bin 1, first 5 frames): {x_mag[0,1,:5].numpy()}")

    # power
    if fbm.mag_power != 1.0:
        x_pow = x_mag.pow(fbm.mag_power)
    else:
        x_pow = x_mag
    print(f"\n[step 5] power = |spec|^{fbm.mag_power}:")
    print(f"  x[0,1,:5] (FFT bin 1, first 5 frames): {x_pow[0,1,:5].numpy()}")
    print(f"  x[0,0,:5] (FFT bin 0=DC, first 5 frames): {x_pow[0,0,:5].numpy()}")

    # mel = fb @ power
    with torch.amp.autocast('cpu', enabled=False):
        x_mel = torch.matmul(fbm.fb.to(x_pow.dtype), x_pow)
    print(f"\n[step 6] mel pre-log:")
    print(f"  shape = {tuple(x_mel.shape)}")
    print(f"  x[0,0,:10] (mel bin 0, first 10 frames): {x_mel[0,0,:10].numpy()}")
    print(f"  x[0,5,:10] (mel bin 5, first 10 frames): {x_mel[0,5,:10].numpy()}")

    # log
    if fbm.log:
        if fbm.log_zero_guard_type == "add":
            zg = fbm.log_zero_guard_value_fn(x_mel)
            x_log = torch.log(x_mel + zg)
            print(f"\n[step 7] log_zero_guard=add, guard_value={zg}")
        else:
            zg = fbm.log_zero_guard_value_fn(x_mel)
            x_log = torch.log(torch.clamp(x_mel, min=zg))
            print(f"\n[step 7] log_zero_guard=clamp, guard_value={zg}")
        print(f"  log(x)[0,0,:10] (bin 0): {x_log[0,0,:10].numpy()}")
        print(f"  log(x)[0,5,:10] (bin 5): {x_log[0,5,:10].numpy()}")

    # normalize_batch
    seq_len_unfixed = fbm.get_seq_len(seq_len)
    seq_len_norm = torch.where(seq_len == 0,
                               torch.zeros_like(seq_len_unfixed),
                               seq_len_unfixed)
    x_norm, mean_, std_ = fbm.normalize_batch(x_log, seq_len_norm,
                                               normalize_type=fbm.normalize)
    print(f"\n[step 8] post-normalize:")
    print(f"  seq_len_norm = {seq_len_norm}")
    print(f"  mean per row[:5] = {mean_[0,:5].numpy()}")
    print(f"  std  per row[:5] = {std_[0,:5].numpy()}")
    print(f"  x[0,0,:10] (bin 0): {x_norm[0,0,:10].numpy()}")
    print(f"  x[0,5,:10] (bin 5): {x_norm[0,5,:10].numpy()}")


if __name__ == "__main__":
    main()
