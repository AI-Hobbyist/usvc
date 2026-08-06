"""Recompute epoch time from throughput data using audio-duration model.

Epoch time = total_audio_duration / (B * segment_duration_per_sample) * ms_per_step
This accounts for the fact that each clip yields many training segments.
"""
import json

data = json.load(open("/tmp/usvc_validate/throughput.json"))

# Representative dataset: 155.6 min total audio
TOTAL_AUDIO_MIN = 155.6
TOTAL_AUDIO_SEC = TOTAL_AUDIO_MIN * 60

# Per-backend segment duration (seconds of audio per training sample)
# rift_dit: 256 mel frames @ hop=512, sr=44100 -> 256*512/44100 = 2.97s
# ddsp6: 256 frames @ block_size=512, sr=44100 -> 256*512/44100 = 2.97s
# diffusion: 256 frames @ hop=256(some)/512, ~2.97s (same mel)
# sovits: 20 frames @ hop=512, sr=44100 -> 20*512/44100 = 0.232s
# rvc: 20 frames @ hop=480, sr=48000 -> 20*480/48000 = 0.2s
SEG_DUR = {
    "rift_dit": 256 * 512 / 44100,
    "ddsp6": 256 * 512 / 44100,
    "diffusion": 256 * 512 / 44100,
    "sovits": 20 * 512 / 44100,
    "rvc": 36 * 480 / 48000,
}

print(f"总音频: {TOTAL_AUDIO_MIN:.1f} min ({TOTAL_AUDIO_SEC:.0f}s)")
print()
print(f"{'Backend':<12} {'Config':<18} {'B':>3} {'seg_dur':>8} {'ms/step':>8} "
      f"{'音频s/step':>10} {'steps/ep':>8} {'min/epoch':>9}")
print("-" * 85)

for d in data:
    if d.get("oom"):
        print(f"{d['backend']:<12} {d['config']:<18} {d['batch_size']:>3}  OOM")
        continue
    b = d["backend"]
    B = d["batch_size"]
    sd = SEG_DUR[b]
    ms = d["ms_per_step"]
    audio_per_step = B * sd  # seconds of audio processed per step
    steps_per_epoch = TOTAL_AUDIO_SEC / audio_per_step
    epoch_sec = steps_per_epoch * ms / 1000
    epoch_min = epoch_sec / 60
    print(f"{b:<12} {d['config']:<18} {B:>3} {sd:>7.2f}s {ms:>7.1f} {audio_per_step:>9.1f}s "
          f"{steps_per_epoch:>8.0f} {epoch_min:>8.1f}m")

print()
print("注: epoch 时间 = 总音频 / 每 step 处理音频 × 每 step 耗时")
print("    每个 clip (~6s) 可切出多个 segment, 所以 epoch 内 step 数远大于 clip 数")
