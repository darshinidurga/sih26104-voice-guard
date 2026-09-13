import sys
import time

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = r"models\voiceguard_v2a.pth"

SAMPLE_RATE = 16000
WINDOW_SECONDS = 3
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS

STEP_SECONDS = 1
STEP_SAMPLES = SAMPLE_RATE * STEP_SECONDS

SPOOF_THRESHOLD = 0.90

# Number of consecutive suspicious windows required
# for sustained suspicion in a continuous call.
SUSTAINED_WINDOWS = 3


# ============================================================
# Input
# ============================================================

if len(sys.argv) < 2:
    print("Usage:")
    print(r"python training\realtime_detector.py <audio_path>")
    sys.exit(1)

AUDIO_PATH = sys.argv[1]


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# V2-A Model
# ============================================================

class AttentionPooling(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        scores = self.attention(x)

        weights = torch.softmax(
            scores,
            dim=1
        )

        return torch.sum(
            weights * x,
            dim=1
        )


class VoiceGuardV2A(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            bidirectional=True,
            batch_first=True
        )

        self.attention = AttentionPooling(256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        x, _ = self.lstm(x)

        x = self.attention(x)

        return self.classifier(x).squeeze(1)


# ============================================================
# Load models
# ============================================================

model = VoiceGuardV2A().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

wav2vec = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
).to(device)

wav2vec.eval()

print("Models loaded.")


# ============================================================
# Load audio
# ============================================================

audio, sr = sf.read(
    AUDIO_PATH,
    dtype="float32"
)


# Convert stereo → mono
if audio.ndim > 1:

    audio = audio.mean(axis=1)


# Resample if necessary
if sr != SAMPLE_RATE:

    print(
        f"Resampling audio: "
        f"{sr} Hz → {SAMPLE_RATE} Hz"
    )

    audio = librosa.resample(
        audio,
        orig_sr=sr,
        target_sr=SAMPLE_RATE
    )

    audio = audio.astype(
        np.float32
    )


audio = np.asarray(
    audio,
    dtype=np.float32
)

duration = len(audio) / SAMPLE_RATE

print(
    f"Audio duration: "
    f"{duration:.2f} seconds"
)

print("\nStarting rolling analysis...\n")


# ============================================================
# Generate rolling windows
# ============================================================

scores = []

timestamps = []


# Always analyze at least one window.
#
# Example:
# 1.5 sec → one padded 3 sec window
# 2.5 sec → one padded 3 sec window
# 3.0 sec → one 3 sec window
# 4.0 sec → windows at 0s and 1s
# 5.8 sec → windows at 0s, 1s, 2s, 2.8s

if len(audio) <= WINDOW_SAMPLES:

    start_positions = [0]

else:

    last_start = len(audio) - WINDOW_SAMPLES

    start_positions = list(
        range(
            0,
            last_start + 1,
            STEP_SAMPLES
        )
    )

    # Make sure the final portion is also analyzed.
    if start_positions[-1] != last_start:

        start_positions.append(
            last_start
        )


# ============================================================
# Rolling inference
# ============================================================

for start in start_positions:

    end = start + WINDOW_SAMPLES

    window = audio[start:end]


    # Pad final/short window
    if len(window) < WINDOW_SAMPLES:

        window = np.pad(
            window,
            (
                0,
                WINDOW_SAMPLES - len(window)
            )
        )


    inputs = processor(
        [window],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )


    if device.type == "cuda":

        torch.cuda.synchronize()


    inference_start = time.perf_counter()


    with torch.no_grad():

        features = wav2vec(
            inputs.input_values.to(device)
        ).last_hidden_state

        logits = model(features)

        spoof_score = torch.sigmoid(
            logits
        ).item()


    if device.type == "cuda":

        torch.cuda.synchronize()


    inference_time = (
        time.perf_counter()
        - inference_start
    )


    scores.append(
        spoof_score
    )

    timestamp = (
        start / SAMPLE_RATE
    )

    timestamps.append(
        timestamp
    )


    status = (
        "SUSPICIOUS"
        if spoof_score >= SPOOF_THRESHOLD
        else "NORMAL"
    )


    print(
        f"{timestamp:6.1f}s | "
        f"Score: {spoof_score:.3f} | "
        f"{status:10s} | "
        f"Inference: "
        f"{inference_time * 1000:.1f} ms"
    )


# ============================================================
# Temporal analysis
# ============================================================

print("\n==============================")
print("TEMPORAL SUMMARY")
print("==============================")


if not scores:

    print("No windows were analyzed.")

    sys.exit(1)


scores_array = np.array(
    scores,
    dtype=np.float32
)


print(
    "Windows:",
    len(scores_array)
)

print(
    "Mean score:",
    round(
        float(scores_array.mean()),
        4
    )
)

print(
    "Maximum:",
    round(
        float(scores_array.max()),
        4
    )
)


suspicious_mask = (
    scores_array >= SPOOF_THRESHOLD
)


suspicious_windows = int(
    suspicious_mask.sum()
)


print(
    "Suspicious windows:",
    f"{suspicious_windows}/"
    f"{len(scores_array)}"
)


# ============================================================
# Consecutive-window detection
# ============================================================

max_consecutive = 0
current_consecutive = 0


for suspicious in suspicious_mask:

    if suspicious:

        current_consecutive += 1

        max_consecutive = max(
            max_consecutive,
            current_consecutive
        )

    else:

        current_consecutive = 0


print(
    "Maximum consecutive suspicious:",
    max_consecutive
)


# ============================================================
# Final decision
# ============================================================

if max_consecutive >= SUSTAINED_WINDOWS:

    print(
        "\n⚠️ Sustained voice suspicion detected."
    )

elif len(scores_array) < SUSTAINED_WINDOWS:

    # For very short clips, don't claim
    # "sustained" because there aren't enough
    # windows to establish persistence.
    if suspicious_windows > 0:

        print(
            "\n⚠️ Suspicious voice detected "
            "(short recording; persistence "
            "cannot be established)."
        )

    else:

        print(
            "\n✓ No voice suspicion detected."
        )

else:

    print(
        "\n✓ No sustained voice suspicion."
    )