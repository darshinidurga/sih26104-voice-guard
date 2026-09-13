import time
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model

MODEL_PATH = r"models\voiceguard_v2a.pth"
AUDIO_PATH = r"..\test_3sec.wav"

SAMPLE_RATE = 16000
NUM_SAMPLES = 48000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        weights = torch.softmax(scores, dim=1)
        return torch.sum(weights * x, dim=1)


class VoiceGuardV2A(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            768, 128,
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


# Load model
model = VoiceGuardV2A().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load Wav2Vec2
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

wav2vec = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
).to(device)

wav2vec.eval()

# Load audio
audio, sr = sf.read(
    AUDIO_PATH,
    dtype="float32"
)

if audio.ndim > 1:
    audio = audio.mean(axis=1)

if len(audio) > NUM_SAMPLES:
    audio = audio[:NUM_SAMPLES]
elif len(audio) < NUM_SAMPLES:
    audio = np.pad(
        audio,
        (0, NUM_SAMPLES - len(audio))
    )

inputs = processor(
    [audio],
    sampling_rate=SAMPLE_RATE,
    return_tensors="pt",
    padding=True
)

input_values = inputs.input_values.to(device)

# Warm-up
with torch.no_grad():
    for _ in range(5):
        features = wav2vec(input_values)
        model(features.last_hidden_state)

if device.type == "cuda":
    torch.cuda.synchronize()

# Benchmark complete pipeline
times = []
score = None

with torch.no_grad():

    for _ in range(30):

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        features = wav2vec(input_values)
        logits = model(features.last_hidden_state)
        score = torch.sigmoid(logits).item()

        if device.type == "cuda":
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

times = np.array(times)

print("\n==============================")
print("V2 COMPLETE AUDIO BENCHMARK")
print("==============================")

print("Audio:", AUDIO_PATH)
print("Duration: 3 seconds")

print("\nSpoof score:", round(score, 4))

print("\nInference time:")
print("Average:", round(times.mean(), 4), "seconds")
print("Median :", round(np.median(times), 4), "seconds")
print("Fastest:", round(times.min(), 4), "seconds")
print("Slowest:", round(times.max(), 4), "seconds")

print(
    "\nReal-time factor:",
    round(3 / times.mean(), 2),
    "x"
)