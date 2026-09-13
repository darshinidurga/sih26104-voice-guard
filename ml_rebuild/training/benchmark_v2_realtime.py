import time
import torch
import torch.nn as nn
import numpy as np

MODEL_PATH = r"models\voiceguard_v2a.pth"

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
            768,
            128,
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


model = VoiceGuardV2A().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Device:", device)
print("Model loaded.")

# One 3-second Wav2Vec2 sequence
dummy = torch.randn(
    1, 149, 768,
    device=device
)

# Warm-up
with torch.no_grad():
    for _ in range(10):
        model(dummy)

if device.type == "cuda":
    torch.cuda.synchronize()

# Benchmark
times = []

with torch.no_grad():

    for i in range(100):

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        times.append(elapsed)

times = np.array(times)

print("\n==============================")
print("V2 REAL-TIME BENCHMARK")
print("==============================")

print("Average:", round(times.mean(), 4), "seconds")
print("Median :", round(np.median(times), 4), "seconds")
print("Fastest:", round(times.min(), 4), "seconds")
print("Slowest:", round(times.max(), 4), "seconds")

print(
    "\n3-second window processing ratio:",
    round(3 / times.mean(), 2),
    "x real-time"
)