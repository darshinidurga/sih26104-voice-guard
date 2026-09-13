import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"models\voiceguard_v2a.pth"

EVAL_CSV = r"data\metadata\eval_metadata.csv"

# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
NUM_SAMPLES = 48000
MAX_PER_CLASS = 500
BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# ============================================================
# MODEL
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
        weights = torch.softmax(scores, dim=1)

        return torch.sum(weights * x, dim=1)


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
        x = self.classifier(x)

        return x.squeeze(1)


model = VoiceGuardV2A().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded V2-A")
print("Best validation F1:", checkpoint["best_val_f1"])

# ============================================================
# WAV2VEC2
# ============================================================

print("Loading Wav2Vec2...")

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

wav2vec = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
).to(device)

wav2vec.eval()

# ============================================================
# LOAD EVAL DATA
# ============================================================

df = pd.read_csv(EVAL_CSV)

print("\nFull evaluation set:", len(df))

# Take 500 bona fide + 500 spoof
bonafide = df[df["label"] == "bonafide"].head(MAX_PER_CLASS)
spoof = df[df["label"] == "spoof"].head(MAX_PER_CLASS)

test_df = pd.concat([bonafide, spoof]).reset_index(drop=True)

print("Testing:", len(test_df))
print(test_df["label"].value_counts())

# ============================================================
# FEATURE EXTRACTION + PREDICTION
# ============================================================

predictions = []
scores = []
true_labels = []

for start in range(0, len(test_df), BATCH_SIZE):

    batch = test_df.iloc[start:start + BATCH_SIZE]

    audio_batch = []

    for _, row in batch.iterrows():

        audio, sr = sf.read(
            row["audio_path"],
            dtype="float32"
        )

        if sr != SAMPLE_RATE:
            raise ValueError(
                f"Unexpected sample rate {sr}: {row['audio_path']}"
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

        audio_batch.append(audio)

        true_labels.append(
            1 if row["label"] == "spoof" else 0
        )

    inputs = processor(
        audio_batch,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():

        features = wav2vec(
            input_values
        ).last_hidden_state

        logits = model(features)

        probs = torch.sigmoid(logits)

    batch_scores = probs.cpu().numpy()

    scores.extend(batch_scores.tolist())

    predictions.extend(
        (batch_scores >= 0.5).astype(int).tolist()
    )

    print(
        f"Processed {min(start + BATCH_SIZE, len(test_df))}"
        f"/{len(test_df)}"
    )

# ============================================================
# RESULTS
# ============================================================

predictions = np.array(predictions).reshape(-1)
scores = np.array(scores).reshape(-1)
true_labels = np.array(true_labels)

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(
    true_labels,
    predictions,
    zero_division=0
)
recall = recall_score(
    true_labels,
    predictions,
    zero_division=0
)
f1 = f1_score(
    true_labels,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    true_labels,
    predictions
)

print("\n==============================")
print("V2-A UNSEEN EVALUATION")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nSpoof score statistics:")

print(
    "Bonafide mean:",
    round(scores[true_labels == 0].mean(), 4)
)

print(
    "Spoof mean:",
    round(scores[true_labels == 1].mean(), 4)
)