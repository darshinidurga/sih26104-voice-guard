import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODEL_PATH = r"models\voiceguard_v2a.pth"
EVAL_CSV = r"data\metadata\eval_metadata.csv"

SAMPLE_RATE = 16000
NUM_SAMPLES = 48000
MAX_PER_CLASS = 500
BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


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
        x = self.classifier(x)
        return x.squeeze(1)


# Load model
model = VoiceGuardV2A().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded V2-A")


# Load Wav2Vec2
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

wav2vec = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
).to(device)

wav2vec.eval()


# Evaluation subset
df = pd.read_csv(EVAL_CSV)

bonafide = df[df["label"] == "bonafide"].head(MAX_PER_CLASS)
spoof = df[df["label"] == "spoof"].head(MAX_PER_CLASS)

test_df = pd.concat(
    [bonafide, spoof]
).reset_index(drop=True)

print("Evaluation samples:", len(test_df))


# Generate scores once
scores = []
true_labels = []

for start in range(0, len(test_df), BATCH_SIZE):

    batch = test_df.iloc[start:start+BATCH_SIZE]

    audio_batch = []

    for _, row in batch.iterrows():

        audio, sr = sf.read(
            row["audio_path"],
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

    with torch.no_grad():

        features = wav2vec(
            inputs.input_values.to(device)
        ).last_hidden_state

        logits = model(features)

        probabilities = torch.sigmoid(logits)

    scores.extend(
        probabilities.cpu().numpy().tolist()
    )

    print(
        f"Processed {min(start+BATCH_SIZE, len(test_df))}"
        f"/{len(test_df)}"
    )


scores = np.array(scores).reshape(-1)
true_labels = np.array(true_labels)


# Threshold sweep
print("\n==============================")
print("THRESHOLD ANALYSIS")
print("==============================")

print(
    "\nThreshold | Accuracy | Precision | Recall | F1 | FPR"
)

for threshold in np.arange(0.10, 0.96, 0.05):

    predictions = (scores >= threshold).astype(int)

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

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

    false_positives = (
        (predictions == 1) &
        (true_labels == 0)
    ).sum()

    actual_bonafide = (true_labels == 0).sum()

    fpr = false_positives / actual_bonafide

    print(
        f"{threshold:9.2f} | "
        f"{accuracy:8.4f} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f1:6.4f} | "
        f"{fpr:6.4f}"
    )