from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.preprocessing import load_audio


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "facebook/wav2vec2-base-960h"

BASE_DIR = Path(__file__).resolve().parents[1]

SPLIT_FILE = BASE_DIR / "data" / "splits" / "train_split.csv"

OUTPUT_FILE = BASE_DIR / "data" / "train_sequence_embeddings.npz"

BATCH_SIZE = 8


# ============================================================
# SETUP
# ============================================================

print("=" * 70)
print("VOICEGUARD V2 — FULL TRAIN SEQUENCE EXTRACTION")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nDevice: {device}")

if device.type != "cuda":
    raise RuntimeError("CUDA GPU is required for this extraction.")


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(SPLIT_FILE)

print(f"Training files: {len(df):,}")

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Wav2Vec2...")

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

model = Wav2Vec2Model.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()

print("Model loaded.")


# ============================================================
# OUTPUT ARRAYS
# ============================================================

all_embeddings = []
all_labels = []
all_ids = []


# ============================================================
# BATCHED EXTRACTION
# ============================================================

total = len(df)

with torch.no_grad():

    for start in range(0, total, BATCH_SIZE):

        batch_df = df.iloc[start:start + BATCH_SIZE]

        audio_batch = []

        valid_ids = []
        valid_labels = []

        for _, row in batch_df.iterrows():

            try:
                audio = load_audio(row["audio_path"])

                audio_batch.append(audio)
                valid_ids.append(row["utterance_id"])
                valid_labels.append(
                    1 if row["label"] == "spoof" else 0
                )

            except Exception as e:
                print(
                    f"\nWARNING: Failed to load "
                    f"{row['utterance_id']}: {e}"
                )

        if not audio_batch:
            continue

        inputs = processor(
            audio_batch,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(device)

        outputs = model(input_values)

        sequences = outputs.last_hidden_state

        all_embeddings.append(
            sequences.cpu().numpy()
        )

        all_labels.extend(valid_labels)
        all_ids.extend(valid_ids)

        processed = min(start + BATCH_SIZE, total)

        if processed % 100 == 0 or processed == total:
            print(
                f"Processed {processed:,}/{total:,} "
                f"({processed / total * 100:.1f}%)"
            )


# ============================================================
# COMBINE
# ============================================================

print("\nCombining extracted batches...")

embeddings = np.concatenate(all_embeddings, axis=0)
labels = np.array(all_labels, dtype=np.int64)
utterance_ids = np.array(all_ids)


# ============================================================
# VERIFY
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTION RESULT")
print("=" * 70)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Labels shape:     {labels.shape}")
print(f"IDs shape:        {utterance_ids.shape}")

print(f"\nBona fide: {(labels == 0).sum():,}")
print(f"Spoof:     {(labels == 1).sum():,}")

expected_samples = len(df)

if len(embeddings) != expected_samples:
    raise RuntimeError(
        f"Expected {expected_samples} samples, "
        f"but extracted {len(embeddings)}."
    )

if embeddings.shape[1:] != (149, 768):
    raise RuntimeError(
        f"Unexpected embedding shape: {embeddings.shape}"
    )

print("\nAll verification checks passed.")


# ============================================================
# SAVE
# ============================================================

np.savez_compressed(
    OUTPUT_FILE,
    embeddings=embeddings,
    labels=labels,
    utterance_ids=utterance_ids
)

print(f"\nSaved:")
print(OUTPUT_FILE)

print("\nFULL TRAIN EXTRACTION COMPLETE.")