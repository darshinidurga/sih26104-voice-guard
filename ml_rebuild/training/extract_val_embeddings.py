import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Paths
CSV_PATH = r"data\splits\val_split.csv"
OUT_DIR = r"data\memmap_val"

os.makedirs(OUT_DIR, exist_ok=True)

# Settings
SAMPLE_RATE = 16000
NUM_SAMPLES = 48000
BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load validation metadata
df = pd.read_csv(CSV_PATH)

print("Validation samples:", len(df))
print("Labels:")
print(df["label"].value_counts())

# Load Wav2Vec2
print("Loading Wav2Vec2...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.to(device)
model.eval()

# Expected output shape for 3-second audio
NUM_SAMPLES_TOTAL = len(df)
EMBED_DIM = 768
SEQ_LEN = 149

# Create disk-backed arrays
embeddings = np.lib.format.open_memmap(
    os.path.join(OUT_DIR, "embeddings.npy"),
    mode="w+",
    dtype=np.float32,
    shape=(NUM_SAMPLES_TOTAL, SEQ_LEN, EMBED_DIM),
)

labels = np.lib.format.open_memmap(
    os.path.join(OUT_DIR, "labels.npy"),
    mode="w+",
    dtype=np.int64,
    shape=(NUM_SAMPLES_TOTAL,),
)

utterance_ids = np.empty(NUM_SAMPLES_TOTAL, dtype="<U32")

# Process batches
for start in range(0, NUM_SAMPLES_TOTAL, BATCH_SIZE):
    end = min(start + BATCH_SIZE, NUM_SAMPLES_TOTAL)

    audio_batch = []

    for i in range(start, end):
        path = df.iloc[i]["audio_path"]

        audio, sr = sf.read(path, dtype="float32")

        if sr != SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate {sr}: {path}")

        # Convert stereo to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Exactly 3 seconds
        if len(audio) > NUM_SAMPLES:
            audio = audio[:NUM_SAMPLES]
        elif len(audio) < NUM_SAMPLES:
            audio = np.pad(
                audio,
                (0, NUM_SAMPLES - len(audio))
            )

        audio_batch.append(audio)

    inputs = processor(
        audio_batch,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)

    batch_embeddings = outputs.last_hidden_state.cpu().numpy()

    # Safety check
    if batch_embeddings.shape[1:] != (SEQ_LEN, EMBED_DIM):
        raise ValueError(
            f"Unexpected embedding shape: {batch_embeddings.shape}"
        )

    embeddings[start:end] = batch_embeddings

    for j, i in enumerate(range(start, end)):
        labels[i] = 1 if df.iloc[i]["label"] == "spoof" else 0
        utterance_ids[i] = str(df.iloc[i]["utterance_id"])

    if end % 200 == 0 or end == NUM_SAMPLES_TOTAL:
        print(f"Processed {end}/{NUM_SAMPLES_TOTAL}")

# Flush to disk
embeddings.flush()
labels.flush()

np.save(
    os.path.join(OUT_DIR, "utterance_ids.npy"),
    utterance_ids
)

print("\nDONE")
print("Embeddings:", embeddings.shape)
print("Labels:", labels.shape)
print("Output:", OUT_DIR)

print("\nVerification:")
print("Bona fide:", int((labels == 0).sum()))
print("Spoof:", int((labels == 1).sum()))
print("First ID:", utterance_ids[0])