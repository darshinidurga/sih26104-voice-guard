from pathlib import Path
import sys
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.preprocessing import load_audio


MODEL_NAME = "facebook/wav2vec2-base-960h"
DATASET_ROOT = Path(r"C:\Users\Darshini Durga\OneDrive\Desktop\ASVPROOFS\LA")

AUDIO_DIR = DATASET_ROOT / "ASVspoof2019_LA_train" / "flac"
NUM_FILES = 100


print("=" * 70)
print("VOICEGUARD V2 — WAV2VEC2 SEQUENCE EMBEDDING TEST")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

print("\nLoading Wav2Vec2...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

files = sorted(AUDIO_DIR.glob("*.flac"))[:NUM_FILES]

print(f"Files to process: {len(files)}")

embeddings = []

with torch.no_grad():

    for i, file_path in enumerate(files, 1):

        audio = load_audio(file_path)

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_values = inputs.input_values.to(device)

        outputs = model(input_values)

        sequence = outputs.last_hidden_state

        embeddings.append(sequence.squeeze(0).cpu().numpy())

        if i % 10 == 0:
            print(f"Processed {i}/{len(files)}")

embeddings = np.stack(embeddings)

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"Embedding shape: {embeddings.shape}")
print(f"Data type: {embeddings.dtype}")

expected = (NUM_FILES, 149, 768)

print(f"Expected shape: {expected}")

if embeddings.shape == expected:
    print("\nSUCCESS: Sequence embeddings are correct.")
else:
    print("\nWARNING: Shape differs from expected.")

output_path = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "test_sequence_embeddings.npz"
)

np.savez_compressed(
    output_path,
    embeddings=embeddings
)

print(f"\nSaved test embeddings:")
print(output_path)

print("\nGPU extraction test complete.")