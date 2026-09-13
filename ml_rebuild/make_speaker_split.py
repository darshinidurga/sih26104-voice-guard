from pathlib import Path
import pandas as pd
import random


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BASE_DIR = Path(__file__).resolve().parent
METADATA_DIR = BASE_DIR / "data" / "metadata"
SPLIT_DIR = BASE_DIR / "data" / "splits"

TRAIN_METADATA = METADATA_DIR / "train_metadata.csv"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD TRAIN METADATA
# ============================================================

df = pd.read_csv(TRAIN_METADATA)

print("=" * 70)
print("VOICEGUARD SPEAKER-DISJOINT SPLIT")
print("=" * 70)

print(f"\nTotal training records: {len(df):,}")

speakers = sorted(df["speaker_id"].unique())

print(f"Unique speakers: {len(speakers)}")

# ============================================================
# SPLIT SPEAKERS
# ============================================================

random.seed(SEED)

shuffled_speakers = speakers.copy()
random.shuffle(shuffled_speakers)

# 80% speakers for training, 20% for validation
num_train_speakers = int(len(shuffled_speakers) * 0.80)

train_speakers = sorted(shuffled_speakers[:num_train_speakers])
val_speakers = sorted(shuffled_speakers[num_train_speakers:])

train_df = df[df["speaker_id"].isin(train_speakers)].copy()
val_df = df[df["speaker_id"].isin(val_speakers)].copy()


# ============================================================
# VERIFY NO SPEAKER OVERLAP
# ============================================================

train_set = set(train_df["speaker_id"])
val_set = set(val_df["speaker_id"])

overlap = train_set.intersection(val_set)


# ============================================================
# SAVE
# ============================================================

train_path = SPLIT_DIR / "train_split.csv"
val_path = SPLIT_DIR / "val_split.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)


# ============================================================
# REPORT
# ============================================================

print("\nTRAIN SPEAKERS:")
print(train_speakers)

print("\nVALIDATION SPEAKERS:")
print(val_speakers)

print("\n" + "=" * 70)
print("SPLIT SUMMARY")
print("=" * 70)

print(f"Training speakers    : {len(train_speakers)}")
print(f"Validation speakers  : {len(val_speakers)}")

print(f"\nTraining records     : {len(train_df):,}")
print(f"Validation records   : {len(val_df):,}")

print("\nTraining labels:")
print(train_df["label"].value_counts())

print("\nValidation labels:")
print(val_df["label"].value_counts())

print(f"\nSpeaker overlap      : {len(overlap)}")

if len(overlap) == 0:
    print("\nGOOD: Training and validation speakers are completely disjoint.")
else:
    print("\nERROR: Speaker overlap detected!")

print("\nSaved:")
print(train_path)
print(val_path)

print("\nNo audio was copied or modified.")
print("No GPU work was performed.")