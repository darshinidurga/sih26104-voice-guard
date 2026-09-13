from pathlib import Path
from collections import Counter
import csv

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\Darshini Durga\OneDrive\Desktop\ASVPROOFS\LA"
)

OUTPUT_DIR = Path(__file__).parent / "data" / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Official protocol locations
PROTOCOL_ROOT = DATASET_ROOT / "ASVspoof2019_LA_cm_protocols"

# Audio locations
TRAIN_AUDIO = DATASET_ROOT / "ASVspoof2019_LA_train" / "flac"
DEV_AUDIO = DATASET_ROOT / "ASVspoof2019_LA_dev" / "flac"
EVAL_AUDIO = DATASET_ROOT / "ASVspoof2019_LA_eval" / "flac"


# ============================================================
# FIND PROTOCOL FILES
# ============================================================

def find_protocol(keyword):
    matches = list(PROTOCOL_ROOT.rglob(f"*{keyword}*"))

    txt_files = [
        p for p in matches
        if p.is_file() and p.suffix.lower() == ".txt"
    ]

    return txt_files


def choose_protocol(keyword):
    files = find_protocol(keyword)

    if not files:
        return None

    # Prefer CM protocol files
    cm_files = [p for p in files if "cm" in p.name.lower()]

    if cm_files:
        return cm_files[0]

    return files[0]


train_protocol = choose_protocol("train")
dev_protocol = choose_protocol("dev")
eval_protocol = choose_protocol("eval")


print("=" * 70)
print("VOICEGUARD DATASET AUDIT")
print("=" * 70)

print("\nDataset root:")
print(DATASET_ROOT)

print("\nProtocol root:")
print(PROTOCOL_ROOT)

print("\nProtocol files found:")

print("TRAIN:", train_protocol)
print("DEV  :", dev_protocol)
print("EVAL :", eval_protocol)


# ============================================================
# PARSE PROTOCOL
# ============================================================

def parse_protocol(protocol_path, partition):
    records = []

    if protocol_path is None:
        print(f"\nWARNING: No protocol found for {partition}")
        return records

    print(f"\nReading {partition} protocol:")
    print(protocol_path)

    with open(protocol_path, "r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # Expected ASVspoof CM format:
            #
            # speaker_id
            # utterance_id
            # -
            # attack_id
            # label
            #
            # Example:
            # LA_0079 LA_T_1138215 - - bonafide
            #
            # Example spoof:
            # LA_0079 LA_T_1004644 - A01 spoof

            if len(parts) < 5:
                print(
                    f"WARNING: {partition} line {line_number} "
                    f"has only {len(parts)} fields:"
                )
                print(line)
                continue

            speaker_id = parts[0]
            utterance_id = parts[1]
            attack_id = parts[3]
            label = parts[4].lower()

            records.append({
                "partition": partition,
                "speaker_id": speaker_id,
                "utterance_id": utterance_id,
                "attack_id": attack_id,
                "label": label,
                "protocol_line": line_number,
            })

    return records


train_records = parse_protocol(
    train_protocol,
    "train"
)

dev_records = parse_protocol(
    dev_protocol,
    "dev"
)

eval_records = parse_protocol(
    eval_protocol,
    "eval"
)


# ============================================================
# AUDIO PATH RESOLUTION
# ============================================================

AUDIO_ROOTS = {
    "train": TRAIN_AUDIO,
    "dev": DEV_AUDIO,
    "eval": EVAL_AUDIO,
}


def resolve_audio_path(partition, utterance_id):
    root = AUDIO_ROOTS[partition]

    path = root / f"{utterance_id}.flac"

    return path


def add_audio_paths(records):

    missing = []

    for record in records:

        path = resolve_audio_path(
            record["partition"],
            record["utterance_id"]
        )

        record["audio_path"] = str(path)

        record["audio_exists"] = path.exists()

        if not path.exists():
            missing.append(record)

    return missing


train_missing = add_audio_paths(train_records)
dev_missing = add_audio_paths(dev_records)
eval_missing = add_audio_paths(eval_records)


# ============================================================
# DATASET STATISTICS
# ============================================================

def print_statistics(records, partition):

    print("\n" + "=" * 70)
    print(partition.upper())
    print("=" * 70)

    total = len(records)

    speakers = set(
        r["speaker_id"]
        for r in records
    )

    labels = Counter(
        r["label"]
        for r in records
    )

    attacks = Counter(
        r["attack_id"]
        for r in records
        if r["label"] == "spoof"
    )

    utterance_ids = [
        r["utterance_id"]
        for r in records
    ]

    duplicate_ids = [
        uid
        for uid, count in Counter(utterance_ids).items()
        if count > 1
    ]

    existing = sum(
        r["audio_exists"]
        for r in records
    )

    missing = total - existing

    print(f"Total utterances : {total:,}")
    print(f"Unique speakers  : {len(speakers):,}")

    print("\nLabels:")
    for label, count in labels.items():
        print(f"  {label:12s}: {count:,}")

    print("\nSpoof attack types:")

    if attacks:
        for attack, count in sorted(attacks.items()):
            print(f"  {attack:12s}: {count:,}")
    else:
        print("  None")

    print("\nAudio:")
    print(f"  Existing : {existing:,}")
    print(f"  Missing  : {missing:,}")

    print("\nDuplicate utterance IDs:")
    print(f"  {len(duplicate_ids):,}")

    if duplicate_ids:
        print("  Examples:")
        for uid in duplicate_ids[:10]:
            print("   ", uid)

    return {
        "total": total,
        "speakers": speakers,
        "labels": labels,
        "attacks": attacks,
        "duplicate_ids": duplicate_ids,
        "missing": missing,
    }


train_stats = print_statistics(
    train_records,
    "train"
)

dev_stats = print_statistics(
    dev_records,
    "dev"
)

eval_stats = print_statistics(
    eval_records,
    "eval"
)


# ============================================================
# SPEAKER OVERLAP CHECK
# ============================================================

print("\n" + "=" * 70)
print("SPEAKER OVERLAP CHECK")
print("=" * 70)

train_speakers = train_stats["speakers"]
dev_speakers = dev_stats["speakers"]
eval_speakers = eval_stats["speakers"]

train_dev_overlap = train_speakers & dev_speakers
train_eval_overlap = train_speakers & eval_speakers
dev_eval_overlap = dev_speakers & eval_speakers

print(
    f"TRAIN ∩ DEV   : {len(train_dev_overlap):,}"
)

print(
    f"TRAIN ∩ EVAL  : {len(train_eval_overlap):,}"
)

print(
    f"DEV ∩ EVAL    : {len(dev_eval_overlap):,}"
)

if train_dev_overlap:
    print("\nWARNING: TRAIN/DEV speaker overlap detected.")

if train_eval_overlap:
    print("\nWARNING: TRAIN/EVAL speaker overlap detected.")

if dev_eval_overlap:
    print("\nWARNING: DEV/EVAL speaker overlap detected.")

if not (
    train_dev_overlap
    or train_eval_overlap
    or dev_eval_overlap
):
    print("\nGOOD: No speaker overlap detected between official partitions.")


# ============================================================
# SAVE CLEAN METADATA CSV FILES
# ============================================================

def save_csv(records, filename):

    output_path = OUTPUT_DIR / filename

    fields = [
        "partition",
        "speaker_id",
        "utterance_id",
        "attack_id",
        "label",
        "audio_path",
        "audio_exists",
        "protocol_line",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for record in records:
            writer.writerow(record)

    print(
        f"\nSaved metadata: {output_path}"
    )


save_csv(
    train_records,
    "train_metadata.csv"
)

save_csv(
    dev_records,
    "dev_metadata.csv"
)

save_csv(
    eval_records,
    "eval_metadata.csv"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL AUDIT SUMMARY")
print("=" * 70)

print(
    f"TRAIN: {len(train_records):,} records"
)

print(
    f"DEV  : {len(dev_records):,} records"
)

print(
    f"EVAL : {len(eval_records):,} records"
)

print(
    f"Missing audio files: "
    f"{len(train_missing) + len(dev_missing) + len(eval_missing):,}"
)

print(
    f"TRAIN/DEV speaker overlap: "
    f"{len(train_dev_overlap):,}"
)

print(
    f"TRAIN/EVAL speaker overlap: "
    f"{len(train_eval_overlap):,}"
)

print(
    f"DEV/EVAL speaker overlap: "
    f"{len(dev_eval_overlap):,}"
)

print("\nAudit complete.")
print("NO MODEL TRAINING WAS PERFORMED.")
print("NO GPU WORK WAS PERFORMED.")