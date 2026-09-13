from pathlib import Path
import zipfile
import shutil

BASE_DIR = Path(__file__).resolve().parents[1]

SOURCE = BASE_DIR / "data" / "train_sequence_embeddings.npz"
OUTPUT_DIR = BASE_DIR / "data" / "memmap"

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("VOICEGUARD — PREPARE DISK-BACKED EMBEDDINGS")
print("=" * 70)

print(f"\nSource: {SOURCE}")

with zipfile.ZipFile(SOURCE, "r") as z:

    print("\nFiles inside archive:")
    for name in z.namelist():
        print(" ", name)

    # Extract without loading the 8.7 GB array into RAM
    for name in ["embeddings.npy", "labels.npy", "utterance_ids.npy"]:

        output = OUTPUT_DIR / name

        print(f"\nExtracting {name}...")

        with z.open(name) as src, open(output, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)

        print(f"Saved: {output}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print("\nNo GPU work performed.")
print("Embeddings were streamed directly to disk.")