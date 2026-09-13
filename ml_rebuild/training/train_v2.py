import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# PATHS
# ============================================================

TRAIN_EMB = r"data\memmap\embeddings.npy"
TRAIN_LABELS = r"data\memmap\labels.npy"

VAL_EMB = r"data\memmap_val\embeddings.npy"
VAL_LABELS = r"data\memmap_val\labels.npy"

MODEL_OUT = r"models\voiceguard_v2a.pth"

# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 32
EPOCHS = 25
PATIENCE = 5
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# LOAD MEMMAPS
# ============================================================

train_embeddings = np.load(TRAIN_EMB, mmap_mode="r")
train_labels = np.load(TRAIN_LABELS, mmap_mode="r")

val_embeddings = np.load(VAL_EMB, mmap_mode="r")
val_labels = np.load(VAL_LABELS, mmap_mode="r")

print("Train:", train_embeddings.shape)
print("Train labels:", train_labels.shape)

print("Val:", val_embeddings.shape)
print("Val labels:", val_labels.shape)

# ============================================================
# DATASET
# ============================================================

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = np.array(self.embeddings[idx], dtype=np.float32, copy=True)
        y = float(self.labels[idx])

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


train_dataset = EmbeddingDataset(train_embeddings, train_labels)
val_dataset = EmbeddingDataset(val_embeddings, val_labels)

# ============================================================
# BALANCED SAMPLING
# ============================================================

train_labels_np = np.asarray(train_labels)

class_counts = np.bincount(train_labels_np)
class_weights = 1.0 / class_counts

sample_weights = class_weights[train_labels_np]

sampler = WeightedRandomSampler(
    torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

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

        # x: batch × time × features
        scores = self.attention(x)

        weights = torch.softmax(scores, dim=1)

        pooled = torch.sum(weights * x, dim=1)

        return pooled


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

print("\nModel:")
print(model)

# ============================================================
# TRAINING
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

criterion = nn.BCEWithLogitsLoss()

best_f1 = 0.0
best_state = None
patience_counter = 0

# ============================================================
# TRAIN LOOP
# ============================================================

for epoch in range(1, EPOCHS + 1):

    model.train()

    train_losses = []

    for x, y in train_loader:

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        train_losses.append(loss.item())

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device, non_blocking=True)

            logits = model(x)

            probabilities = torch.sigmoid(logits)

            predictions = (probabilities >= 0.5).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(y.numpy())

    accuracy = accuracy_score(all_labels, all_predictions)

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    avg_loss = np.mean(train_losses)

    print(
        f"Epoch {epoch:02d} | "
        f"Loss {avg_loss:.4f} | "
        f"Val Acc {accuracy:.4f} | "
        f"Precision {precision:.4f} | "
        f"Recall {recall:.4f} | "
        f"F1 {f1:.4f}"
    )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if f1 > best_f1:

        best_f1 = f1
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0

        print("  ✓ New best model")

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print("  Early stopping.")

            break

# ============================================================
# SAVE BEST MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_val_f1": best_f1
        },
        MODEL_OUT
    )

    print("\nBest validation F1:", round(best_f1, 4))
    print("Saved:", MODEL_OUT)

else:

    print("ERROR: No model was saved.")