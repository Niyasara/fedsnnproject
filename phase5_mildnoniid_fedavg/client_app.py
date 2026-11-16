print(f"### THIS CLIENT_APP FILE IS: {__file__}")
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from flwr.clientapp import ClientApp
from flwr.app import Message, ArrayRecord, MetricRecord, RecordDict


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    os.environ.get(
        "FEDSNN_PROJECT_ROOT",
        Path(__file__).resolve().parents[1]
    )
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT MODEL AND DATASET
# ============================================================

from model.lif_snn import LIFSNN
from dataset.dataset_loader import NMNISTDataset


# ============================================================
# CONFIGURATION
# ============================================================
#
# IMPORTANT: this is the ONE value you change when scaling up
# client count (5 -> 10 -> 15 -> 20). Everything else (results
# folder, SuperNode handling) follows automatically. See the
# running instructions for the matching
# `flwr federation simulation-config --num-supernodes=N` step
# you must also run whenever you change this.
# ------------------------------------------------------------

NUM_CLIENTS = 5

BATCH_SIZE = 32
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.001

# ------------------------------------------------------------
# Experiment 6: Non-IID Data Analysis - Mild Non-IID split
#
# Partition uses a Dirichlet distribution over class labels,
# per client (the standard non-IID partitioning method used
# in federated learning literature, e.g. Hsu et al. 2019).
#
# NONIID_ALPHA controls skew:
#   Large alpha (e.g. 100)  -> nearly IID (every client gets a
#                               near-uniform mix of all classes)
#   Mild  alpha (e.g. 1.0)  -> moderate skew  <- USED HERE
#   Small alpha (e.g. 0.1)  -> severe skew (clients dominated
#                               by 1-2 classes) - a later phase
# ------------------------------------------------------------

NONIID_ALPHA = 1.0   # Mild Non-IID

PARTITION_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLIENT INFORMATION
# ============================================================

print("=" * 70)
print("EXPERIMENT 6 - MILD NON-IID FEDAVG CLIENT")
print("=" * 70)

print(f"Device       : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
else:
    print("GPU          : CPU")

print(f"Clients      : {NUM_CLIENTS}")
print(f"Local epochs : {LOCAL_EPOCHS}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Non-IID alpha: {NONIID_ALPHA} (mild)")

print("=" * 70)


# ============================================================
# DIRICHLET NON-IID PARTITION
#
# Computed once per process, cached, so every client call in
# this actor reuses the same partition (deterministic given
# PARTITION_SEED - every client/actor computes the identical
# partition independently and just reads its own slice out of
# it, the same pattern your original IID code used).
# ============================================================

_CACHED_PARTITION = None


def build_noniid_partition(dataset):

    global _CACHED_PARTITION

    if _CACHED_PARTITION is not None:
        return _CACHED_PARTITION

    # --------------------------------------------------------
    # Group sample indices by class label
    # --------------------------------------------------------

    labels_by_class = defaultdict(list)

    for idx, (_, label) in enumerate(dataset.samples):
        labels_by_class[label].append(idx)

    rng = np.random.default_rng(PARTITION_SEED)

    client_indices = [[] for _ in range(NUM_CLIENTS)]

    # --------------------------------------------------------
    # For each class, draw a Dirichlet distribution over the
    # NUM_CLIENTS clients, and split that class's samples
    # according to the drawn proportions.
    # --------------------------------------------------------

    for class_label, indices in labels_by_class.items():

        indices = np.array(indices)
        rng.shuffle(indices)

        proportions = rng.dirichlet(
            alpha=[NONIID_ALPHA] * NUM_CLIENTS
        )

        # Convert proportions to cumulative split points
        split_points = (
            np.cumsum(proportions) * len(indices)
        ).astype(int)[:-1]

        class_splits = np.split(indices, split_points)

        for client_id, split in enumerate(class_splits):
            client_indices[client_id].extend(split.tolist())

    _CACHED_PARTITION = client_indices

    # --------------------------------------------------------
    # Diagnostic: print per-client class distribution so you
    # can visually confirm the split is genuinely non-IID.
    # --------------------------------------------------------

    print()
    print("Non-IID partition class distribution (samples per class):")

    for client_id in range(NUM_CLIENTS):

        counts = defaultdict(int)

        for idx in client_indices[client_id]:
            _, label = dataset.samples[idx]
            counts[label] += 1

        counts_str = ", ".join(
            f"{c}:{counts.get(c, 0)}" for c in range(10)
        )

        print(f"  Client {client_id}: {counts_str}")

    return client_indices


# ============================================================
# DATASET PARTITION
# ============================================================

def get_client_data(client_id: int):

    dataset = NMNISTDataset(
        root=str(
            PROJECT_ROOT
            / "data"
            / "nmnist"
            / "N-MNIST"
        ),
        train=True,
    )

    partition = build_noniid_partition(dataset)

    client_dataset = Subset(dataset, partition[client_id])

    loader = DataLoader(
        client_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


# ============================================================
# LOCAL TRAINING
# ============================================================

def train_model(model, train_loader):

    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    last_epoch_loss = None
    last_epoch_accuracy = None

    for epoch in range(LOCAL_EPOCHS):

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"      Epoch {epoch + 1}/{LOCAL_EPOCHS}",
            leave=False,
            unit="batch",
        )

        for frames, labels in progress_bar:

            frames = frames.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(frames)        # [time, batch, classes]
            outputs = outputs.sum(dim=0)   # sum spikes over time

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

            running_loss = total_loss / total
            running_accuracy = 100.0 * correct / total

            progress_bar.set_postfix(
                loss=f"{running_loss:.4f}",
                acc=f"{running_accuracy:.2f}%"
            )

        last_epoch_loss = total_loss / total
        last_epoch_accuracy = 100.0 * correct / total

        print(
            f"      Epoch {epoch + 1}/{LOCAL_EPOCHS} "
            f"| Loss: {last_epoch_loss:.4f} "
            f"| Accuracy: {last_epoch_accuracy:.2f}%"
        )

    return last_epoch_loss, last_epoch_accuracy


# ============================================================
# FLOWER CLIENT APP
# ============================================================

app = ClientApp()


@app.train()
def train(msg, context):

    client_id = int(context.node_config.get("partition-id", 0))

    print()
    print("-" * 70)
    print(f"CLIENT {client_id} - MILD NON-IID FEDAVG ROUND")
    print("-" * 70)

    # ========================================================
    # LOAD GLOBAL MODEL
    # ========================================================

    model = LIFSNN().to(DEVICE)

    arrays = msg.content["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())

    # ========================================================
    # CLIENT DATA (Mild Non-IID partition)
    # ========================================================

    train_loader = get_client_data(client_id)

    print(f"Client {client_id} samples: {len(train_loader.dataset)}")

    if len(train_loader.dataset) == 0:
        print(
            f"WARNING: Client {client_id} received 0 samples "
            f"from the non-IID partition this run."
        )

    # ========================================================
    # LOCAL TRAINING
    # ========================================================

    print(f"Client {client_id}: Local training started...")

    current_loss, current_accuracy = train_model(model, train_loader)

    print()
    print(f"Client {client_id}: training completed.")
    print(f"Final local loss     : {current_loss:.4f}")
    print(f"Final local accuracy : {current_accuracy:.2f}%")

    # ========================================================
    # PLAIN FEDAVG - every client always uploads
    # ========================================================

    model_record = ArrayRecord(model.state_dict())

    metrics = MetricRecord(
        {
            "train_loss": float(current_loss),
            "train_accuracy": float(current_accuracy),
            "num-examples": len(train_loader.dataset),
        }
    )

    content = RecordDict(
        {
            "arrays": model_record,
            "metrics": metrics,
        }
    )

    print(f"CLIENT {client_id}: sending update to server.")

    return Message(content=content, reply_to=msg)