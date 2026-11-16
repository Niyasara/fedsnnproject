print(f"### THIS CLIENT_APP FILE IS: {__file__}")
import os
import sys
import csv
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
# NUM_CLIENTS is the ONE value you change when scaling up
# (5 -> 10 -> 15 -> 20). It must be changed identically in
# server_app.py, and `num-supernodes` in ~/.flwr/config.toml
# must be set to the same number. See the run instructions.
# ------------------------------------------------------------

NUM_CLIENTS = 5

BATCH_SIZE = 32
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.001

# ------------------------------------------------------------
# Experiment 6: Non-IID Data Analysis - SEVERE Non-IID split
#
# Dirichlet partition over class labels, per class, across
# clients (Hsu et al. 2019). Smaller alpha = stronger skew:
#
#   alpha = 100  -> nearly IID
#   alpha = 1.0  -> mild non-IID    (previous experiment)
#   alpha = 0.1  -> severe non-IID  <- USED HERE
#
# At alpha = 0.1 each client is dominated by a few digits.
# ------------------------------------------------------------

NONIID_ALPHA = 0.1   # Severe Non-IID

# Every client must own at least this many samples. With
# alpha = 0.1 a client can otherwise receive almost nothing
# (or nothing at all, which would crash local training), so
# the Dirichlet draw is repeated - deterministically, from the
# same seeded generator - until all clients meet the minimum.
MIN_SAMPLES_PER_CLIENT = 300
MAX_PARTITION_ATTEMPTS = 5000

NUM_CLASSES = 10

PARTITION_SEED = 42

# Must match RESULTS_DIR in server_app.py (used only to save the
# partition summary CSV next to that run's other results).
RESULTS_SUBDIR = "phase5_severenoniid_fedavg"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLIENT INFORMATION
# ============================================================

print("=" * 70)
print("EXPERIMENT 6 - SEVERE NON-IID FEDAVG CLIENT")
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
print(f"Non-IID alpha: {NONIID_ALPHA} (severe)")
print(f"Min samples per client: {MIN_SAMPLES_PER_CLIENT}")

print("=" * 70)


# ============================================================
# SAVE PARTITION SUMMARY (class counts per client)
#
# Written once per actor process, atomically, so concurrent
# actors computing the identical partition cannot corrupt the
# file. Failure to save is never fatal.
# ============================================================

def save_partition_summary(class_counts, sizes):

    try:
        out_dir = (
            PROJECT_ROOT
            / "results"
            / RESULTS_SUBDIR
            / f"clients_{NUM_CLIENTS}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / "partition_class_distribution.csv"
        tmp_file = out_dir / f".partition_{os.getpid()}.tmp"

        with open(tmp_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Client", "Total samples"]
                + [f"Class {c}" for c in range(NUM_CLASSES)]
                + ["Top-2 class share (%)"]
            )
            for client_id in range(NUM_CLIENTS):
                row = class_counts[client_id]
                top2 = np.sort(row)[-2:].sum()
                share = 100.0 * top2 / max(sizes[client_id], 1)
                writer.writerow(
                    [client_id, int(sizes[client_id])]
                    + [int(v) for v in row]
                    + [f"{share:.1f}"]
                )

        os.replace(tmp_file, out_file)

    except Exception as exc:
        print(f"WARNING: could not save partition summary: {exc}")


# ============================================================
# DIRICHLET NON-IID PARTITION
#
# Computed once per process and cached. Every actor derives the
# identical partition from PARTITION_SEED and reads only its own
# client's slice, the same pattern the IID code used.
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

    client_indices = None
    attempts_used = 0

    # --------------------------------------------------------
    # Draw Dirichlet partitions until every client has at least
    # MIN_SAMPLES_PER_CLIENT samples.
    # --------------------------------------------------------

    for attempt in range(1, MAX_PARTITION_ATTEMPTS + 1):

        attempts_used = attempt

        candidate = [[] for _ in range(NUM_CLIENTS)]

        for class_label in sorted(labels_by_class.keys()):

            indices = np.array(labels_by_class[class_label])
            rng.shuffle(indices)

            proportions = rng.dirichlet(
                alpha=[NONIID_ALPHA] * NUM_CLIENTS
            )

            split_points = (
                np.cumsum(proportions) * len(indices)
            ).astype(int)[:-1]

            class_splits = np.split(indices, split_points)

            for client_id, split in enumerate(class_splits):
                candidate[client_id].extend(split.tolist())

        if min(len(c) for c in candidate) >= MIN_SAMPLES_PER_CLIENT:
            client_indices = candidate
            break

    if client_indices is None:
        raise RuntimeError(
            f"Could not build a severe non-IID partition with at least "
            f"{MIN_SAMPLES_PER_CLIENT} samples per client for "
            f"{NUM_CLIENTS} clients after {MAX_PARTITION_ATTEMPTS} attempts. "
            f"Lower MIN_SAMPLES_PER_CLIENT."
        )

    # --------------------------------------------------------
    # Per-client class counts, diagnostics, summary CSV
    # --------------------------------------------------------

    class_counts = np.zeros((NUM_CLIENTS, NUM_CLASSES), dtype=np.int64)

    for client_id in range(NUM_CLIENTS):
        for idx in client_indices[client_id]:
            _, label = dataset.samples[idx]
            class_counts[client_id, label] += 1

    sizes = class_counts.sum(axis=1)

    print()
    print(
        f"Severe non-IID partition (alpha={NONIID_ALPHA}, "
        f"{attempts_used} draw(s) to meet the minimum size):"
    )

    for client_id in range(NUM_CLIENTS):

        counts_str = ", ".join(
            f"{c}:{class_counts[client_id, c]}" for c in range(NUM_CLASSES)
        )

        top2 = np.sort(class_counts[client_id])[-2:].sum()
        share = 100.0 * top2 / max(sizes[client_id], 1)

        print(
            f"  Client {client_id} "
            f"(n={sizes[client_id]}, top-2 classes={share:.0f}%): "
            f"{counts_str}"
        )

    save_partition_summary(class_counts, sizes)

    _CACHED_PARTITION = client_indices

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
    print(f"CLIENT {client_id} - SEVERE NON-IID FEDAVG ROUND")
    print("-" * 70)

    # ========================================================
    # LOAD GLOBAL MODEL
    # ========================================================

    model = LIFSNN().to(DEVICE)

    arrays = msg.content["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())

    # ========================================================
    # CLIENT DATA (Severe Non-IID partition)
    # ========================================================

    train_loader = get_client_data(client_id)

    print(f"Client {client_id} samples: {len(train_loader.dataset)}")

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