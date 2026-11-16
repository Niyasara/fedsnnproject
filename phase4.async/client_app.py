print(f"### THIS CLIENT_APP FILE IS: {__file__}")
import os
import sys
from pathlib import Path

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

NUM_CLIENTS = 5

BATCH_SIZE = 32
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.001

PARTITION_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLIENT INFORMATION
# ============================================================
#
# NOTE: Phase 4 (async aggregation) tests a different
# AGGREGATION RULE on the server, not a different client
# participation trigger. So, unlike phase3 / phase3.1, this
# client always trains and always returns its update - there
# is no event-trigger upload/skip decision here. The server
# is what changed: it no longer weighted-averages all clients
# together, it folds each client's update in one at a time
# using an incremental update rule.
# ============================================================

print("=" * 70)
print("PHASE 5 (phase4.async) - ASYNCHRONOUS AGGREGATION CLIENT")
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

print("=" * 70)


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

    total_samples = len(dataset)

    generator = torch.Generator().manual_seed(PARTITION_SEED)

    shuffled_indices = torch.randperm(
        total_samples,
        generator=generator
    ).tolist()

    samples_per_client = total_samples // NUM_CLIENTS

    start = client_id * samples_per_client

    if client_id == NUM_CLIENTS - 1:
        end = total_samples
    else:
        end = start + samples_per_client

    client_indices = shuffled_indices[start:end]

    client_dataset = Subset(dataset, client_indices)

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
    print(f"CLIENT {client_id} - ASYNC AGGREGATION ROUND")
    print("-" * 70)

    # ========================================================
    # LOAD GLOBAL MODEL
    # ========================================================

    model = LIFSNN().to(DEVICE)

    arrays = msg.content["arrays"]
    model.load_state_dict(arrays.to_torch_state_dict())

    # ========================================================
    # CLIENT DATA
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
    # ALWAYS RETURN THE UPDATE
    #
    # No event-trigger here - every client always sends its
    # update. What makes this "asynchronous-style" is how the
    # SERVER processes these updates (see server_app.py), not
    # whether the client withholds anything.
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