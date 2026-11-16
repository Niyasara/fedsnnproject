print(f"### THIS CLIENT_APP FILE IS: {__file__}")
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from flwr.clientapp import ClientApp
from flwr.app import Message, ArrayRecord, MetricRecord, RecordDict

# ------------------------------------------------------------
# Project path (fixed, so it works from ~/.flwr/apps too)
# ------------------------------------------------------------
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(
    os.environ.get(
        "FEDSNN_PROJECT_ROOT",
        Path(__file__).resolve().parent
    )
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from model.lif_snn import LIFSNN
from dataset.dataset_loader import NMNISTDataset

# ------------------------------------------------------------
# Fixed settings (delta and number of clients come from Flower)
# ------------------------------------------------------------
BATCH_SIZE = 32
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.001
PARTITION_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# IID partition
# ------------------------------------------------------------
def get_client_data(client_id: int, num_clients: int):

    dataset = NMNISTDataset(
        root=str(PROJECT_ROOT / "data" / "nmnist" / "N-MNIST"),
        train=True,
    )

    total_samples = len(dataset)

    generator = torch.Generator().manual_seed(PARTITION_SEED)
    shuffled_indices = torch.randperm(
        total_samples, generator=generator
    ).tolist()

    samples_per_client = total_samples // num_clients
    start = client_id * samples_per_client

    if client_id == num_clients - 1:
        end = total_samples
    else:
        end = start + samples_per_client

    client_dataset = Subset(dataset, shuffled_indices[start:end])

    return DataLoader(
        client_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ------------------------------------------------------------
# Local training
# ------------------------------------------------------------
def train_model(model, train_loader):

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

            outputs = model(frames)       # [time, batch, classes]
            outputs = outputs.sum(dim=0)  # spike count over time

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size

            progress_bar.set_postfix(
                loss=f"{total_loss / total:.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

        last_epoch_loss = total_loss / total
        last_epoch_accuracy = 100.0 * correct / total

        print(
            f"      Epoch {epoch + 1}/{LOCAL_EPOCHS} "
            f"| Loss: {last_epoch_loss:.4f} "
            f"| Accuracy: {last_epoch_accuracy:.2f}%"
        )

    return last_epoch_loss, last_epoch_accuracy


# ------------------------------------------------------------
# Flower client
# ------------------------------------------------------------
app = ClientApp()


@app.train()
def train(msg, context):

    client_id = int(context.node_config.get("partition-id", 0))

    # Read per-run settings from Flower (no hard-coded values)
    DELTA = float(context.run_config["delta"])
    num_clients = int(context.node_config["num-partitions"])

    print()
    print("-" * 70)
    print(f"CLIENT {client_id} - EVENT TRIGGER CHECK "
          f"(delta={DELTA}, clients={num_clients})")
    print("-" * 70)

    # Load global model
    model = LIFSNN().to(DEVICE)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Client data
    train_loader = get_client_data(client_id, num_clients)
    print(f"Client {client_id} samples: {len(train_loader.dataset)}")

    # Previous local loss (kept between rounds)
    if "previous_loss" in context.state:
        previous_loss = float(context.state["previous_loss"]["train_loss"])
    else:
        previous_loss = float("inf")

    # Local training
    print(f"Client {client_id}: Local training started...")
    current_loss, current_accuracy = train_model(model, train_loader)

    # Event trigger
    if previous_loss == float("inf"):
        improvement = float("inf")
        upload = True
        print(f"Client {client_id}: First update -> UPLOAD")
    else:
        improvement = previous_loss - current_loss
        upload = improvement >= DELTA

    context.state["previous_loss"] = MetricRecord(
        {"train_loss": float(current_loss)}
    )

    # Display
    print()
    if previous_loss != float("inf"):
        print(f"Previous loss : {previous_loss:.6f}")
    else:
        print("Previous loss : N/A")
    print(f"Current loss  : {current_loss:.6f}")
    if improvement != float("inf"):
        print(f"Loss improvement Φ: {improvement:.6f}")
    print(f"Threshold δ   : {DELTA:.6f}")
    print(f"Local accuracy: {current_accuracy:.2f}%")

    metrics = MetricRecord(
        {
            "train_loss": float(current_loss),
            "train_accuracy": float(current_accuracy),
            "num-examples": len(train_loader.dataset),
            "uploaded": 1.0 if upload else 0.0,
            "loss_improvement": float(
                improvement if improvement != float("inf") else 0.0
            ),
        }
    )

    # Upload case
    if upload:
        print()
        print(f"✓ CLIENT {client_id}: UPLOAD")
        if improvement != float("inf"):
            print(f"  Φ >= δ ({improvement:.6f} >= {DELTA:.6f})")
        else:
            print("  Initial client update")

        content = RecordDict(
            {
                "arrays": ArrayRecord(model.state_dict()),
                "metrics": metrics,
            }
        )

    # Skip case: no ArrayRecord, only metrics
    else:
        print()
        print(f"✗ CLIENT {client_id}: SKIP")
        print(f"  Φ < δ ({improvement:.6f} < {DELTA:.6f})")
        print("  Model update will NOT be uploaded.")

        content = RecordDict({"metrics": metrics})

    print(f"CLIENT {client_id} DEBUG: reply keys = {list(content.keys())}")

    return Message(content=content, reply_to=msg)