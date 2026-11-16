print(f"### THIS CLIENT_APP FILE IS: {__file__}")
import os
import sys
import math
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

# ------------------------------------------------------------
# Phase 4 (folder: phase3.1) - Adaptive Trigger Design
#
# Adaptive threshold:
#     delta_i(t) = ALPHA * Std(L_i)
#
# Std(L_i) is computed from each client's own history of local
# training losses (Welford's online algorithm - no need to
# store the full loss history, just running mean/variance).
#
# ALPHA is the one tunable knob here, playing the same role
# DELTA played in the fixed-threshold experiments.
# ------------------------------------------------------------

ALPHA = 1.0
# Fallback threshold used only while a client has fewer than 2
# recorded losses (Std is undefined with < 2 samples). Keeps
# behaviour sane for round 1-2 before real variability exists.
FALLBACK_DELTA = 0.001

PARTITION_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLIENT INFORMATION
# ============================================================

print("=" * 70)
print("PHASE 4 (phase3.1) - ADAPTIVE EVENT-TRIGGERED FEDERATED SNN CLIENT")
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
print(f"Alpha (adaptive multiplier): {ALPHA}")
print(f"Fallback delta (rounds 1-2): {FALLBACK_DELTA}")

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
    print(f"CLIENT {client_id} - ADAPTIVE EVENT TRIGGER CHECK")
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
    # PREVIOUS LOCAL LOSS (for Phi_i(t) = L_i(t-1) - L_i(t))
    # ========================================================

    if "previous_loss" in context.state:
        previous_loss = float(
            context.state["previous_loss"]["train_loss"]
        )
    else:
        previous_loss = float("inf")

    # ========================================================
    # LOSS HISTORY STATS (Welford's online algorithm)
    #
    # Tracks running count (n), mean, and M2 (sum of squared
    # deviations) of this client's past local losses, so we
    # can compute Std(L_i) without storing the full history.
    #
    # Std(L_i) here is based on losses seen BEFORE this round
    # (i.e. it reflects the client's historical variability,
    # not including the loss it is about to produce this round).
    # ========================================================

    if "loss_stats" in context.state:
        stats = context.state["loss_stats"]
        loss_n = int(stats["n"])
        loss_mean = float(stats["mean"])
        loss_m2 = float(stats["m2"])
    else:
        loss_n = 0
        loss_mean = 0.0
        loss_m2 = 0.0

    if loss_n >= 2:
        variance = loss_m2 / (loss_n - 1)   # sample variance
        loss_std = math.sqrt(variance)
    else:
        loss_std = None   # not enough history yet

    # ========================================================
    # LOCAL TRAINING
    # ========================================================

    print(f"Client {client_id}: Local training started...")

    current_loss, current_accuracy = train_model(model, train_loader)

    # ========================================================
    # ADAPTIVE EVENT TRIGGER
    #
    #   delta_i(t) = ALPHA * Std(L_i)
    #   upload if  Phi_i(t) = L_i(t-1) - L_i(t)  >=  delta_i(t)
    # ========================================================

    if previous_loss == float("inf"):

        # First round for this client: no previous loss to
        # compare against, so always upload (same rule as the
        # fixed-threshold experiments).
        improvement = float("inf")
        adaptive_delta = None
        upload = True
        print(f"Client {client_id}: First update -> UPLOAD")

    else:

        improvement = previous_loss - current_loss

        if loss_std is None:
            # Not enough history yet to compute Std(L_i) -
            # fall back to a small fixed threshold for this
            # round only.
            adaptive_delta = FALLBACK_DELTA
        else:
            adaptive_delta = ALPHA * loss_std

        upload = improvement >= adaptive_delta

    # ========================================================
    # UPDATE LOSS HISTORY STATS (Welford update, using this
    # round's current_loss, for NEXT round's Std(L_i))
    # ========================================================

    loss_n += 1
    delta_from_mean = current_loss - loss_mean
    loss_mean += delta_from_mean / loss_n
    delta_from_new_mean = current_loss - loss_mean
    loss_m2 += delta_from_mean * delta_from_new_mean

    context.state["loss_stats"] = MetricRecord(
        {
            "n": float(loss_n),
            "mean": float(loss_mean),
            "m2": float(loss_m2),
        }
    )

    # ========================================================
    # SAVE CURRENT LOSS FOR NEXT ROUND'S Phi CALCULATION
    # ========================================================

    context.state["previous_loss"] = MetricRecord(
        {"train_loss": float(current_loss)}
    )

    # ========================================================
    # DISPLAY TRIGGER RESULT
    # ========================================================

    print()

    if previous_loss != float("inf"):
        print(f"Previous loss : {previous_loss:.6f}")
    else:
        print("Previous loss : N/A")

    print(f"Current loss  : {current_loss:.6f}")

    if improvement != float("inf"):
        print(f"Loss improvement Phi: {improvement:.6f}")

    if loss_std is not None:
        print(f"Std(L_i) (history): {loss_std:.6f}")
    else:
        print("Std(L_i) (history): N/A (fewer than 2 samples yet)")

    if adaptive_delta is not None:
        print(f"Adaptive delta_i(t) = ALPHA * Std(L_i) = {adaptive_delta:.6f}")

    print(f"Local accuracy: {current_accuracy:.2f}%")

    # ========================================================
    # UPLOAD CASE
    # ========================================================

    if upload:

        print()
        print(f"CLIENT {client_id}: UPLOAD")

        if adaptive_delta is not None:
            print(
                f"  Phi >= delta_i(t) "
                f"({improvement:.6f} >= {adaptive_delta:.6f})"
            )
        else:
            print("  Initial client update")

        model_record = ArrayRecord(model.state_dict())

        metrics = MetricRecord(
            {
                "train_loss": float(current_loss),
                "train_accuracy": float(current_accuracy),
                "num-examples": len(train_loader.dataset),
                "uploaded": 1.0,
                "loss_improvement": float(
                    improvement if improvement != float("inf") else 0.0
                ),
                "adaptive_delta": float(
                    adaptive_delta if adaptive_delta is not None else 0.0
                ),
            }
        )

        content = RecordDict(
            {
                "arrays": model_record,
                "metrics": metrics,
            }
        )

        print(f"CLIENT {client_id} DEBUG: reply keys = {list(content.keys())}")

        return Message(content=content, reply_to=msg)

    # ========================================================
    # SKIP CASE
    # ========================================================

    else:

        print()
        print(f"CLIENT {client_id}: SKIP")
        print(
            f"  Phi < delta_i(t) "
            f"({improvement:.6f} < {adaptive_delta:.6f})"
        )
        print("  Model update will NOT be uploaded.")

        metrics = MetricRecord(
            {
                "train_loss": float(current_loss),
                "train_accuracy": float(current_accuracy),
                "num-examples": len(train_loader.dataset),
                "uploaded": 0.0,
                "loss_improvement": float(improvement),
                "adaptive_delta": float(adaptive_delta),
            }
        )

        content = RecordDict({"metrics": metrics})

        print(f"CLIENT {client_id} DEBUG: reply keys = {list(content.keys())}")

        return Message(content=content, reply_to=msg)