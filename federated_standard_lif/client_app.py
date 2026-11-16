import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get(
        "FEDSNN_PROJECT_ROOT",
        Path(__file__).resolve().parents[1]
    )
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import torch
import torch.nn as nn

from tqdm import tqdm

from flwr.app import (
    ArrayRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)

from flwr.clientapp import ClientApp

from model.lif_snn import LIFSNN
from dataset.dataset_loader import get_data_loaders


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 32
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.001

# Fixed seed used to shuffle indices before partitioning them
# across clients. Every client must use the SAME seed so that
# they agree on the same shuffled ordering (this keeps the
# start:end slices non-overlapping and still covering the
# full training set).
PARTITION_SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CREATE STANDARD LIF MODEL
# ============================================================

def create_model():

    model = LIFSNN(
        num_classes=10
    )

    return model


# ============================================================
# GET CLIENT DATA
# ============================================================

def get_client_data(
    client_id,
    num_clients
):

    (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader
    ) = get_data_loaders(
        batch_size=BATCH_SIZE
    )

    # --------------------------------------------------------
    # FIX: NMNISTDataset builds self.samples by looping over
    # labels 0-9 and appending each class folder's files, then
    # sorts them. That means the underlying dataset is grouped
    # by class (all 0s, then all 1s, ...). Slicing it
    # contiguously (as the old code did) gives each client only
    # 1-2 digit classes -> each client trains a "2-class
    # specialist" that scores 97-99% locally, but FedAvg-ing
    # those specialists together collapses the global model to
    # ~chance level (~10% on 10 classes).
    #
    # Fix: shuffle the indices (deterministically, so every
    # client computes the identical shuffled order) BEFORE
    # slicing, so each client gets a class-balanced, IID-ish
    # subset of the full dataset.
    # --------------------------------------------------------

    total_samples = len(train_dataset)

    generator = torch.Generator().manual_seed(
        PARTITION_SEED
    )

    shuffled_indices = torch.randperm(
        total_samples,
        generator=generator
    ).tolist()

    samples_per_client = (
        total_samples // num_clients
    )

    start = (
        client_id * samples_per_client
    )

    if client_id == num_clients - 1:

        end = total_samples

    else:

        end = (
            start + samples_per_client
        )

    client_indices = shuffled_indices[
        start:end
    ]

    # --------------------------------------------------------
    # Create subset
    # --------------------------------------------------------

    client_dataset = torch.utils.data.Subset(
        train_dataset,
        client_indices
    )

    # --------------------------------------------------------
    # Create DataLoader
    # --------------------------------------------------------

    client_loader = torch.utils.data.DataLoader(
        client_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return client_loader


# ============================================================
# LOCAL TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    client_id
):

    model.to(DEVICE)

    model.train()

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # DIAGNOSTIC: Save initial FC weights
    # --------------------------------------------------------

    initial_fc_weight = (
        model.fc.weight.detach()
        .clone()
    )

    print(
        f"\nClient {client_id + 1} | "
        f"Initial FC weight mean: "
        f"{initial_fc_weight.mean().item():.6f}"
    )

    print(
        f"Client {client_id + 1} | "
        f"Initial FC weight std: "
        f"{initial_fc_weight.std().item():.6f}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    final_loss = 0.0
    final_accuracy = 0.0

    # --------------------------------------------------------
    # Epoch progress bar
    # --------------------------------------------------------

    epoch_bar = tqdm(
        range(1, LOCAL_EPOCHS + 1),
        desc=f"Client {client_id + 1}",
        leave=True
    )

    for epoch in epoch_bar:

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # ----------------------------------------------------
        # Batch training
        # ----------------------------------------------------

        for data, targets in train_loader:

            data = data.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            # ------------------------------------------------
            # Standard LIF forward pass
            #
            # Output:
            # [time, batch, classes]
            # ------------------------------------------------

            output = model(data)

            # ------------------------------------------------
            # Sum spikes across time
            # ------------------------------------------------

            logits = output.sum(
                dim=0
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                logits,
                targets
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            optimizer.step()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            running_loss += (
                loss.item()
                * targets.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)

        # ----------------------------------------------------
        # Epoch results
        # ----------------------------------------------------

        final_loss = (
            running_loss / total
        )

        final_accuracy = (
            100.0 * correct / total
        )

        epoch_bar.set_postfix(
            loss=f"{final_loss:.4f}",
            acc=f"{final_accuracy:.2f}%"
        )

    # ========================================================
    # DIAGNOSTIC: Check whether weights changed
    # ========================================================

    final_fc_weight = (
        model.fc.weight.detach()
    )

    weight_difference = (
        final_fc_weight - initial_fc_weight
    )

    print(
        f"\nClient {client_id + 1} | "
        f"Final FC weight mean: "
        f"{final_fc_weight.mean().item():.6f}"
    )

    print(
        f"Client {client_id + 1} | "
        f"Final FC weight std: "
        f"{final_fc_weight.std().item():.6f}"
    )

    print(
        f"Client {client_id + 1} | "
        f"FC weight change: "
        f"{weight_difference.abs().mean().item():.6f}"
    )

    return final_loss, final_accuracy


# ============================================================
# FLOWER CLIENT APP
# ============================================================

app = ClientApp()


# ============================================================
# CLIENT TRAINING
# ============================================================

@app.train()
def train(
    msg: Message,
    context: Context
):

    # --------------------------------------------------------
    # Client ID
    # --------------------------------------------------------

    client_id = int(
        context.node_config[
            "partition-id"
        ]
    )

    num_clients = int(
        context.node_config[
            "num-partitions"
        ]
    )

    print("\n" + "=" * 60)

    print(
        f"FLOWER CLIENT {client_id + 1}"
    )

    print("=" * 60)

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"Client ID    : {client_id}"
    )

    print(
        f"Total clients: {num_clients}"
    )

    print(
        f"Local epochs : {LOCAL_EPOCHS}"
    )

    print(
        f"Batch size   : {BATCH_SIZE}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model()

    model.to(DEVICE)

    # --------------------------------------------------------
    # Receive global model from server
    # --------------------------------------------------------

    global_state = (
        msg.content[
            "arrays"
        ].to_torch_state_dict()
    )

    model.load_state_dict(
        global_state
    )

    print(
        "\nGlobal model received."
    )

    # --------------------------------------------------------
    # Load client's local data
    # --------------------------------------------------------

    train_loader = get_client_data(
        client_id=client_id,
        num_clients=num_clients
    )

    print(
        f"Local training samples: "
        f"{len(train_loader.dataset)}"
    )

    # --------------------------------------------------------
    # Train locally
    # --------------------------------------------------------

    print(
        "\nStarting local training..."
    )

    final_loss, final_accuracy = (
        train_model(
            model=model,
            train_loader=train_loader,
            client_id=client_id
        )
    )

    # --------------------------------------------------------
    # Training result
    # --------------------------------------------------------

    print(
        f"\nClient {client_id + 1} "
        f"training completed."
    )

    print(
        f"Final local loss     : "
        f"{final_loss:.4f}"
    )

    print(
        f"Final local accuracy : "
        f"{final_accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # Prepare updated model
    # --------------------------------------------------------

    updated_arrays = ArrayRecord(
        model.state_dict()
    )

    # --------------------------------------------------------
    # Prepare metrics
    # --------------------------------------------------------

    metrics = MetricRecord(
        {
            "train_loss": float(
                final_loss
            ),

            "train_accuracy": float(
                final_accuracy
            ),

            "num-examples": int(
                len(train_loader.dataset)
            )
        }
    )

    # --------------------------------------------------------
    # Prepare response
    # --------------------------------------------------------

    content = RecordDict(
        {
            "arrays": updated_arrays,
            "metrics": metrics
        }
    )

    print(
        f"Client {client_id + 1}: "
        f"sending updated model to server."
    )

    return Message(
        content=content,
        reply_to=msg
    )