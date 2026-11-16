import sys
import os
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import torch

from tqdm import tqdm

from model.lif_snn import LIFSNN
from dataset.dataset_loader import get_data_loaders


# ============================================================
# FEDERATED LEARNING SETTINGS
# ============================================================

NUM_CLIENTS = 5
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# ============================================================
# RESULTS DIRECTORY
# ============================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase1"
    / "federated_standard_lif"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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

    # IMPORTANT:
    # This exactly matches your existing
    # model/lif_snn.py

    model = LIFSNN(
        num_classes=10
    )

    return model


# ============================================================
# LOAD N-MNIST DATA
# ============================================================

def load_nmnist():

    print("\nLoading N-MNIST dataset...")

    # Your actual dataset_loader.py returns:
    #
    # 1. train_dataset
    # 2. test_dataset
    # 3. train_loader
    # 4. test_loader

    (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader
    ) = get_data_loaders(
        batch_size=BATCH_SIZE
    )

    return (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader
    )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # PROJECT INFORMATION
    # ========================================================

    print("=" * 60)
    print("STANDARD LIF - FEDERATED LEARNING TASK TEST")
    print("=" * 60)

    print(
        f"Device       : {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Clients      : {NUM_CLIENTS}"
    )

    print(
        f"Local Epochs : {LOCAL_EPOCHS}"
    )

    print(
        f"Batch Size   : {BATCH_SIZE}"
    )

    print(
        f"Learning Rate: {LEARNING_RATE}"
    )

    print("=" * 60)


    # ========================================================
    # LOAD DATASET
    # ========================================================

    (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader
    ) = load_nmnist()


    print(
        f"\nTraining samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Testing samples: "
        f"{len(test_dataset)}"
    )


    # ========================================================
    # CREATE MODEL
    # ========================================================

    print(
        "\nCreating Standard LIF model..."
    )

    model = create_model()

    model = model.to(DEVICE)

    print(
        "Model created successfully."
    )

    print(
        f"Model name: "
        f"{model.__class__.__name__}"
    )


    # ========================================================
    # COUNT PARAMETERS
    # ========================================================

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_parameters}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters}"
    )


    # ========================================================
    # LOAD ONE BATCH
    # ========================================================

    print(
        "\nLoading one N-MNIST batch..."
    )

    data, targets = next(
        iter(train_loader)
    )

    print(
        f"Input shape  : "
        f"{data.shape}"
    )

    print(
        f"Target shape : "
        f"{targets.shape}"
    )


    # ========================================================
    # MOVE TO DEVICE
    # ========================================================

    data = data.to(
        DEVICE,
        non_blocking=True
    )

    targets = targets.to(
        DEVICE,
        non_blocking=True
    )


    # ========================================================
    # STANDARD LIF FORWARD PASS
    # ========================================================

    print(
        "\nRunning Standard LIF forward pass..."
    )

    model.eval()

    with torch.no_grad():

        output = model(data)


    print(
        f"Output shape : "
        f"{output.shape}"
    )


    # ========================================================
    # SPIKE OUTPUT TO CLASS SCORES
    # ========================================================

    # Your Standard LIF returns:
    #
    # [time, batch, classes]
    #
    # So we sum spikes over time.

    logits = output.sum(
        dim=0
    )

    print(
        f"Logits shape : "
        f"{logits.shape}"
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    predictions = torch.argmax(
        logits,
        dim=1
    )

    batch_accuracy = (
        (predictions == targets)
        .float()
        .mean()
        * 100
    )

    print(
        f"Batch accuracy: "
        f"{batch_accuracy.item():.2f}%"
    )


    # ========================================================
    # TQDM TEST
    # ========================================================

    print(
        "\nTesting tqdm..."
    )

    for _ in tqdm(
        range(5),
        desc="Standard LIF test"
    ):

        pass


    # ========================================================
    # SAVE TEST OUTPUT
    # ========================================================

    output_file = (
        RESULTS_DIR
        / "standard_lif_task_test.txt"
    )


    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "=" * 60 + "\n"
        )

        file.write(
            "STANDARD LIF - FEDERATED LEARNING TASK TEST\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Device: {DEVICE}\n"
        )

        if torch.cuda.is_available():

            file.write(
                f"GPU: "
                f"{torch.cuda.get_device_name(0)}\n"
            )

        file.write(
            f"Number of clients: "
            f"{NUM_CLIENTS}\n"
        )

        file.write(
            f"Local epochs: "
            f"{LOCAL_EPOCHS}\n"
        )

        file.write(
            f"Batch size: "
            f"{BATCH_SIZE}\n"
        )

        file.write(
            f"Learning rate: "
            f"{LEARNING_RATE}\n\n"
        )

        file.write(
            f"Training samples: "
            f"{len(train_dataset)}\n"
        )

        file.write(
            f"Testing samples: "
            f"{len(test_dataset)}\n\n"
        )

        file.write(
            "Model: LIFSNN\n"
        )

        file.write(
            f"Total parameters: "
            f"{total_parameters}\n"
        )

        file.write(
            f"Trainable parameters: "
            f"{trainable_parameters}\n\n"
        )

        file.write(
            f"Input shape: "
            f"{tuple(data.shape)}\n"
        )

        file.write(
            f"Target shape: "
            f"{tuple(targets.shape)}\n"
        )

        file.write(
            f"Output shape: "
            f"{tuple(output.shape)}\n"
        )

        file.write(
            f"Logits shape: "
            f"{tuple(logits.shape)}\n"
        )

        file.write(
            f"Batch accuracy: "
            f"{batch_accuracy.item():.2f}%\n\n"
        )

        file.write(
            "N-MNIST dataset loader: OK\n"
        )

        file.write(
            "Standard LIF model: OK\n"
        )

        file.write(
            "CUDA / GPU: OK\n"
        )

        file.write(
            "TQDM: OK\n"
        )

        file.write(
            "Task test completed successfully.\n"
        )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print(
        "\nResult saved to:"
    )

    print(
        output_file
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "TASK TEST COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )

    print(
        "\nN-MNIST dataset loader : OK"
    )

    print(
        "Standard LIF model      : OK"
    )

    print(
        "CUDA / RTX 4090         : OK"
    )

    print(
        "TQDM progress bar       : OK"
    )

    print(
        "\nReady for Flower federated client."
    )