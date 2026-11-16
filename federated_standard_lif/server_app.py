import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get(
        "FEDSNN_PROJECT_ROOT",
        Path(__file__).resolve().parent
    )
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    MetricRecord,
)

from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg

from model.lif_snn import LIFSNN
from dataset.dataset_loader import get_data_loaders


# ------------------------------------------------------------
# Federated Learning Configuration
# ------------------------------------------------------------
NUM_CLIENTS = 20
NUM_ROUNDS = 50

BATCH_SIZE = 32
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase1"
    / f"federated_standard_lif_{NUM_CLIENTS}clients"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# Output Files
# ------------------------------------------------------------
CSV_FILE = (
    RESULTS_DIR
    / "global_test_results.csv"
)

ACCURACY_PLOT = (
    RESULTS_DIR
    / "global_accuracy_vs_round.png"
)

LOSS_PLOT = (
    RESULTS_DIR
    / "global_loss_vs_round.png"
)

# ------------------------------------------------------------
# Store Global Evaluation Results
# ------------------------------------------------------------
ROUND_RESULTS = []


# ------------------------------------------------------------
# Create Standard LIF model
# ------------------------------------------------------------
def create_model():
    return LIFSNN(num_classes=10)


# ------------------------------------------------------------
# Global Evaluation
# ------------------------------------------------------------
def global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
) -> MetricRecord:

    model = create_model().to(DEVICE)

    # --------------------------------------------------------
    # Load global model
    # --------------------------------------------------------
    state_dict = arrays.to_torch_state_dict()
    model.load_state_dict(state_dict)

    # --------------------------------------------------------
    # DIAGNOSTIC: Check global model weights
    # --------------------------------------------------------
    fc_weight = model.fc.weight.detach()

    print(
        f"\nRound {server_round} | "
        f"FC weight mean: {fc_weight.mean().item():.6f} | "
        f"FC weight std: {fc_weight.std().item():.6f}"
    )

    model.eval()

    # --------------------------------------------------------
    # Load N-MNIST test data
    # --------------------------------------------------------
    _, _, _, test_loader = get_data_loaders(
        batch_size=BATCH_SIZE
    )

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    # --------------------------------------------------------
    # Global test evaluation
    # --------------------------------------------------------
    with torch.no_grad():

        for data, targets in tqdm(
            test_loader,
            desc=f"Global Evaluation - Round {server_round}",
            leave=False
        ):

            data = data.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            # ------------------------------------------------
            # SNN forward pass
            # ------------------------------------------------
            output = model(data)

            # ------------------------------------------------
            # Sum spikes over time
            # ------------------------------------------------
            logits = output.sum(dim=0)

            loss = criterion(
                logits,
                targets
            )

            total_loss += (
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

    # --------------------------------------------------------
    # Calculate global test metrics
    # --------------------------------------------------------
    test_loss = total_loss / total

    test_accuracy = (
        100.0 * correct / total
    )

    print(
        f"\nRound {server_round} "
        f"Global Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print(
        f"Round {server_round} "
        f"Global Test Loss: "
        f"{test_loss:.4f}"
    )

    # --------------------------------------------------------
    # Store Round Results
    # --------------------------------------------------------
    ROUND_RESULTS.append(
        {
            "round": server_round,
            "global_test_accuracy": float(
                test_accuracy
            ),
            "global_test_loss": float(
                test_loss
            ),
        }
    )

    return MetricRecord(
        {
            "global_test_loss": float(test_loss),
            "global_test_accuracy": float(test_accuracy),
        }
    )


# ------------------------------------------------------------
# Save Results as CSV
# ------------------------------------------------------------
def save_results_csv():

    with open(
        CSV_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Round",
                "Global Test Accuracy (%)",
                "Global Test Loss",
            ]
        )

        for result in ROUND_RESULTS:

            writer.writerow(
                [
                    result["round"],
                    f'{result["global_test_accuracy"]:.4f}',
                    f'{result["global_test_loss"]:.6f}',
                ]
            )

    print(
        f"\nCSV results saved to:\n"
        f"{CSV_FILE}"
    )


# ------------------------------------------------------------
# Save Accuracy Plot
# ------------------------------------------------------------
def save_accuracy_plot():

    rounds = [
        result["round"]
        for result in ROUND_RESULTS
    ]

    accuracies = [
        result["global_test_accuracy"]
        for result in ROUND_RESULTS
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        rounds,
        accuracies,
        marker="o"
    )

    plt.xlabel(
        "Communication Round"
    )

    plt.ylabel(
        "Global Test Accuracy (%)"
    )

    plt.title(
        "Global Test Accuracy vs Communication Round"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        ACCURACY_PLOT,
        dpi=300
    )

    plt.close()

    print(
        f"Accuracy plot saved to:\n"
        f"{ACCURACY_PLOT}"
    )


# ------------------------------------------------------------
# Save Loss Plot
# ------------------------------------------------------------
def save_loss_plot():

    rounds = [
        result["round"]
        for result in ROUND_RESULTS
    ]

    losses = [
        result["global_test_loss"]
        for result in ROUND_RESULTS
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        rounds,
        losses,
        marker="o"
    )

    plt.xlabel(
        "Communication Round"
    )

    plt.ylabel(
        "Global Test Loss"
    )

    plt.title(
        "Global Test Loss vs Communication Round"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        LOSS_PLOT,
        dpi=300
    )

    plt.close()

    print(
        f"Loss plot saved to:\n"
        f"{LOSS_PLOT}"
    )


# ------------------------------------------------------------
# Flower Server Application
# ------------------------------------------------------------
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):

    print("\n" + "=" * 60)
    print("STANDARD LIF - FEDERATED LEARNING SERVER")
    print("=" * 60)

    print(f"Device       : {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(f"Clients      : {NUM_CLIENTS}")
    print(f"Rounds       : {NUM_ROUNDS}")
    print(f"Local Epochs : {LOCAL_EPOCHS}")
    print(f"Batch Size   : {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")

    print("=" * 60)

    # --------------------------------------------------------
    # Create initial global model
    # --------------------------------------------------------
    print("\nCreating initial global Standard LIF model...")

    global_model = create_model()

    print(
        f"Model name   : "
        f"{global_model.__class__.__name__}"
    )

    print(
        f"Parameters   : "
        f"{sum(p.numel() for p in global_model.parameters())}"
    )

    initial_arrays = ArrayRecord(
        global_model.state_dict()
    )

    # --------------------------------------------------------
    # FedAvg Strategy
    # --------------------------------------------------------
    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=NUM_CLIENTS,
        min_available_nodes=NUM_CLIENTS,
        weighted_by_key="num-examples",
    )

    print("\nFedAvg strategy created.")

    print(
        f"Training fraction : "
        f"{strategy.fraction_train}"
    )

    print(
        f"Minimum clients   : "
        f"{strategy.min_train_nodes}"
    )

    print(
        "\nStarting federated training..."
    )

    # --------------------------------------------------------
    # Start FedAvg
    # --------------------------------------------------------
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord(
            {
                "local_epochs": LOCAL_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
            }
        ),
        num_rounds=NUM_ROUNDS,
        evaluate_fn=global_evaluate,
    )

    # --------------------------------------------------------
    # Save CSV and Plots
    # --------------------------------------------------------
    save_results_csv()

    save_accuracy_plot()

    save_loss_plot()

    # --------------------------------------------------------
    # Save final global model
    # --------------------------------------------------------
    final_model = create_model()

    final_model.load_state_dict(
        result.arrays.to_torch_state_dict()
    )

    final_model_path = (
        RESULTS_DIR
        / "final_global_standard_lif.pth"
    )

    torch.save(
        final_model.state_dict(),
        final_model_path
    )

    print("\n" + "=" * 60)
    print("FEDERATED TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Final model saved to:\n"
        f"{final_model_path}"
    )

    print("=" * 60)


# ------------------------------------------------------------
# Direct execution protection
# ------------------------------------------------------------
if __name__ == "__main__":
    print(
        "This file is a Flower ServerApp."
    )

    print(
        "Run it using:"
    )

    print(
        "    flwr run . --stream"
    )