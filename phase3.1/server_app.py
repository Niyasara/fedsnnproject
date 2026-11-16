import os
import sys
import csv
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from flwr.serverapp import ServerApp, Grid
from flwr.common import Context, ArrayRecord, MetricRecord
from flwr.serverapp.strategy import FedAvg


# ============================================================
# PROJECT PATH
# ============================================================

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

# Must match client_app.py's ALPHA
ALPHA = 1.0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# RESULTS DIRECTORY
# ============================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase3.1_adaptive"
    / f"alpha_{ALPHA}"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = RESULTS_DIR / "adaptive_trigger_results.csv"
ACCURACY_PLOT = RESULTS_DIR / "adaptive_trigger_accuracy_vs_round.png"
LOSS_PLOT = RESULTS_DIR / "adaptive_trigger_loss_vs_round.png"
SUMMARY_FILE = RESULTS_DIR / "adaptive_trigger_summary.txt"


# ============================================================
# RESULT STORAGE
# ============================================================

ROUND_RESULTS = []
TOTAL_UPLOADS = 0
TOTAL_CLIENT_DECISIONS = 0


# ============================================================
# SERVER INFORMATION
# ============================================================

print("=" * 70)
print("PHASE 4 (phase3.1) - ADAPTIVE EVENT-TRIGGERED FEDERATED SNN")
print("=" * 70)

print(f"Device       : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
else:
    print("GPU          : CPU")

print(f"Clients      : {NUM_CLIENTS}")
print("Rounds       : Read from Flower configuration")
print(f"Local epochs : {LOCAL_EPOCHS}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Alpha        : {ALPHA}")

print("=" * 70)


# ============================================================
# GLOBAL TEST DATASET
# ============================================================

print()
print("Loading N-MNIST test dataset...")

NMNIST_ROOT = PROJECT_ROOT / "data" / "nmnist" / "N-MNIST"

print(f"Test dataset path: {NMNIST_ROOT}")

if not NMNIST_ROOT.exists():
    raise FileNotFoundError(
        f"N-MNIST dataset path does not exist:\n{NMNIST_ROOT}"
    )

test_dataset = NMNISTDataset(root=str(NMNIST_ROOT), train=False)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

print(f"Testing samples: {len(test_dataset)}")

if len(test_dataset) == 0:
    raise RuntimeError(
        "N-MNIST test dataset contains 0 samples. "
        "Check the dataset path and folder structure."
    )


# ============================================================
# GLOBAL EVALUATION
# ============================================================

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:

    model = LIFSNN().to(DEVICE)
    model.load_state_dict(arrays.to_torch_state_dict())
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        test_loader,
        desc=f"Round {server_round} Evaluation",
        leave=False,
        unit="batch",
    )

    with torch.no_grad():
        for frames, labels in progress:

            frames = frames.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(frames)
            outputs = outputs.sum(dim=0)

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    if total == 0:
        raise RuntimeError("Global evaluation received zero test samples.")

    test_loss = total_loss / total
    test_accuracy = 100.0 * correct / total

    ROUND_RESULTS.append(
        {
            "round": server_round,
            "global_test_accuracy": float(test_accuracy),
            "global_test_loss": float(test_loss),
        }
    )

    print()
    print(
        f"Round {server_round} "
        f"Global Test Accuracy: {test_accuracy:.2f}% "
        f"| Global Test Loss: {test_loss:.4f}"
    )

    return MetricRecord(
        {
            "global_test_accuracy": float(test_accuracy),
            "global_test_loss": float(test_loss),
        }
    )


# ============================================================
# EVENT-TRIGGER METRIC AGGREGATION
# ============================================================

def aggregate_trigger_metrics(results):

    global TOTAL_UPLOADS
    global TOTAL_CLIENT_DECISIONS

    uploads = 0
    decisions = 0
    adaptive_deltas_seen = []

    for message in results:

        if message.has_error():
            print(
                "WARNING: Client reply contains an error: "
                f"{message.error.reason}"
            )
            continue

        if not message.has_content():
            print("WARNING: Client reply has no content. Ignoring.")
            continue

        record = message.content
        decisions += 1

        metrics = record.get("metrics")
        if metrics is not None and "adaptive_delta" in metrics:
            adaptive_deltas_seen.append(float(metrics["adaptive_delta"]))

        if "arrays" in record:
            uploads += 1
            print("DEBUG: Client uploaded model (arrays found)")
        else:
            print("DEBUG: Client skipped model (no arrays)")

    TOTAL_UPLOADS += uploads
    TOTAL_CLIENT_DECISIONS += decisions

    if adaptive_deltas_seen:
        avg_delta = sum(adaptive_deltas_seen) / len(adaptive_deltas_seen)
        print(f"DEBUG: Average adaptive delta_i(t) this round: {avg_delta:.6f}")

    return uploads, decisions


# ============================================================
# CUSTOM ADAPTIVE EVENT-TRIGGERED FEDAVG
# ============================================================

class AdaptiveEventTriggeredFedAvg(FedAvg):

    def aggregate_train(self, server_round, results):

        results = list(results)
        print(f"DEBUG: aggregate_train called with {len(results)} result(s)")

        uploads, decisions = aggregate_trigger_metrics(results)

        print()
        print("=" * 70)
        print(f"ROUND {server_round} ADAPTIVE EVENT-TRIGGER STATISTICS")
        print("=" * 70)
        print(f"Client decisions : {decisions}")
        print(f"Model uploads    : {uploads}")

        participation = (
            100.0 * uploads / decisions if decisions > 0 else 0.0
        )
        print(f"Upload frequency : {participation:.2f}%")

        valid_results = []

        for message in results:

            if message.has_error() or not message.has_content():
                continue

            if "arrays" in message.content:
                valid_results.append(message)

        print(f"Updates aggregated: {len(valid_results)}")

        if len(valid_results) == 0:
            print("WARNING: No client uploaded a model update in this round.")
            return None, None

        return super().aggregate_train(server_round, valid_results)


# ============================================================
# SERVER APP
# ============================================================

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):

    global TOTAL_UPLOADS
    global TOTAL_CLIENT_DECISIONS

    num_rounds = int(context.run_config.get("num-server-rounds", 50))

    print()
    print(f"Configured communication rounds: {num_rounds}")

    TOTAL_UPLOADS = 0
    TOTAL_CLIENT_DECISIONS = 0
    ROUND_RESULTS.clear()

    strategy = AdaptiveEventTriggeredFedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=NUM_CLIENTS,
        min_available_nodes=NUM_CLIENTS,
        weighted_by_key="num-examples",
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(LIFSNN().state_dict()),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
        timeout=10800,
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Round", "Global Test Accuracy (%)", "Global Test Loss"]
        )
        for row in ROUND_RESULTS:
            writer.writerow(
                [
                    row["round"],
                    f'{row["global_test_accuracy"]:.4f}',
                    f'{row["global_test_loss"]:.6f}',
                ]
            )

    # ========================================================
    # PLOTS
    # ========================================================

    rounds = [row["round"] for row in ROUND_RESULTS]
    accuracies = [row["global_test_accuracy"] for row in ROUND_RESULTS]
    losses = [row["global_test_loss"] for row in ROUND_RESULTS]

    if len(rounds) > 0:

        plt.figure(figsize=(8, 5))
        plt.plot(rounds, accuracies, marker="o", markersize=3)
        plt.xlabel("Communication Round")
        plt.ylabel("Global Test Accuracy (%)")
        plt.title("Adaptive Event-Triggered Federated SNN - Global Test Accuracy")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(ACCURACY_PLOT, dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(rounds, losses, marker="o", markersize=3)
        plt.xlabel("Communication Round")
        plt.ylabel("Global Test Loss")
        plt.title("Adaptive Event-Triggered Federated SNN - Global Test Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(LOSS_PLOT, dpi=300)
        plt.close()

    # ========================================================
    # COMMUNICATION STATISTICS
    # ========================================================

    baseline_uploads = NUM_CLIENTS * num_rounds
    event_uploads = TOTAL_UPLOADS

    communication_reduction = (
        100.0 * (baseline_uploads - event_uploads) / baseline_uploads
        if baseline_uploads > 0 else 0.0
    )

    overall_upload_frequency = (
        100.0 * event_uploads / TOTAL_CLIENT_DECISIONS
        if TOTAL_CLIENT_DECISIONS > 0 else 0.0
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    with open(SUMMARY_FILE, "w") as file:
        file.write("PHASE 4 (phase3.1) - ADAPTIVE EVENT-TRIGGERED FEDERATED SNN\n")
        file.write("=" * 60 + "\n")
        file.write(f"Alpha: {ALPHA}\n")
        file.write(f"Clients: {NUM_CLIENTS}\n")
        file.write(f"Communication rounds: {num_rounds}\n")
        file.write(f"Baseline uploads: {baseline_uploads}\n")
        file.write(f"Event-triggered uploads: {event_uploads}\n")
        file.write(f"Overall upload frequency: {overall_upload_frequency:.2f}%\n")
        file.write(f"Communication reduction: {communication_reduction:.2f}%\n")

        if len(ROUND_RESULTS) > 0:
            final_result = ROUND_RESULTS[-1]
            file.write(
                f"Final global test accuracy: "
                f'{final_result["global_test_accuracy"]:.4f}%\n'
            )
            file.write(
                f"Final global test loss: "
                f'{final_result["global_test_loss"]:.6f}\n'
            )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    if result is not None and result.arrays is not None:

        try:
            final_state_dict = result.arrays.to_torch_state_dict()
            final_model_file = RESULTS_DIR / "final_adaptive_event_triggered_lif.pth"
            torch.save(final_state_dict, final_model_file)

            print()
            print("Final global model saved to:")
            print(final_model_file)

        except Exception as exc:
            print("Warning: final model could not be saved.")
            print(exc)

    else:
        print()
        print(
            "Warning: no final aggregated model available "
            "(no round produced valid uploads)."
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("PHASE 4 (phase3.1) COMPLETED")
    print("=" * 70)
    print(f"Alpha                    : {ALPHA}")
    print(f"Communication rounds     : {num_rounds}")
    print(f"Baseline uploads         : {baseline_uploads}")
    print(f"Event-trigger uploads    : {event_uploads}")
    print(f"Upload frequency         : {overall_upload_frequency:.2f}%")
    print(f"Communication reduction  : {communication_reduction:.2f}%")

    if len(ROUND_RESULTS) > 0:
        final_result = ROUND_RESULTS[-1]
        print(
            f"Final test accuracy      : "
            f'{final_result["global_test_accuracy"]:.2f}%'
        )
        print(
            f"Final test loss          : "
            f'{final_result["global_test_loss"]:.4f}'
        )

    print()
    print("Results saved in:")
    print(RESULTS_DIR)
    print("=" * 70)