import os
import sys
import csv
import statistics
from pathlib import Path

import numpy as np
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
#
# MUST MATCH client_app.py's NUM_CLIENTS exactly.
# This is the ONE value to change (5 -> 10 -> 15 -> 20).
# ============================================================

NUM_CLIENTS = 5

BATCH_SIZE = 32
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.001

NONIID_ALPHA = 0.1   # Severe Non-IID (must match client_app.py)

# Number of final rounds used to compute the "Stability" metric
# (standard deviation of global test accuracy over this window).
STABILITY_WINDOW = 10

# N-MNIST has 10 digit classes (used for the confusion matrix
# behind precision / recall / F1).
NUM_CLASSES = 10


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# RESULTS DIRECTORY
#
# Segmented by client count, so results from clients=5 and a
# later clients=10 run never overwrite each other - exactly
# the same pattern used for delta_X / alpha_X in earlier phases.
# ============================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase5_severenoniid_fedavg"
    / f"clients_{NUM_CLIENTS}"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = RESULTS_DIR / "severenoniid_results.csv"
ACCURACY_PLOT = RESULTS_DIR / "severenoniid_accuracy_vs_round.png"
LOSS_PLOT = RESULTS_DIR / "severenoniid_loss_vs_round.png"
PRF_PLOT = RESULTS_DIR / "severenoniid_precision_recall_f1_vs_round.png"
SUMMARY_FILE = RESULTS_DIR / "severenoniid_summary.txt"


# ============================================================
# RESULT STORAGE
# ============================================================

ROUND_RESULTS = []


# ============================================================
# SERVER INFORMATION
# ============================================================

print("=" * 70)
print("EXPERIMENT 6 - SEVERE NON-IID FEDAVG SERVER")
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
print(f"Non-IID alpha: {NONIID_ALPHA} (severe)")
print(f"Results dir  : {RESULTS_DIR}")

print("=" * 70)


# ============================================================
# GLOBAL TEST DATASET
# ============================================================

print()
print("Loading N-MNIST test dataset...")

NMNIST_ROOT = PROJECT_ROOT / "data" / "nmnist" / "N-MNIST"

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
# MACRO PRECISION / RECALL / F1 (from a confusion matrix)
#
# confusion[true_class, predicted_class] = count
#
#   precision_c = TP_c / (all samples predicted as class c)
#   recall_c    = TP_c / (all samples truly in class c)
#   f1_c        = 2 * P_c * R_c / (P_c + R_c)
#
# "Macro" = unweighted mean over the 10 classes, so every digit
# counts equally. This matters under Non-IID: a model that is
# strong on common classes but weak on rare ones can still show
# high accuracy, while macro recall/F1 expose the weak classes.
# Classes with zero predictions / zero samples score 0 for that
# term instead of raising a divide-by-zero.
# ============================================================

def compute_macro_prf(confusion):

    diag = np.diag(confusion).astype(np.float64)
    predicted_per_class = confusion.sum(axis=0).astype(np.float64)
    true_per_class = confusion.sum(axis=1).astype(np.float64)

    precision = np.divide(
        diag, predicted_per_class,
        out=np.zeros_like(diag), where=predicted_per_class > 0,
    )

    recall = np.divide(
        diag, true_per_class,
        out=np.zeros_like(diag), where=true_per_class > 0,
    )

    denominator = precision + recall

    f1 = np.divide(
        2 * precision * recall, denominator,
        out=np.zeros_like(diag), where=denominator > 0,
    )

    return (
        float(precision.mean()) * 100.0,
        float(recall.mean()) * 100.0,
        float(f1.mean()) * 100.0,
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

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

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

            np.add.at(
                confusion,
                (labels.cpu().numpy(), predictions.cpu().numpy()),
                1,
            )

    if total == 0:
        raise RuntimeError("Global evaluation received zero test samples.")

    test_loss = total_loss / total
    test_accuracy = 100.0 * correct / total

    test_precision, test_recall, test_f1 = compute_macro_prf(confusion)

    ROUND_RESULTS.append(
        {
            "round": server_round,
            "global_test_accuracy": float(test_accuracy),
            "global_test_loss": float(test_loss),
            "global_test_precision": test_precision,
            "global_test_recall": test_recall,
            "global_test_f1": test_f1,
        }
    )

    print()
    print(
        f"Round {server_round} "
        f"Global Test Accuracy: {test_accuracy:.2f}% "
        f"| Global Test Loss: {test_loss:.4f}"
    )
    print(
        f"Round {server_round} "
        f"Precision: {test_precision:.2f}% "
        f"| Recall: {test_recall:.2f}% "
        f"| F1: {test_f1:.2f}%"
    )

    return MetricRecord(
        {
            "global_test_accuracy": float(test_accuracy),
            "global_test_loss": float(test_loss),
            "global_test_precision": test_precision,
            "global_test_recall": test_recall,
            "global_test_f1": test_f1,
        }
    )


# ============================================================
# SERVER APP (plain FedAvg - no custom aggregation override)
# ============================================================

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):

    num_rounds = int(context.run_config.get("num-server-rounds", 50))

    print()
    print(f"Configured communication rounds: {num_rounds}")

    ROUND_RESULTS.clear()

    strategy = FedAvg(
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
            [
                "Round",
                "Global Test Accuracy (%)",
                "Global Test Loss",
                "Macro Precision (%)",
                "Macro Recall (%)",
                "Macro F1 (%)",
            ]
        )
        for row in ROUND_RESULTS:
            writer.writerow(
                [
                    row["round"],
                    f'{row["global_test_accuracy"]:.4f}',
                    f'{row["global_test_loss"]:.6f}',
                    f'{row["global_test_precision"]:.4f}',
                    f'{row["global_test_recall"]:.4f}',
                    f'{row["global_test_f1"]:.4f}',
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
        plt.title(
            f"Severe Non-IID FedAvg ({NUM_CLIENTS} clients) "
            f"- Global Test Accuracy"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(ACCURACY_PLOT, dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(rounds, losses, marker="o", markersize=3)
        plt.xlabel("Communication Round")
        plt.ylabel("Global Test Loss")
        plt.title(
            f"Severe Non-IID FedAvg ({NUM_CLIENTS} clients) "
            f"- Global Test Loss"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(LOSS_PLOT, dpi=300)
        plt.close()

        precisions = [row["global_test_precision"] for row in ROUND_RESULTS]
        recalls = [row["global_test_recall"] for row in ROUND_RESULTS]
        f1_scores = [row["global_test_f1"] for row in ROUND_RESULTS]

        plt.figure(figsize=(8, 5))
        plt.plot(rounds, precisions, marker="o", markersize=3, label="Precision")
        plt.plot(rounds, recalls, marker="s", markersize=3, label="Recall")
        plt.plot(rounds, f1_scores, marker="^", markersize=3, label="F1")
        plt.xlabel("Communication Round")
        plt.ylabel("Macro Score (%)")
        plt.title(
            f"Severe Non-IID FedAvg ({NUM_CLIENTS} clients) "
            f"- Macro Precision / Recall / F1"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PRF_PLOT, dpi=300)
        plt.close()

    # ========================================================
    # STABILITY METRIC (Experiment 6 required metric)
    #
    # Standard deviation of global test accuracy over the last
    # STABILITY_WINDOW rounds - lower = more stable convergence.
    # ========================================================

    if len(accuracies) >= 2:

        window = accuracies[-STABILITY_WINDOW:]

        if len(window) >= 2:
            stability_std = statistics.stdev(window)
        else:
            stability_std = 0.0

    else:
        stability_std = None

    # Under severe non-IID the curve can swing from round to round,
    # so the final round alone may not represent the run. Report the
    # peak and the mean over the stability window as well.
    if len(accuracies) > 0:
        peak_accuracy = max(accuracies)
        peak_round = rounds[accuracies.index(peak_accuracy)]
        window_mean_accuracy = statistics.mean(accuracies[-STABILITY_WINDOW:])
    else:
        peak_accuracy = None
        peak_round = None
        window_mean_accuracy = None

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    with open(SUMMARY_FILE, "w") as file:
        file.write("EXPERIMENT 6 - SEVERE NON-IID FEDAVG\n")
        file.write("=" * 60 + "\n")
        file.write(f"Clients: {NUM_CLIENTS}\n")
        file.write(f"Non-IID alpha: {NONIID_ALPHA} (severe)\n")
        file.write(f"Communication rounds: {num_rounds}\n")

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
            file.write(
                f"Final macro precision: "
                f'{final_result["global_test_precision"]:.4f}%\n'
            )
            file.write(
                f"Final macro recall: "
                f'{final_result["global_test_recall"]:.4f}%\n'
            )
            file.write(
                f"Final macro F1: "
                f'{final_result["global_test_f1"]:.4f}%\n'
            )

        if peak_accuracy is not None:
            file.write(
                f"Peak global test accuracy: "
                f"{peak_accuracy:.4f}% (round {peak_round})\n"
            )
            file.write(
                f"Mean accuracy over last "
                f"{min(STABILITY_WINDOW, len(accuracies))} rounds: "
                f"{window_mean_accuracy:.4f}%\n"
            )

        if stability_std is not None:
            file.write(
                f"Stability (std dev of accuracy, "
                f"last {min(STABILITY_WINDOW, len(accuracies))} rounds): "
                f"{stability_std:.4f}\n"
            )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    if result is not None and result.arrays is not None:
        try:
            final_state_dict = result.arrays.to_torch_state_dict()
            final_model_file = RESULTS_DIR / "final_severenoniid_lif.pth"
            torch.save(final_state_dict, final_model_file)

            print()
            print("Final global model saved to:")
            print(final_model_file)

        except Exception as exc:
            print("Warning: final model could not be saved.")
            print(exc)
    else:
        print()
        print("Warning: no final aggregated model available.")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(f"EXPERIMENT 6 (clients={NUM_CLIENTS}) COMPLETED")
    print("=" * 70)
    print(f"Clients                  : {NUM_CLIENTS}")
    print(f"Non-IID alpha            : {NONIID_ALPHA} (severe)")
    print(f"Communication rounds     : {num_rounds}")

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
        print(
            f"Final macro precision    : "
            f'{final_result["global_test_precision"]:.2f}%'
        )
        print(
            f"Final macro recall       : "
            f'{final_result["global_test_recall"]:.2f}%'
        )
        print(
            f"Final macro F1           : "
            f'{final_result["global_test_f1"]:.2f}%'
        )

    if peak_accuracy is not None:
        print(
            f"Peak test accuracy       : "
            f"{peak_accuracy:.2f}% (round {peak_round})"
        )
        print(
            f"Mean accuracy (last {min(STABILITY_WINDOW, len(accuracies))}) : "
            f"{window_mean_accuracy:.2f}%"
        )

    if stability_std is not None:
        print(f"Stability (accuracy std) : {stability_std:.4f}")

    print()
    print("Results saved in:")
    print(RESULTS_DIR)
    print("=" * 70)