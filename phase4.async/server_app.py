import os
import sys
import csv
import copy
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

# ------------------------------------------------------------
# Phase 5 (folder: phase4.async) - Asynchronous Aggregation
#
# Aggregation Rule (per your document):
#     W_g(t+1) = W_g(t) + ETA * (W_i(t) - W_g(t))
#
# Applied INCREMENTALLY, one client update at a time, instead
# of FedAvg's single weighted average over all clients at once.
# This is the core architectural difference from Phase 3/3.1.
#
# ETA is the aggregation step size - how much of each client's
# update gets folded into the global model per application.
# ------------------------------------------------------------

ETA = 0.3

# ------------------------------------------------------------
# min_train_nodes lowered below NUM_CLIENTS: the strategy no
# longer requires every client to reply before proceeding.
# (Honest caveat: Flower's simulation engine still dispatches
# training to all sampled clients and waits up to `timeout`
# for replies each round - true continuous, non-blocking
# asynchrony across round boundaries isn't offered by
# strategy.start()'s round-based loop. What this DOES achieve:
# a round can complete and aggregate successfully even if some
# clients fail/time out, rather than discarding the whole
# round the way Phase 3's min_train_nodes=NUM_CLIENTS did.)
# ------------------------------------------------------------

MIN_TRAIN_NODES = 1


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
    / "phase4_async"
    / f"eta_{ETA}"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = RESULTS_DIR / "async_results.csv"
ACCURACY_PLOT = RESULTS_DIR / "async_accuracy_vs_round.png"
LOSS_PLOT = RESULTS_DIR / "async_loss_vs_round.png"
SUMMARY_FILE = RESULTS_DIR / "async_summary.txt"


# ============================================================
# RESULT STORAGE
# ============================================================

ROUND_RESULTS = []
TOTAL_UPDATES_APPLIED = 0
TOTAL_CLIENT_REPLIES = 0

# We track the global model's state_dict ourselves, rather than
# relying on any internal Flower strategy attribute (which isn't
# a documented/verified API). This is initialized right before
# strategy.start() is called, and kept in sync every round.
GLOBAL_STATE_DICT = None


# ============================================================
# SERVER INFORMATION
# ============================================================

print("=" * 70)
print("PHASE 5 (phase4.async) - ASYNCHRONOUS AGGREGATION SERVER")
print("=" * 70)

print(f"Device       : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
else:
    print("GPU          : CPU")

print(f"Clients          : {NUM_CLIENTS}")
print("Rounds           : Read from Flower configuration")
print(f"Local epochs     : {LOCAL_EPOCHS}")
print(f"Batch size       : {BATCH_SIZE}")
print(f"Learning rate    : {LEARNING_RATE}")
print(f"Eta (agg. weight): {ETA}")
print(f"Min train nodes  : {MIN_TRAIN_NODES} (not requiring all {NUM_CLIENTS})")

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
# ASYNCHRONOUS-STYLE INCREMENTAL AGGREGATION
#
#   W_g(t+1) = W_g(t) + ETA * (W_i(t) - W_g(t))
#
# Applied once PER CLIENT REPLY, in sequence, instead of a
# single FedAvg weighted average over all replies at once.
# Each client's update nudges the global model a bit toward
# itself; later clients in the sequence are nudging an
# already-partially-updated global model, not the original.
# ============================================================

class AsyncIncrementalFedAvg(FedAvg):

    def aggregate_train(self, server_round, results):

        global TOTAL_UPDATES_APPLIED
        global TOTAL_CLIENT_REPLIES
        global GLOBAL_STATE_DICT

        results = list(results)
        print(f"DEBUG: aggregate_train called with {len(results)} result(s)")

        valid_results = []

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

            if "arrays" in message.content:
                valid_results.append(message)

        TOTAL_CLIENT_REPLIES += len(results)

        print()
        print("=" * 70)
        print(f"ROUND {server_round} ASYNC AGGREGATION STATISTICS")
        print("=" * 70)
        print(f"Client replies received : {len(results)}")
        print(f"Valid updates to apply  : {len(valid_results)}")

        if len(valid_results) == 0:
            print("WARNING: No valid client updates this round.")
            return None, None

        # --------------------------------------------------
        # Use OUR OWN tracked global state_dict (set up before
        # strategy.start() and kept in sync every round below),
        # so we can apply the incremental update rule directly
        # in tensor space without depending on any internal,
        # unverified Flower strategy attribute.
        # --------------------------------------------------

        original_dtypes = {k: v.dtype for k, v in GLOBAL_STATE_DICT.items()}
        global_state = {
            k: v.clone().float() for k, v in GLOBAL_STATE_DICT.items()
        }

        applied_count = 0

        for message in valid_results:

            client_arrays = message.content["arrays"]
            client_state = client_arrays.to_torch_state_dict()

            # W_g(t+1) = W_g(t) + ETA * (W_i(t) - W_g(t))
            for key in global_state.keys():
                client_val = client_state[key].float()
                global_state[key] = (
                    global_state[key]
                    + ETA * (client_val - global_state[key])
                )

            applied_count += 1
            TOTAL_UPDATES_APPLIED += 1

            print(f"  Applied incremental update #{applied_count} (eta={ETA})")

        # --------------------------------------------------
        # Cast back to the original dtypes before wrapping
        # as an ArrayRecord, and build an aggregated metrics
        # summary across the applied updates. Also update our
        # own tracked copy so next round starts from here.
        # --------------------------------------------------

        for key in global_state.keys():
            global_state[key] = global_state[key].to(original_dtypes[key])

        GLOBAL_STATE_DICT = {k: v.clone() for k, v in global_state.items()}

        new_arrays = ArrayRecord(global_state)

        avg_train_loss = sum(
            float(m.content["metrics"]["train_loss"]) for m in valid_results
        ) / len(valid_results)

        avg_train_accuracy = sum(
            float(m.content["metrics"]["train_accuracy"]) for m in valid_results
        ) / len(valid_results)

        aggregated_metrics = MetricRecord(
            {
                "train_loss": avg_train_loss,
                "train_accuracy": avg_train_accuracy,
            }
        )

        print(f"Updates applied this round: {applied_count}")

        return new_arrays, aggregated_metrics


# ============================================================
# SERVER APP
# ============================================================

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):

    global TOTAL_UPDATES_APPLIED
    global TOTAL_CLIENT_REPLIES
    global GLOBAL_STATE_DICT

    num_rounds = int(context.run_config.get("num-server-rounds", 50))

    print()
    print(f"Configured communication rounds: {num_rounds}")

    TOTAL_UPDATES_APPLIED = 0
    TOTAL_CLIENT_REPLIES = 0
    ROUND_RESULTS.clear()

    initial_model = LIFSNN()

    # Initialize our tracked global state_dict BEFORE training
    # starts, matching the same initial weights passed to
    # strategy.start() below.
    GLOBAL_STATE_DICT = {
        k: v.clone() for k, v in initial_model.state_dict().items()
    }

    strategy = AsyncIncrementalFedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=MIN_TRAIN_NODES,
        min_available_nodes=MIN_TRAIN_NODES,
        weighted_by_key="num-examples",
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(initial_model.state_dict()),
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
        plt.title("Asynchronous Aggregation - Global Test Accuracy")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(ACCURACY_PLOT, dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(rounds, losses, marker="o", markersize=3)
        plt.xlabel("Communication Round")
        plt.ylabel("Global Test Loss")
        plt.title("Asynchronous Aggregation - Global Test Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(LOSS_PLOT, dpi=300)
        plt.close()

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    with open(SUMMARY_FILE, "w") as file:
        file.write("PHASE 5 (phase4.async) - ASYNCHRONOUS AGGREGATION\n")
        file.write("=" * 60 + "\n")
        file.write(f"Eta: {ETA}\n")
        file.write(f"Min train nodes: {MIN_TRAIN_NODES}\n")
        file.write(f"Clients: {NUM_CLIENTS}\n")
        file.write(f"Communication rounds: {num_rounds}\n")
        file.write(f"Total client replies: {TOTAL_CLIENT_REPLIES}\n")
        file.write(f"Total incremental updates applied: {TOTAL_UPDATES_APPLIED}\n")

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
            final_model_file = RESULTS_DIR / "final_async_lif.pth"
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
    print("PHASE 5 (phase4.async) COMPLETED")
    print("=" * 70)
    print(f"Eta                        : {ETA}")
    print(f"Communication rounds       : {num_rounds}")
    print(f"Total client replies       : {TOTAL_CLIENT_REPLIES}")
    print(f"Total updates applied      : {TOTAL_UPDATES_APPLIED}")

    if len(ROUND_RESULTS) > 0:
        final_result = ROUND_RESULTS[-1]
        print(
            f"Final test accuracy        : "
            f'{final_result["global_test_accuracy"]:.2f}%'
        )
        print(
            f"Final test loss            : "
            f'{final_result["global_test_loss"]:.4f}'
        )

    print()
    print("Results saved in:")
    print(RESULTS_DIR)
    print("=" * 70)