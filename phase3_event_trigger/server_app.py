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
from flwr.app import Context, ArrayRecord, MetricRecord
from flwr.serverapp.strategy import FedAvg

# ------------------------------------------------------------
# Project path (fixed)
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

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Filled in by main() from the Flower run config
ROUND_RESULTS = []
TOTAL_UPLOADS = 0
TOTAL_CLIENT_DECISIONS = 0

# ------------------------------------------------------------
# Global test dataset (loaded once)
# ------------------------------------------------------------
NMNIST_ROOT = PROJECT_ROOT / "data" / "nmnist" / "N-MNIST"

if not NMNIST_ROOT.exists():
    raise FileNotFoundError(f"N-MNIST path does not exist:\n{NMNIST_ROOT}")

test_dataset = NMNISTDataset(root=str(NMNIST_ROOT), train=False)

if len(test_dataset) == 0:
    raise RuntimeError("N-MNIST test dataset contains 0 samples.")

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ------------------------------------------------------------
# Global evaluation
# ------------------------------------------------------------
def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:

    model = LIFSNN().to(DEVICE)
    model.load_state_dict(arrays.to_torch_state_dict())
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in tqdm(
            test_loader,
            desc=f"Round {server_round} Evaluation",
            leave=False,
            unit="batch",
        ):
            frames = frames.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(frames).sum(dim=0)
            loss = criterion(outputs, labels)

            n = labels.size(0)
            total_loss += loss.item() * n
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += n

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
        f"Round {server_round} Global Test Accuracy: {test_accuracy:.2f}% "
        f"| Global Test Loss: {test_loss:.4f}"
    )

    return MetricRecord(
        {
            "global_test_accuracy": float(test_accuracy),
            "global_test_loss": float(test_loss),
        }
    )


# ------------------------------------------------------------
# Event-trigger counting
# ------------------------------------------------------------
def count_trigger_results(results):

    global TOTAL_UPLOADS, TOTAL_CLIENT_DECISIONS

    uploads = 0
    decisions = 0

    for message in results:

        if message.has_error():
            print(f"WARNING: client error: {message.error.reason}")
            continue
        if not message.has_content():
            print("WARNING: client reply has no content. Ignoring.")
            continue

        decisions += 1
        if "arrays" in message.content:
            uploads += 1

    TOTAL_UPLOADS += uploads
    TOTAL_CLIENT_DECISIONS += decisions

    return uploads, decisions


class EventTriggeredFedAvg(FedAvg):

    def aggregate_train(self, server_round, results):

        results = list(results)

        uploads, decisions = count_trigger_results(results)

        print()
        print("=" * 70)
        print(f"ROUND {server_round} EVENT-TRIGGER STATISTICS")
        print("=" * 70)
        print(f"Client decisions : {decisions}")
        print(f"Model uploads    : {uploads}")
        print(
            f"Upload frequency : "
            f"{(100.0 * uploads / decisions) if decisions else 0.0:.2f}%"
        )

        valid_results = [
            m for m in results
            if (not m.has_error()) and m.has_content()
            and "arrays" in m.content
        ]

        print(f"Updates aggregated: {len(valid_results)}")

        if len(valid_results) == 0:
            print("WARNING: No client uploaded a model update this round.")
            return None, None

        return super().aggregate_train(server_round, valid_results)


# ------------------------------------------------------------
# Server app
# ------------------------------------------------------------
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):

    global TOTAL_UPLOADS, TOTAL_CLIENT_DECISIONS

    # Read per-run settings from Flower (no hard-coded values)
    num_rounds = int(context.run_config["num-server-rounds"])
    DELTA = float(context.run_config["delta"])
    NUM_CLIENTS = int(context.run_config["num-clients"])

    # Safety check: simulation size must match num-clients
    try:
        connected = len(list(grid.get_node_ids()))
        if connected != NUM_CLIENTS:
            raise RuntimeError(
                f"num-clients={NUM_CLIENTS} but the simulation has "
                f"{connected} supernodes. Make both numbers equal."
            )
    except AttributeError:
        pass

    RESULTS_DIR = (
        PROJECT_ROOT / "results" / "phase3"
        / f"delta_{DELTA}" / f"clients_{NUM_CLIENTS}"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    CSV_FILE = RESULTS_DIR / "event_trigger_results.csv"
    ACCURACY_PLOT = RESULTS_DIR / "event_trigger_accuracy_vs_round.png"
    LOSS_PLOT = RESULTS_DIR / "event_trigger_loss_vs_round.png"
    SUMMARY_FILE = RESULTS_DIR / "event_trigger_summary.txt"

    print()
    print("=" * 70)
    print("PHASE 3 - EVENT-TRIGGERED FEDERATED SNN")
    print("=" * 70)
    print(f"Device     : {DEVICE}")
    print(f"Clients    : {NUM_CLIENTS}")
    print(f"Rounds     : {num_rounds}")
    print(f"Threshold δ: {DELTA}")
    print(f"Results dir: {RESULTS_DIR}")
    print("=" * 70)

    TOTAL_UPLOADS = 0
    TOTAL_CLIENT_DECISIONS = 0
    ROUND_RESULTS.clear()

    strategy = EventTriggeredFedAvg(
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

    # ---------------- CSV ----------------
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
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

    # ---------------- Plots ----------------
    rounds = [r["round"] for r in ROUND_RESULTS]
    accs = [r["global_test_accuracy"] for r in ROUND_RESULTS]
    losses = [r["global_test_loss"] for r in ROUND_RESULTS]

    if rounds:
        for values, ylabel, title, path in [
            (accs, "Global Test Accuracy (%)",
             "Event-Triggered Federated SNN - Global Test Accuracy",
             ACCURACY_PLOT),
            (losses, "Global Test Loss",
             "Event-Triggered Federated SNN - Global Test Loss",
             LOSS_PLOT),
        ]:
            plt.figure(figsize=(8, 5))
            plt.plot(rounds, values, marker="o", markersize=3)
            plt.xlabel("Communication Round")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(path, dpi=300)
            plt.close()

    # ---------------- Communication statistics ----------------
    baseline_uploads = NUM_CLIENTS * num_rounds
    event_uploads = TOTAL_UPLOADS

    reduction = (
        100.0 * (baseline_uploads - event_uploads) / baseline_uploads
        if baseline_uploads > 0 else 0.0
    )
    frequency = (
        100.0 * event_uploads / TOTAL_CLIENT_DECISIONS
        if TOTAL_CLIENT_DECISIONS > 0 else 0.0
    )

    final = ROUND_RESULTS[-1] if ROUND_RESULTS else None

    with open(SUMMARY_FILE, "w") as f:
        f.write("PHASE 3 - EVENT-TRIGGERED FEDERATED SNN\n")
        f.write("=" * 60 + "\n")
        f.write(f"Threshold delta: {DELTA}\n")
        f.write(f"Clients: {NUM_CLIENTS}\n")
        f.write(f"Communication rounds: {num_rounds}\n")
        f.write(f"Baseline uploads: {baseline_uploads}\n")
        f.write(f"Event-triggered uploads: {event_uploads}\n")
        f.write(f"Overall upload frequency: {frequency:.2f}%\n")
        f.write(f"Communication reduction: {reduction:.2f}%\n")
        if final:
            f.write(
                f'Final global test accuracy: '
                f'{final["global_test_accuracy"]:.4f}%\n'
            )
            f.write(
                f'Final global test loss: '
                f'{final["global_test_loss"]:.6f}\n'
            )

    # ---------------- Save final model ----------------
    if result is not None:
        try:
            torch.save(
                result.arrays.to_torch_state_dict(),
                RESULTS_DIR / "final_event_triggered_lif.pth",
            )
        except Exception as exc:
            print("Warning: final model could not be saved.")
            print(exc)

    # ---------------- Final print ----------------
    print()
    print("=" * 70)
    print("PHASE 3 COMPLETED")
    print("=" * 70)
    print(f"Threshold δ             : {DELTA}")
    print(f"Clients                 : {NUM_CLIENTS}")
    print(f"Communication rounds    : {num_rounds}")
    print(f"Baseline uploads        : {baseline_uploads}")
    print(f"Event-trigger uploads   : {event_uploads}")
    print(f"Upload frequency        : {frequency:.2f}%")
    print(f"Communication reduction : {reduction:.2f}%")
    if final:
        print(f'Final test accuracy     : {final["global_test_accuracy"]:.2f}%')
        print(f'Final test loss         : {final["global_test_loss"]:.4f}')
    print(f"Results saved in: {RESULTS_DIR}")
    print("=" * 70)