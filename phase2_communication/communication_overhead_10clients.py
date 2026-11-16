
# ============================================================
# Phase 2: Communication Overhead Analysis
# Conventional Synchronous FedAvg - 10 Clients
# ============================================================

from pathlib import Path
import csv
import sys

# ------------------------------------------------------------
# Project Path
# ------------------------------------------------------------
# Script location:
# fedsnnproject/phase2_communication/
# PROJECT_ROOT = fedsnnproject/

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# Import Standard LIF Model
# ------------------------------------------------------------
from model.lif_snn import LIFSNN


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
NUM_CLIENTS = 10
NUM_ROUNDS = 50
BYTES_PER_PARAMETER = 4


# ------------------------------------------------------------
# Calculate Exact Model Size
# ------------------------------------------------------------
model = LIFSNN(num_classes=10)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

MODEL_SIZE_MB = (
    total_params
    * BYTES_PER_PARAMETER
    / (1024 * 1024)
)


# ------------------------------------------------------------
# Locate 10-Client Phase 1 Results
# ------------------------------------------------------------
# Actual folder structure:
#
# fedsnnproject/
#   federated_standard_lif/
#       results/
#           phase1/
#               federated_standard_lif_10clients/
#                   global_test_results.csv

PHASE1_DIR = (
    PROJECT_ROOT
    / "federated_standard_lif"
    / "results"
    / "phase1"
)

PHASE1_CSV = (
    PHASE1_DIR
    / "federated_standard_lif_10clients"
    / "global_test_results.csv"
)


# ------------------------------------------------------------
# Verify 10-Client CSV
# ------------------------------------------------------------
if not PHASE1_CSV.exists():

    print("\nERROR: 10-client results CSV not found.")

    print("\nExpected CSV location:")
    print(PHASE1_CSV)

    print("\nAvailable CSV files:")

    if PHASE1_DIR.exists():

        for path in PHASE1_DIR.rglob("*.csv"):
            print(path)

    else:

        print("Phase 1 directory does not exist:")
        print(PHASE1_DIR)

    sys.exit(1)


print("\n10-client Phase 1 results found:")
print(PHASE1_CSV)


# ------------------------------------------------------------
# Phase 2 Output Directory
# ------------------------------------------------------------
RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase2"
    / "communication_overhead"
    / "client_10"
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
    / "communication_statistics.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "communication_summary.txt"
)


# ------------------------------------------------------------
# Read Phase 1 Accuracy and Loss
# ------------------------------------------------------------
phase1_results = []

with open(
    PHASE1_CSV,
    "r",
    newline=""
) as file:

    reader = csv.DictReader(file)

    required_columns = [
        "Round",
        "Global Test Accuracy (%)",
        "Global Test Loss"
    ]

    if not reader.fieldnames or not all(
        column in reader.fieldnames
        for column in required_columns
    ):

        raise ValueError(
            "Missing required CSV columns: "
            + ", ".join(required_columns)
        )

    for row in reader:

        phase1_results.append({

            "round": int(row["Round"]),

            "accuracy": float(
                row["Global Test Accuracy (%)"]
            ),

            "loss": float(
                row["Global Test Loss"]
            )

        })


# ------------------------------------------------------------
# Select Training Rounds
# ------------------------------------------------------------
training_results = sorted(

    [
        result
        for result in phase1_results
        if 1 <= result["round"] <= NUM_ROUNDS
    ],

    key=lambda x: x["round"]

)


# ------------------------------------------------------------
# Verify Training Results
# ------------------------------------------------------------
if not training_results:

    print("\nERROR: No valid training rounds found.")

    sys.exit(1)


if len(training_results) != NUM_ROUNDS:

    print(
        f"\nWARNING: Expected {NUM_ROUNDS} rounds, "
        f"but found {len(training_results)}."
    )


# ------------------------------------------------------------
# Communication Calculations
# ------------------------------------------------------------

# All 10 clients upload in every synchronous round.

uploads_per_round = NUM_CLIENTS

participation_rate = 100.0

total_uploads = (
    uploads_per_round
    * len(training_results)
)

data_per_round = (
    uploads_per_round
    * MODEL_SIZE_MB
)

total_data_transferred = (
    total_uploads
    * MODEL_SIZE_MB
)

aggregation_frequency = (
    "Every communication round"
)


# ------------------------------------------------------------
# Save Communication Statistics CSV
# ------------------------------------------------------------
with open(
    CSV_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([

        "Round",
        "Number of Clients",
        "Participating Clients",
        "Participation Rate (%)",
        "Uploads",
        "Model Size (MB)",
        "Data Transferred (MB)",
        "Cumulative Data Transferred (MB)",
        "Global Test Accuracy (%)",
        "Global Test Loss"

    ])

    cumulative_data = 0.0

    for result in training_results:

        cumulative_data += data_per_round

        writer.writerow([

            result["round"],

            NUM_CLIENTS,

            NUM_CLIENTS,

            f"{participation_rate:.2f}",

            uploads_per_round,

            f"{MODEL_SIZE_MB:.6f}",

            f"{data_per_round:.6f}",

            f"{cumulative_data:.6f}",

            f"{result['accuracy']:.4f}",

            f"{result['loss']:.6f}"

        ])


# ------------------------------------------------------------
# Convergence Analysis
# ------------------------------------------------------------
initial_accuracy = (
    training_results[0]["accuracy"]
)

final_accuracy = (
    training_results[-1]["accuracy"]
)

initial_loss = (
    training_results[0]["loss"]
)

final_loss = (
    training_results[-1]["loss"]
)


# ------------------------------------------------------------
# Save Communication Summary
# ------------------------------------------------------------
with open(
    SUMMARY_FILE,
    "w"
) as file:

    file.write(
        "PHASE 2: COMMUNICATION OVERHEAD ANALYSIS\n"
    )

    file.write(
        "CONVENTIONAL SYNCHRONOUS FEDAVG - 10 CLIENTS\n"
    )

    file.write("\n")

    file.write(
        f"Number of Clients       : {NUM_CLIENTS}\n"
    )

    file.write(
        f"Communication Rounds    : "
        f"{len(training_results)}\n"
    )

    file.write(
        f"Uploads per Round       : "
        f"{uploads_per_round}\n"
    )

    file.write(
        f"Total Number of Uploads : "
        f"{total_uploads}\n"
    )

    file.write(
        f"Client Participation    : "
        f"{participation_rate:.2f}%\n"
    )

    file.write(
        f"Model Parameters        : "
        f"{total_params}\n"
    )

    file.write(
        f"Model Size              : "
        f"{MODEL_SIZE_MB:.6f} MB\n"
    )

    file.write(
        f"Data per Round          : "
        f"{data_per_round:.6f} MB\n"
    )

    file.write(
        f"Total Data Transferred  : "
        f"{total_data_transferred:.6f} MB\n"
    )

    file.write(
        f"Aggregation Frequency   : "
        f"{aggregation_frequency}\n"
    )

    file.write("\nCOMMUNICATION COST\n")

    file.write(
        f"CommCost = {total_uploads} uploads\n"
    )

    file.write("\nCONVERGENCE ANALYSIS\n")

    file.write(
        f"Round 1 Accuracy        : "
        f"{initial_accuracy:.4f}%\n"
    )

    file.write(
        f"Final Accuracy          : "
        f"{final_accuracy:.4f}%\n"
    )

    file.write(
        f"Round 1 Loss            : "
        f"{initial_loss:.6f}\n"
    )

    file.write(
        f"Final Loss              : "
        f"{final_loss:.6f}\n"
    )


# ------------------------------------------------------------
# Display Final Results
# ------------------------------------------------------------
print("\n" + "=" * 65)

print(
    "PHASE 2: COMMUNICATION OVERHEAD ANALYSIS"
)

print(
    "CONVENTIONAL SYNCHRONOUS FEDAVG - 10 CLIENTS"
)

print("=" * 65)

print(
    f"Number of Clients       : {NUM_CLIENTS}"
)

print(
    f"Communication Rounds    : "
    f"{len(training_results)}"
)

print(
    f"Uploads per Round       : {uploads_per_round}"
)

print(
    f"Total Number of Uploads : {total_uploads}"
)

print(
    f"Client Participation    : "
    f"{participation_rate:.2f}%"
)

print(
    f"Model Parameters        : {total_params}"
)

print(
    f"Model Size              : {MODEL_SIZE_MB:.6f} MB"
)

print(
    f"Data per Round          : {data_per_round:.6f} MB"
)

print(
    f"Total Data Transferred  : "
    f"{total_data_transferred:.6f} MB"
)

print(
    f"Aggregation Frequency   : "
    f"{aggregation_frequency}"
)

print("=" * 65)

print("\nCONVERGENCE ANALYSIS")

print(
    f"Round 1 Accuracy        : "
    f"{initial_accuracy:.4f}%"
)

print(
    f"Final Accuracy          : "
    f"{final_accuracy:.4f}%"
)

print(
    f"Round 1 Loss            : "
    f"{initial_loss:.6f}"
)

print(
    f"Final Loss              : "
    f"{final_loss:.6f}"
)

print("=" * 65)

print("\nCOMMUNICATION COST")

print(
    f"CommCost = {total_uploads} uploads"
)

print("\nCSV saved to:")
print(CSV_FILE)

print("\nSummary saved to:")
print(SUMMARY_FILE)

print("\nOUTPUT DIRECTORY:")
print(RESULTS_DIR)

print("=" * 65)
