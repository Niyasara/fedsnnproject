import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "phase3"
)


# ============================================================
# OUTPUT FILE
# ============================================================

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "fixed_threshold_accuracy_loss_comparison.png"
)


# ============================================================
# FIND CSV FILES
# ============================================================

csv_files = glob.glob(
    os.path.join(
        RESULTS_DIR,
        "delta_*",
        "event_trigger_results.csv"
    )
)


# ============================================================
# CHECK FILES
# ============================================================

if len(csv_files) == 0:

    print("ERROR: No event_trigger_results.csv files found.")

    print()
    print("Searching in:")
    print(RESULTS_DIR)

    exit()


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(columns, possible_names):

    # Exact match
    for name in possible_names:

        if name in columns:
            return name

    # Case-insensitive
    lower_columns = {
        str(column).lower(): column
        for column in columns
    }

    for name in possible_names:

        if name.lower() in lower_columns:
            return lower_columns[name.lower()]

    # Partial match
    for column in columns:

        column_lower = str(column).lower()

        for name in possible_names:

            if name.lower() in column_lower:
                return column

    return None


# ============================================================
# READ RESULTS
# ============================================================

results = []


for csv_file in csv_files:

    # --------------------------------------------------------
    # Get threshold
    # --------------------------------------------------------

    folder_name = os.path.basename(
        os.path.dirname(csv_file)
    )

    threshold = folder_name.replace(
        "delta_",
        ""
    )

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(csv_file)

    print()
    print("=" * 70)
    print("Reading:")
    print(csv_file)

    print()
    print("Columns:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Find columns
    # --------------------------------------------------------

    round_column = find_column(
        df.columns,
        [
            "round",
            "Round",
            "communication_round",
            "Communication Round"
        ]
    )

    accuracy_column = find_column(
        df.columns,
        [
            "global_test_accuracy",
            "Global Test Accuracy",
            "test_accuracy",
            "Test Accuracy",
            "accuracy",
            "Accuracy"
        ]
    )

    loss_column = find_column(
        df.columns,
        [
            "global_test_loss",
            "Global Test Loss",
            "test_loss",
            "Test Loss",
            "loss",
            "Loss"
        ]
    )

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    if round_column is None:

        print("ERROR: Round column not found.")
        exit()

    if accuracy_column is None:

        print("ERROR: Accuracy column not found.")
        exit()

    if loss_column is None:

        print("ERROR: Loss column not found.")
        exit()

    print()
    print("Round column    :", round_column)
    print("Accuracy column :", accuracy_column)
    print("Loss column     :", loss_column)

    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    df[round_column] = pd.to_numeric(
        df[round_column],
        errors="coerce"
    )

    df[accuracy_column] = pd.to_numeric(
        df[accuracy_column],
        errors="coerce"
    )

    df[loss_column] = pd.to_numeric(
        df[loss_column],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            round_column,
            accuracy_column,
            loss_column
        ]
    )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    results.append(
        {
            "threshold": threshold,
            "round": df[round_column].values,
            "accuracy": df[accuracy_column].values,
            "loss": df[loss_column].values
        }
    )


# ============================================================
# SORT THRESHOLDS
# ============================================================

results.sort(
    key=lambda x: float(x["threshold"])
)


# ============================================================
# PRINT FOUND THRESHOLDS
# ============================================================

print()
print("=" * 70)
print("THRESHOLDS FOUND")
print("=" * 70)

for result in results:

    print(
        f"δ = {result['threshold']} | "
        f"Rounds = {len(result['round'])}"
    )


# ============================================================
# CREATE ONE GRAPH
# ============================================================

fig, ax_accuracy = plt.subplots(
    figsize=(12, 7)
)


# ============================================================
# SECOND Y AXIS FOR LOSS
# ============================================================

ax_loss = ax_accuracy.twinx()


# ============================================================
# LINE STYLES
# ============================================================

# Accuracy  -> solid line
# Loss      -> dashed line


# ============================================================
# PLOT RESULTS
# ============================================================

for result in results:

    threshold = result["threshold"]

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy_line = ax_accuracy.plot(
        result["round"],
        result["accuracy"],
        linestyle="-",
        linewidth=2.2,
        label=f"δ = {threshold} Accuracy"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    ax_loss.plot(
        result["round"],
        result["loss"],
        linestyle="--",
        linewidth=2.0,
        color=accuracy_line[0].get_color(),
        label=f"δ = {threshold} Loss"
    )


# ============================================================
# LEFT Y AXIS
# ============================================================

ax_accuracy.set_xlabel(
    "Communication Round",
    fontsize=12
)

ax_accuracy.set_ylabel(
    "Global Test Accuracy (%)",
    fontsize=12
)


# ============================================================
# RIGHT Y AXIS
# ============================================================

ax_loss.set_ylabel(
    "Global Test Loss",
    fontsize=12
)


# ============================================================
# TITLE
# ============================================================

ax_accuracy.set_title(
    "Accuracy and Loss for Fixed Event-Trigger Thresholds",
    fontsize=14
)


# ============================================================
# GRID
# ============================================================

ax_accuracy.grid(
    True,
    alpha=0.3
)


# ============================================================
# COMBINE LEGENDS
# ============================================================

accuracy_handles, accuracy_labels = (
    ax_accuracy.get_legend_handles_labels()
)

loss_handles, loss_labels = (
    ax_loss.get_legend_handles_labels()
)

all_handles = (
    accuracy_handles +
    loss_handles
)

all_labels = (
    accuracy_labels +
    loss_labels
)


# ============================================================
# LEGEND
# ============================================================

ax_accuracy.legend(
    all_handles,
    all_labels,
    loc="center right",
    fontsize=9,
    frameon=True
)


# ============================================================
# LAYOUT
# ============================================================

fig.tight_layout()


# ============================================================
# SAVE ONE PNG
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("GRAPH GENERATED SUCCESSFULLY")
print("=" * 70)

print()
print("Output:")
print(OUTPUT_FILE)

print()
print("Graph contains:")
print("  Solid lines  = Accuracy")
print("  Dashed lines = Loss")
print("  One color    = One threshold")
print("  Four thresholds = Four accuracy + four loss curves")

print()
print("Only ONE PNG was generated.")
print("No CSV file was created.")
print("No second PNG was created.")