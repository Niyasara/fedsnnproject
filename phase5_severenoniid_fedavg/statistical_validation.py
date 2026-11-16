
import re
import csv
from pathlib import Path
from statistics import mean, stdev

# ==================================================
# CONFIGURATION
# ==================================================
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "output.txt"
OUTPUT_CSV = BASE_DIR / "statistical_summary.csv"

METRICS = [
    "Final test accuracy",
    "Final test loss",
    "Final macro precision",
    "Final macro recall",
    "Final macro F1",
    "Peak test accuracy",
    "Mean accuracy (last 10)",
    "Stability (accuracy std)",
]


# ==================================================
# READ THE THREE RUNS FROM output.txt
# ==================================================
def read_runs(file_path):
    text = file_path.read_text(encoding="utf-8")

    # Split text at RUN=1, RUN=2, RUN=3, etc.
    sections = re.split(
        r"(?m)^\s*RUN\s*=\s*\d+\s*$",
        text
    )

    runs = []

    for section in sections[1:]:
        results = {}

        for metric in METRICS:
            pattern = (
                r"(?m)^\s*"
                + re.escape(metric)
                + r"\s*:\s*([-+]?\d*\.?\d+)"
            )

            match = re.search(pattern, section)

            if match:
                results[metric] = float(match.group(1))

        if "Final test accuracy" in results:
            runs.append(results)

    if len(runs) < 2:
        raise ValueError(
            "Could not find at least two run summaries. "
            "Check that output.txt contains RUN=1, RUN=2, RUN=3 "
            "and the expected summary labels."
        )

    return runs


# ==================================================
# CALCULATE MEAN AND SAMPLE STANDARD DEVIATION
# ==================================================
def calculate_statistics(runs):
    rows = []

    print("\n" + "=" * 78)
    print("STATISTICAL VALIDATION - SEVERE NON-IID FEDAVG")
    print("=" * 78)
    print(f"Number of runs: {len(runs)}")
    print("Format: Mean +/- Sample Standard Deviation")
    print("-" * 78)

    print(f"{'Metric':32s} {'Mean':>12s} {'Std Dev':>12s} {'Mean +/- SD':>19s}")
    print("-" * 78)

    for metric in METRICS:
        values = [
            run[metric]
            for run in runs
            if metric in run
        ]

        if not values:
            print(f"{metric:32s} {'Not found':>12s}")
            continue

        avg = mean(values)
        sd = stdev(values) if len(values) >= 2 else 0.0

        # Accuracy and classification metrics are already percentages.
        # Keep loss and stability values at four decimal places.
        decimals = 4 if (
            "loss" in metric.lower()
            or "stability" in metric.lower()
        ) else 2

        formatted = (
            f"{avg:.{decimals}f} +/- {sd:.{decimals}f}"
        )

        print(
            f"{metric:32s} "
            f"{avg:12.{decimals}f} "
            f"{sd:12.{decimals}f} "
            f"{formatted:>19s}"
        )

        rows.append({
            "Metric": metric,
            "Mean": round(avg, decimals),
            "Sample_Std_Dev": round(sd, decimals),
            "Mean_Plus_Minus_SD": formatted,
            "Number_of_Runs": len(values),
        })

    print("-" * 78)

    # Save the calculated results.
    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Metric",
                "Mean",
                "Sample_Std_Dev",
                "Mean_Plus_Minus_SD",
                "Number_of_Runs",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {OUTPUT_CSV}")


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Place statistical_validation.py and output.txt "
            "in the same folder."
        )

    runs = read_runs(INPUT_FILE)

    print(f"Input file: {INPUT_FILE}")
    calculate_statistics(runs)
