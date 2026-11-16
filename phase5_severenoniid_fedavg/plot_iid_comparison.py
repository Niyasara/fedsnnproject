
import re
import matplotlib.pyplot as plt
from pathlib import Path

# Input file path (same directory as this Python script)
file_path = Path(__file__).parent / "3diffoutput.txt"

# Read the text file
with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

# Identify the three data sections
sections = {
    "IID": [],
    "Mild Non-IID": [],
    "Severe Non-IID": []
}

current_section = None

for line in content.splitlines():

    text = line.strip().lower()

    if "severe non" in text:
        current_section = "Severe Non-IID"
        continue

    elif "mild non" in text:
        current_section = "Mild Non-IID"
        continue

    elif re.match(r"^iid\s*:?\s*$", text):
        current_section = "IID"
        continue

    if current_section:
        match = re.match(
            r"^\s*(\d+)[,\s]+([\d.]+)%?[,\s]+([\d.]+)\s*$",
            line
        )

        if match:
            round_no, accuracy, loss = match.groups()
            sections[current_section].append(
                (int(round_no), float(accuracy), float(loss))
            )

# Create combined graphs
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for name, values in sections.items():

    if not values:
        raise ValueError(f"No data found for {name}. Check section headings.")

    rounds, accuracy, loss = zip(*values)

    axes[0].plot(rounds, accuracy, marker='o',
                 markersize=3, linewidth=1.8, label=name)

    axes[1].plot(rounds, loss, marker='s',
                 markersize=3, linewidth=1.8, label=name)

# Accuracy graph
axes[0].set_title("Global Test Accuracy")
axes[0].set_xlabel("Communication Rounds")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_xlim(0, 50)
axes[0].set_ylim(0, 100)
axes[0].set_xticks(range(0, 51, 5))
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

# Loss graph
axes[1].set_title("Global Test Loss")
axes[1].set_xlabel("Communication Rounds")
axes[1].set_ylabel("Loss")
axes[1].set_xlim(0, 50)
axes[1].set_xticks(range(0, 51, 5))
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

plt.tight_layout()

# Save combined PNG
output_path = Path(__file__).parent / "IID_NonIID_Comparison.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Graph saved at: {output_path}")
