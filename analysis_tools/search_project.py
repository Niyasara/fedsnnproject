
import os

# Project directory
PROJECT_DIR = os.path.expanduser("~/Niya/fedsnnproject")

# Keywords related to the computational complexity experiment
keywords = [
    "training time",
    "training energy",
    "MAC operations",
    "FLOPs",
    "FedTA",
    "Standard-LIF",
    "Delta-LIF",
    "Standard-SSM",
    "Delta-SSM",
    "Scenario A",
    "Scenario B",
    "energy consumption"
]

# File types to search
extensions = (".py", ".csv", ".txt", ".json", ".ipynb", ".tex", ".log")

print("\nSearching project files...\n")

found = False

for root, dirs, files in os.walk(PROJECT_DIR):

    # Ignore unnecessary folders
    dirs[:] = [
        d for d in dirs
        if d not in ["venv", ".venv", "__pycache__", ".git"]
    ]

    for filename in files:

        if filename.endswith(extensions):

            filepath = os.path.join(root, filename)

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as file:

                    for line_number, line in enumerate(file, start=1):

                        for keyword in keywords:

                            if keyword.lower() in line.lower():

                                print(f"\nFile: {filepath}")
                                print(f"Line: {line_number}")
                                print(f"Keyword: {keyword}")
                                print(f"Content: {line.strip()}")

                                found = True
                                break

            except Exception as error:
                print(f"Could not read {filepath}: {error}")

if not found:
    print("No matching references found.")

print("\nSearch completed.")
