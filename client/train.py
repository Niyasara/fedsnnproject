import sys
import os
import csv
import random

# ============================================================
# Add project root to Python path
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Imports
# ============================================================

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from model.lif_snn import LIFSNN
from dataset.dataset_loader import get_data_loaders


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32

LOCAL_EPOCHS = 5

LEARNING_RATE = 0.001

NUM_CLASSES = 10

SEED = 42

CLASS_NAMES = [
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9"
]


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Evaluation Function
# ============================================================

def evaluate_model(
    model,
    test_loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    correct = 0

    total = 0

    all_labels = []

    all_predictions = []


    with torch.no_grad():

        progress = tqdm(
            test_loader,
            desc="Testing",
            leave=False,
            unit="batch"
        )


        for data, labels in progress:

            data = data.to(DEVICE)

            labels = labels.to(DEVICE)


            # ------------------------------------------------
            # SNN forward pass
            # Output:
            # [time, batch, classes]
            # ------------------------------------------------

            spike_output = model(data)


            # ------------------------------------------------
            # Sum spikes over time
            #
            # [time, batch, classes]
            #          ↓
            # [batch, classes]
            # ------------------------------------------------

            output = spike_output.sum(
                dim=0
            )


            # ------------------------------------------------
            # Test loss
            # ------------------------------------------------

            loss = criterion(
                output,
                labels
            )


            batch_size = labels.size(0)


            total_loss += (
                loss.item()
                * batch_size
            )


            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            predictions = output.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += batch_size


            # ------------------------------------------------
            # Store predictions
            # ------------------------------------------------

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )


        progress.close()


    # ========================================================
    # Average Test Loss
    # ========================================================

    average_test_loss = (
        total_loss / total
    )


    # ========================================================
    # Test Accuracy
    # ========================================================

    test_accuracy = (
        accuracy_score(
            all_labels,
            all_predictions
        ) * 100.0
    )


    # ========================================================
    # Precision
    # ========================================================

    test_precision = (
        precision_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0
        ) * 100.0
    )


    # ========================================================
    # Recall
    # ========================================================

    test_recall = (
        recall_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0
        ) * 100.0
    )


    # ========================================================
    # F1 Score
    # ========================================================

    test_f1 = (
        f1_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0
        ) * 100.0
    )


    # ========================================================
    # Confusion Matrix
    # ========================================================

    confusion_mat = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES))
    )


    # ========================================================
    # Classification Report
    # ========================================================

    classification_rep = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )


    return (
        average_test_loss,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
        confusion_mat,
        classification_rep
    )


# ============================================================
# Training Function
# ============================================================

def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    epochs
):

    train_losses = []

    train_accuracies = []

    test_losses = []

    test_accuracies = []


    # Final metrics
    final_precision = 0.0
    final_recall = 0.0
    final_f1 = 0.0

    final_confusion_matrix = None

    final_classification_report = ""


    # ========================================================
    # Epoch Loop
    # ========================================================

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        correct = 0

        total = 0


        # ====================================================
        # Training Progress Bar
        # ====================================================

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=True
        )


        for data, labels in progress:

            data = data.to(DEVICE)

            labels = labels.to(DEVICE)


            # ------------------------------------------------
            # Clear gradients
            # ------------------------------------------------

            optimizer.zero_grad()


            # ------------------------------------------------
            # SNN forward pass
            # ------------------------------------------------

            spike_output = model(data)


            # ------------------------------------------------
            # Sum spikes over time
            # ------------------------------------------------

            output = spike_output.sum(
                dim=0
            )


            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                output,
                labels
            )


            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            optimizer.step()


            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            batch_size = labels.size(0)


            total_loss += (
                loss.item()
                * batch_size
            )


            predictions = output.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += batch_size


            # ------------------------------------------------
            # Current accuracy
            # ------------------------------------------------

            current_accuracy = (
                100.0
                * correct
                / total
            )


            # ------------------------------------------------
            # Progress bar
            # ------------------------------------------------

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{current_accuracy:.2f}%"
            )


        progress.close()


        # ====================================================
        # Training Epoch Statistics
        # ====================================================

        average_train_loss = (
            total_loss / total
        )


        train_accuracy = (
            100.0
            * correct
            / total
        )


        train_losses.append(
            average_train_loss
        )


        train_accuracies.append(
            train_accuracy
        )


        # ====================================================
        # Test After Each Epoch
        # ====================================================

        (
            test_loss,
            test_accuracy,
            precision,
            recall,
            f1,
            confusion_mat,
            classification_rep

        ) = evaluate_model(
            model,
            test_loader,
            criterion
        )


        test_losses.append(
            test_loss
        )


        test_accuracies.append(
            test_accuracy
        )


        # Store latest/final metrics
        final_precision = precision

        final_recall = recall

        final_f1 = f1

        final_confusion_matrix = confusion_mat

        final_classification_report = classification_rep


        # ====================================================
        # Print Epoch Results
        # ====================================================

        print(
            f"\nEpoch [{epoch + 1}/{epochs}]"
        )

        print(
            f"Train Loss     : "
            f"{average_train_loss:.6f}"
        )

        print(
            f"Train Accuracy : "
            f"{train_accuracy:.2f}%"
        )

        print(
            f"Test Loss      : "
            f"{test_loss:.6f}"
        )

        print(
            f"Test Accuracy  : "
            f"{test_accuracy:.2f}%"
        )

        print(
            f"Precision      : "
            f"{precision:.2f}%"
        )

        print(
            f"Recall         : "
            f"{recall:.2f}%"
        )

        print(
            f"F1 Score       : "
            f"{f1:.2f}%"
        )

        print("-" * 70)


    return (
        train_losses,
        train_accuracies,
        test_losses,
        test_accuracies,
        final_precision,
        final_recall,
        final_f1,
        final_confusion_matrix,
        final_classification_report
    )


# ============================================================
# Save Results
# ============================================================

def save_results(
    train_losses,
    train_accuracies,
    test_losses,
    test_accuracies,
    test_precision,
    test_recall,
    test_f1,
    confusion_mat,
    classification_rep,
    model,
    train_dataset,
    test_dataset
):

    # ========================================================
    # Results Directory
    # ========================================================

    results_dir = os.path.join(
        PROJECT_ROOT,
        "results",
        "phase1"
    )


    models_dir = os.path.join(
        results_dir,
        "models"
    )


    os.makedirs(
        results_dir,
        exist_ok=True
    )


    os.makedirs(
        models_dir,
        exist_ok=True
    )


    # ========================================================
    # Final Values
    # ========================================================

    final_train_loss = train_losses[-1]

    final_train_accuracy = train_accuracies[-1]

    final_test_loss = test_losses[-1]

    final_test_accuracy = test_accuracies[-1]


    # ========================================================
    # Total Test Loss
    # ========================================================

    total_test_loss = (
        final_test_loss
        * len(test_dataset)
    )


    # ========================================================
    # Training Log CSV
    # ========================================================

    csv_file = os.path.join(
        results_dir,
        "training_log.csv"
    )


    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)


        writer.writerow(
            [
                "Epoch",
                "Train Loss",
                "Train Accuracy",
                "Test Loss",
                "Test Accuracy"
            ]
        )


        for epoch in range(
            len(train_losses)
        ):

            writer.writerow(
                [
                    epoch + 1,
                    f"{train_losses[epoch]:.6f}",
                    f"{train_accuracies[epoch]:.4f}",
                    f"{test_losses[epoch]:.6f}",
                    f"{test_accuracies[epoch]:.4f}"
                ]
            )


    # ========================================================
    # Test Metrics CSV
    # ========================================================

    metrics_file = os.path.join(
        results_dir,
        "test_metrics.csv"
    )


    with open(
        metrics_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)


        writer.writerow(
            [
                "Metric",
                "Value"
            ]
        )


        writer.writerow(
            [
                "Accuracy (%)",
                f"{final_test_accuracy:.4f}"
            ]
        )


        writer.writerow(
            [
                "Precision (%)",
                f"{test_precision:.4f}"
            ]
        )


        writer.writerow(
            [
                "Recall (%)",
                f"{test_recall:.4f}"
            ]
        )


        writer.writerow(
            [
                "F1 Score (%)",
                f"{test_f1:.4f}"
            ]
        )


    # ========================================================
    # Confusion Matrix CSV
    # ========================================================

    confusion_csv = os.path.join(
        results_dir,
        "confusion_matrix.csv"
    )


    with open(
        confusion_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)


        writer.writerow(
            [
                "Actual / Predicted"
            ] + CLASS_NAMES
        )


        for i, row in enumerate(
            confusion_mat
        ):

            writer.writerow(
                [
                    CLASS_NAMES[i]
                ] + row.tolist()
            )


    # ========================================================
    # Classification Report
    # ========================================================

    report_file = os.path.join(
        results_dir,
        "classification_report.txt"
    )


    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "STANDARD LIF-SNN "
            "CLASSIFICATION REPORT\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            classification_rep
        )


    # ========================================================
    # Save Model
    # ========================================================

    model_file = os.path.join(
        models_dir,
        "local_snn_model.pth"
    )


    torch.save(
        model.state_dict(),
        model_file
    )


    # ========================================================
    # Training Summary
    # ========================================================

    summary_file = os.path.join(
        results_dir,
        "training_summary.txt"
    )


    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "STANDARD LIF-SNN "
            "SINGLE-CLIENT TRAINING SUMMARY\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )


        f.write(
            f"Device: {DEVICE}\n"
        )


        if torch.cuda.is_available():

            f.write(
                f"GPU: "
                f"{torch.cuda.get_device_name(0)}\n"
            )

            f.write(
                f"CUDA: "
                f"{torch.version.cuda}\n"
            )


        f.write(
            "\nDataset: N-MNIST\n"
        )


        f.write(
            f"Training samples: "
            f"{len(train_dataset)}\n"
        )


        f.write(
            f"Testing samples: "
            f"{len(test_dataset)}\n"
        )


        f.write(
            f"Batch size: "
            f"{BATCH_SIZE}\n"
        )


        f.write(
            f"Local epochs: "
            f"{LOCAL_EPOCHS}\n"
        )


        f.write(
            f"Learning rate: "
            f"{LEARNING_RATE}\n"
        )


        f.write(
            "Optimizer: Adam\n"
        )


        f.write(
            "Neuron: Standard LIF\n\n"
        )


        # ----------------------------------------------------
        # Final Training Results
        # ----------------------------------------------------

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "FINAL RESULTS\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )


        f.write(
            f"Final Training Loss: "
            f"{final_train_loss:.6f}\n"
        )


        f.write(
            f"Final Training Accuracy: "
            f"{final_train_accuracy:.2f}%\n"
        )


        f.write(
            f"Final Test Loss: "
            f"{final_test_loss:.6f}\n"
        )


        f.write(
            f"Final Test Accuracy: "
            f"{final_test_accuracy:.2f}%\n"
        )


        f.write(
            f"Precision: "
            f"{test_precision:.2f}%\n"
        )


        f.write(
            f"Recall: "
            f"{test_recall:.2f}%\n"
        )


        f.write(
            f"F1 Score: "
            f"{test_f1:.2f}%\n"
        )


        f.write(
            f"Total Test Loss: "
            f"{total_test_loss:.6f}\n\n"
        )


        # ----------------------------------------------------
        # Classification Report
        # ----------------------------------------------------

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "CLASSIFICATION REPORT\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            classification_rep
        )


    # ========================================================
    # Training Loss Graph
    # ========================================================

    plt.figure()


    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        marker="o"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Training Loss"
    )


    plt.title(
        "Standard LIF-SNN Training Loss"
    )


    plt.grid(True)

    plt.tight_layout()


    train_loss_plot = os.path.join(
        results_dir,
        "training_loss.png"
    )


    plt.savefig(
        train_loss_plot,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # ========================================================
    # Training Accuracy Graph
    # ========================================================

    plt.figure()


    plt.plot(
        range(1, len(train_accuracies) + 1),
        train_accuracies,
        marker="o"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Training Accuracy (%)"
    )


    plt.title(
        "Standard LIF-SNN Training Accuracy"
    )


    plt.grid(True)

    plt.tight_layout()


    train_accuracy_plot = os.path.join(
        results_dir,
        "training_accuracy.png"
    )


    plt.savefig(
        train_accuracy_plot,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # ========================================================
    # Test Loss Graph
    # ========================================================

    plt.figure()


    plt.plot(
        range(1, len(test_losses) + 1),
        test_losses,
        marker="o"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Test Loss"
    )


    plt.title(
        "Standard LIF-SNN Test Loss"
    )


    plt.grid(True)

    plt.tight_layout()


    test_loss_plot = os.path.join(
        results_dir,
        "test_loss.png"
    )


    plt.savefig(
        test_loss_plot,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # ========================================================
    # Test Accuracy Graph
    # ========================================================

    plt.figure()


    plt.plot(
        range(1, len(test_accuracies) + 1),
        test_accuracies,
        marker="o"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Test Accuracy (%)"
    )


    plt.title(
        "Standard LIF-SNN Test Accuracy"
    )


    plt.grid(True)

    plt.tight_layout()


    test_accuracy_plot = os.path.join(
        results_dir,
        "test_accuracy.png"
    )


    plt.savefig(
        test_accuracy_plot,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # ========================================================
    # Confusion Matrix Figure
    # ========================================================

    plt.figure(
        figsize=(8, 7)
    )


    plt.imshow(
        confusion_mat,
        interpolation="nearest"
    )


    plt.title(
        "Standard LIF-SNN Confusion Matrix"
    )


    plt.colorbar()


    plt.xticks(
        range(NUM_CLASSES),
        CLASS_NAMES
    )


    plt.yticks(
        range(NUM_CLASSES),
        CLASS_NAMES
    )


    plt.xlabel(
        "Predicted Label"
    )


    plt.ylabel(
        "True Label"
    )


    threshold = (
        confusion_mat.max()
        / 2.0
    )


    for i in range(
        confusion_mat.shape[0]
    ):

        for j in range(
            confusion_mat.shape[1]
        ):

            plt.text(
                j,
                i,
                str(
                    confusion_mat[i, j]
                ),
                ha="center",
                va="center",
                color=(
                    "white"
                    if confusion_mat[i, j]
                    > threshold
                    else "black"
                )
            )


    plt.tight_layout()


    confusion_plot = os.path.join(
        results_dir,
        "confusion_matrix.png"
    )


    plt.savefig(
        confusion_plot,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # ========================================================
    # Print Saved Files
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "RESULTS SAVED"
    )

    print("=" * 70)


    print(
        "\nTraining Log:"
    )

    print(
        csv_file
    )


    print(
        "\nTest Metrics:"
    )

    print(
        metrics_file
    )


    print(
        "\nConfusion Matrix CSV:"
    )

    print(
        confusion_csv
    )


    print(
        "\nClassification Report:"
    )

    print(
        report_file
    )


    print(
        "\nSummary:"
    )

    print(
        summary_file
    )


    print(
        "\nModel:"
    )

    print(
        model_file
    )


    print(
        "\nTraining Loss:"
    )

    print(
        train_loss_plot
    )


    print(
        "\nTraining Accuracy:"
    )

    print(
        train_accuracy_plot
    )


    print(
        "\nTest Loss:"
    )

    print(
        test_loss_plot
    )


    print(
        "\nTest Accuracy:"
    )

    print(
        test_accuracy_plot
    )


    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_plot
    )


    # ========================================================
    # Final Results
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "FINAL RESULTS"
    )

    print("=" * 70)


    print(
        f"\nFinal Training Loss     : "
        f"{final_train_loss:.6f}"
    )


    print(
        f"Final Training Accuracy : "
        f"{final_train_accuracy:.2f}%"
    )


    print(
        f"Final Test Loss         : "
        f"{final_test_loss:.6f}"
    )


    print(
        f"Final Test Accuracy     : "
        f"{final_test_accuracy:.2f}%"
    )


    print(
        f"Precision               : "
        f"{test_precision:.2f}%"
    )


    print(
        f"Recall                  : "
        f"{test_recall:.2f}%"
    )


    print(
        f"F1 Score                : "
        f"{test_f1:.2f}%"
    )


    print(
        f"Total Test Loss         : "
        f"{total_test_loss:.6f}"
    )


    # ========================================================
    # Classification Report
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "CLASSIFICATION REPORT"
    )

    print("=" * 70)

    print(
        classification_rep
    )


    # ========================================================
    # Confusion Matrix
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "CONFUSION MATRIX"
    )

    print("=" * 70)

    print(
        confusion_mat
    )


# ============================================================
# Main Program
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "STANDARD LIF-SNN SINGLE-CLIENT TRAINING"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print(
        "\nDevice:",
        DEVICE
    )


    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print(
        "\nLoading N-MNIST..."
    )


    (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader
    ) = get_data_loaders(
        batch_size=BATCH_SIZE
    )


    print(
        "\nDataset loaded successfully."
    )


    print(
        "Training samples:",
        len(train_dataset)
    )


    print(
        "Testing samples:",
        len(test_dataset)
    )


    print(
        "Batch size:",
        BATCH_SIZE
    )


    # --------------------------------------------------------
    # Create Model
    # --------------------------------------------------------

    model = LIFSNN(
        num_classes=NUM_CLASSES
    ).to(DEVICE)


    print(
        "\nSNN model created successfully."
    )


    # --------------------------------------------------------
    # Parameter Count
    # --------------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )


    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    print(
        "Total parameters:",
        total_parameters
    )


    print(
        "Trainable parameters:",
        trainable_parameters
    )


    # --------------------------------------------------------
    # Loss Function
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    print(
        "\nOptimizer: Adam"
    )


    print(
        "Learning rate:",
        LEARNING_RATE
    )


    print(
        "Local epochs:",
        LOCAL_EPOCHS
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print(
        "\nStarting training...\n"
    )


    (
        train_losses,
        train_accuracies,
        test_losses,
        test_accuracies,
        test_precision,
        test_recall,
        test_f1,
        confusion_mat,
        classification_rep

    ) = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=LOCAL_EPOCHS
    )


    # --------------------------------------------------------
    # Save Results
    # --------------------------------------------------------

    save_results(
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        test_losses=test_losses,
        test_accuracies=test_accuracies,
        test_precision=test_precision,
        test_recall=test_recall,
        test_f1=test_f1,
        confusion_mat=confusion_mat,
        classification_rep=classification_rep,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )


    print(
        "\n" + "=" * 70
    )


    print(
        "TRAINING COMPLETED"
    )


    print(
        "=" * 70
    )