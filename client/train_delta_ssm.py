import os
import sys
import csv

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


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT DATASET
# ============================================================

from dataset.dataset_loader import get_data_loaders


# ============================================================
# IMPORT MODEL
# ============================================================

from model.delta_ssm_snn import DeltaSSMSNN


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.001

NUM_CLASSES = 10


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# RESULT DIRECTORIES
# ============================================================

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "phase1"
)

MODEL_DIR = os.path.join(
    RESULTS_DIR,
    "models"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

TRAIN_ACCURACY_FILE = os.path.join(
    RESULTS_DIR,
    "delta_ssm_final_train_accuracy.png"
)

TRAIN_LOSS_FILE = os.path.join(
    RESULTS_DIR,
    "delta_ssm_final_train_loss.png"
)

TEST_ACCURACY_FILE = os.path.join(
    RESULTS_DIR,
    "delta_ssm_final_test_accuracy.png"
)

TEST_LOSS_FILE = os.path.join(
    RESULTS_DIR,
    "delta_ssm_final_test_loss.png"
)

CONFUSION_MATRIX_FILE = os.path.join(
    RESULTS_DIR,
    "delta_ssm_confusion_matrix.png"
)

CONFUSION_MATRIX_CSV = os.path.join(
    RESULTS_DIR,
    "delta_ssm_confusion_matrix.csv"
)

METRICS_CSV_FILE = os.path.join(
    RESULTS_DIR,
    "delta_ssm_metrics.csv"
)

CLASSIFICATION_REPORT_FILE = os.path.join(
    RESULTS_DIR,
    "delta_ssm_classification_report.txt"
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "delta_ssm_summary.txt"
)

CSV_FILE = os.path.join(
    RESULTS_DIR,
    "delta_ssm_training_log.csv"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "delta_ssm_model.pth"
)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# TEST / EVALUATION FUNCTION
# ============================================================

def evaluate(
    model,
    test_loader
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_labels = []

    test_progress = tqdm(
        test_loader,
        desc="Testing",
        unit="batch",
        leave=False
    )

    with torch.no_grad():

        for data, labels in test_progress:

            # ------------------------------------------------
            # Move data to device
            # ------------------------------------------------

            data = data.to(DEVICE)
            labels = labels.to(DEVICE)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            output = model(
                data
            )

            # ------------------------------------------------
            # Average output over time
            # ------------------------------------------------

            logits = output.mean(
                dim=0
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                logits,
                labels
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            predictions = logits.argmax(
                dim=1
            )

            # ------------------------------------------------
            # Store predictions and labels
            # ------------------------------------------------

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            # ------------------------------------------------
            # Sample-weighted loss
            # ------------------------------------------------

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            # ------------------------------------------------
            # Current accuracy
            # ------------------------------------------------

            current_accuracy = (
                100.0
                * accuracy_score(
                    all_labels,
                    all_predictions
                )
            )

            test_progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{current_accuracy:.2f}%"
            )

    # ========================================================
    # FINAL TEST LOSS
    # ========================================================

    average_loss = (
        total_loss / total_samples
    )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    accuracy = (
        100.0
        * accuracy_score(
            all_labels,
            all_predictions
        )
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES))
    )

    return (
        average_loss,
        accuracy,
        precision,
        recall,
        f1,
        cm,
        all_labels,
        all_predictions
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SINGLE-CLIENT DELTA-SSM SNN TRAINING")
    print("=" * 60)

    # ========================================================
    # PROJECT INFORMATION
    # ========================================================

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Python execution device:")
    print(DEVICE)

    # --------------------------------------------------------
    # GPU INFORMATION
    # --------------------------------------------------------

    if torch.cuda.is_available():

        print("GPU:")

        print(
            torch.cuda.get_device_name(0)
        )

        print("CUDA:")

        print(
            torch.version.cuda
        )

    else:

        print("GPU: Not available")

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print()
    print("Loading N-MNIST...")

    (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader
    ) = get_data_loaders(
        batch_size=BATCH_SIZE
    )

    print()
    print("Dataset loaded successfully.")

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

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = DeltaSSMSNN(
        input_channels=2,
        height=34,
        width=34,
        hidden_size=128,
        num_classes=NUM_CLASSES,
        threshold=1.0
    ).to(DEVICE)

    print()
    print(
        "Delta-SSM SNN model created successfully."
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(
        "Optimizer: Adam"
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

    print(
        "Local epochs:",
        LOCAL_EPOCHS
    )

    # ========================================================
    # HISTORY
    # ========================================================

    train_losses = []
    train_accuracies = []

    test_losses = []
    test_accuracies = []

    test_precisions = []
    test_recalls = []
    test_f1_scores = []

    csv_rows = []

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("Starting training...")

    for epoch in range(
        LOCAL_EPOCHS
    ):

        model.train()

        running_loss = 0.0

        correct = 0
        total = 0

        total_batches = len(
            train_loader
        )

        # ----------------------------------------------------
        # TRAINING PROGRESS BAR
        # ----------------------------------------------------

        progress_bar = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch + 1}/{LOCAL_EPOCHS}"
            ),
            unit="batch"
        )

        for data, labels in progress_bar:

            # ------------------------------------------------
            # Move data to GPU/CPU
            # ------------------------------------------------

            data = data.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            # ------------------------------------------------
            # Clear gradients
            # ------------------------------------------------

            optimizer.zero_grad()

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            output = model(
                data
            )

            # ------------------------------------------------
            # Average temporal outputs
            # ------------------------------------------------

            logits = output.mean(
                dim=0
            )

            # ------------------------------------------------
            # Calculate loss
            # ------------------------------------------------

            loss = criterion(
                logits,
                labels
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Update parameters
            # ------------------------------------------------

            optimizer.step()

            # ------------------------------------------------
            # Training statistics
            # ------------------------------------------------

            running_loss += (
                loss.item()
            )

            predictions = (
                logits.detach()
                .argmax(dim=1)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += (
                labels.size(0)
            )

            # ------------------------------------------------
            # Current batch accuracy
            # ------------------------------------------------

            current_accuracy = (
                100.0
                * correct
                / total
            )

            # ------------------------------------------------
            # Update tqdm
            # ------------------------------------------------

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{current_accuracy:.2f}%"
            )

        # ====================================================
        # TRAINING METRICS
        # ====================================================

        train_loss = (
            running_loss
            / total_batches
        )

        train_accuracy = (
            100.0
            * correct
            / total
        )

        # ====================================================
        # TEST AFTER EACH EPOCH
        # ====================================================

        (
            test_loss,
            test_accuracy,
            test_precision,
            test_recall,
            test_f1,
            test_cm,
            epoch_labels,
            epoch_predictions
        ) = evaluate(
            model,
            test_loader
        )

        # ====================================================
        # STORE RESULTS
        # ====================================================

        train_losses.append(
            train_loss
        )

        train_accuracies.append(
            train_accuracy
        )

        test_losses.append(
            test_loss
        )

        test_accuracies.append(
            test_accuracy
        )

        test_precisions.append(
            test_precision
        )

        test_recalls.append(
            test_recall
        )

        test_f1_scores.append(
            test_f1
        )

        csv_rows.append([
            epoch + 1,
            train_loss,
            train_accuracy,
            test_loss,
            test_accuracy,
            test_precision,
            test_recall,
            test_f1
        ])

        # ====================================================
        # EPOCH OUTPUT
        # ====================================================

        print()

        print(
            f"Epoch [{epoch + 1}/"
            f"{LOCAL_EPOCHS}]"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.2f}%"
        )

        print(
            f"Test Loss: "
            f"{test_loss:.4f}"
        )

        print(
            f"Test Accuracy: "
            f"{test_accuracy:.2f}%"
        )

        print(
            f"Test Precision: "
            f"{test_precision:.4f}"
        )

        print(
            f"Test Recall: "
            f"{test_recall:.4f}"
        )

        print(
            f"Test F1 Score: "
            f"{test_f1:.4f}"
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    final_train_accuracy = (
        train_accuracies[-1]
    )

    final_train_loss = (
        train_losses[-1]
    )

    final_test_accuracy = (
        test_accuracies[-1]
    )

    final_test_loss = (
        test_losses[-1]
    )

    final_precision = (
        test_precisions[-1]
    )

    final_recall = (
        test_recalls[-1]
    )

    final_f1 = (
        test_f1_scores[-1]
    )

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    (
        final_eval_loss,
        final_eval_accuracy,
        final_precision,
        final_recall,
        final_f1,
        final_cm,
        final_labels,
        final_predictions
    ) = evaluate(
        model,
        test_loader
    )

    final_test_loss = final_eval_loss
    final_test_accuracy = final_eval_accuracy

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print()

    print(
        f"Final Train Accuracy: "
        f"{final_train_accuracy:.2f}%"
    )

    print(
        f"Final Train Loss: "
        f"{final_train_loss:.4f}"
    )

    print()

    print(
        f"Final Test Accuracy: "
        f"{final_test_accuracy:.2f}%"
    )

    print(
        f"Final Test Loss: "
        f"{final_test_loss:.4f}"
    )

    print(
        f"Final Precision: "
        f"{final_precision:.4f}"
    )

    print(
        f"Final Recall: "
        f"{final_recall:.4f}"
    )

    print(
        f"Final F1 Score: "
        f"{final_f1:.4f}"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    torch.save(
        model.state_dict(),
        MODEL_FILE
    )

    # ========================================================
    # SAVE TRAINING CSV
    # ========================================================

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "Epoch",
            "Train Loss",
            "Train Accuracy",
            "Test Loss",
            "Test Accuracy",
            "Test Precision",
            "Test Recall",
            "Test F1 Score"
        ])

        writer.writerows(
            csv_rows
        )

    # ========================================================
    # SAVE FINAL METRICS CSV
    # ========================================================

    with open(
        METRICS_CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "Metric",
            "Value"
        ])

        writer.writerow([
            "Accuracy (%)",
            final_test_accuracy
        ])

        writer.writerow([
            "Precision",
            final_precision
        ])

        writer.writerow([
            "Recall",
            final_recall
        ])

        writer.writerow([
            "F1 Score",
            final_f1
        ])

        writer.writerow([
            "Test Loss",
            final_test_loss
        ])

    # ========================================================
    # SAVE CONFUSION MATRIX CSV
    # ========================================================

    with open(
        CONFUSION_MATRIX_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "True/Predicted"
        ] + list(range(NUM_CLASSES)))

        for i, row in enumerate(
            final_cm
        ):

            writer.writerow(
                [i] + list(row)
            )

    # ========================================================
    # CONFUSION MATRIX GRAPH
    # ========================================================

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        final_cm,
        interpolation="nearest",
        cmap="Blues"
    )

    plt.title(
        "Delta-SSM Confusion Matrix"
    )

    plt.colorbar()

    tick_marks = range(
        NUM_CLASSES
    )

    plt.xticks(
        tick_marks,
        tick_marks
    )

    plt.yticks(
        tick_marks,
        tick_marks
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    # --------------------------------------------------------
    # Write values inside confusion matrix
    # --------------------------------------------------------

    threshold = (
        final_cm.max() / 2.0
    )

    for i in range(
        NUM_CLASSES
    ):

        for j in range(
            NUM_CLASSES
        ):

            plt.text(
                j,
                i,
                str(final_cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if final_cm[i, j] > threshold
                    else "black"
                )
            )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=300
    )

    plt.close()

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        final_labels,
        final_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=[
            str(i)
            for i in range(NUM_CLASSES)
        ],
        zero_division=0
    )

    with open(
        CLASSIFICATION_REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "DELTA-SSM SNN CLASSIFICATION REPORT\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            report
        )

    # ========================================================
    # TRAIN ACCURACY GRAPH
    # ========================================================

    plt.figure()

    plt.plot(
        range(
            1,
            LOCAL_EPOCHS + 1
        ),
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
        "Delta-SSM Training Accuracy"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        TRAIN_ACCURACY_FILE,
        dpi=300
    )

    plt.close()

    # ========================================================
    # TRAIN LOSS GRAPH
    # ========================================================

    plt.figure()

    plt.plot(
        range(
            1,
            LOCAL_EPOCHS + 1
        ),
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
        "Delta-SSM Training Loss"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        TRAIN_LOSS_FILE,
        dpi=300
    )

    plt.close()

    # ========================================================
    # TEST ACCURACY GRAPH
    # ========================================================

    plt.figure()

    plt.plot(
        range(
            1,
            LOCAL_EPOCHS + 1
        ),
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
        "Delta-SSM Test Accuracy"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        TEST_ACCURACY_FILE,
        dpi=300
    )

    plt.close()

    # ========================================================
    # TEST LOSS GRAPH
    # ========================================================

    plt.figure()

    plt.plot(
        range(
            1,
            LOCAL_EPOCHS + 1
        ),
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
        "Delta-SSM Test Loss"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        TEST_LOSS_FILE,
        dpi=300
    )

    plt.close()

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "=" * 60 + "\n"
        )

        file.write(
            "SINGLE-CLIENT DELTA-SSM SNN TRAINING\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Device: {DEVICE}\n"
        )

        if torch.cuda.is_available():

            file.write(
                f"GPU: "
                f"{torch.cuda.get_device_name(0)}\n"
            )

            file.write(
                f"CUDA: "
                f"{torch.version.cuda}\n"
            )

        else:

            file.write(
                "GPU: Not available\n"
            )

        file.write(
            "Clients: 1\n"
        )

        file.write(
            f"Training samples: "
            f"{len(train_dataset)}\n"
        )

        file.write(
            f"Testing samples: "
            f"{len(test_dataset)}\n"
        )

        file.write(
            f"Batch size: "
            f"{BATCH_SIZE}\n"
        )

        file.write(
            f"Local epochs: "
            f"{LOCAL_EPOCHS}\n"
        )

        file.write(
            "Optimizer: Adam\n"
        )

        file.write(
            f"Learning rate: "
            f"{LEARNING_RATE}\n"
        )

        file.write(
            "Threshold: 1.0\n\n"
        )

        # ----------------------------------------------------
        # Epoch results
        # ----------------------------------------------------

        file.write(
            "EPOCH RESULTS\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        for i in range(
            LOCAL_EPOCHS
        ):

            file.write(
                f"Epoch {i + 1}: "
                f"Train Loss = "
                f"{train_losses[i]:.4f}, "
                f"Train Accuracy = "
                f"{train_accuracies[i]:.2f}%, "
                f"Test Loss = "
                f"{test_losses[i]:.4f}, "
                f"Test Accuracy = "
                f"{test_accuracies[i]:.2f}%, "
                f"Precision = "
                f"{test_precisions[i]:.4f}, "
                f"Recall = "
                f"{test_recalls[i]:.4f}, "
                f"F1 = "
                f"{test_f1_scores[i]:.4f}\n"
            )

        file.write(
            "\n"
        )

        # ----------------------------------------------------
        # Final results
        # ----------------------------------------------------

        file.write(
            "FINAL RESULTS\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            f"Final Train Accuracy: "
            f"{final_train_accuracy:.2f}%\n"
        )

        file.write(
            f"Final Train Loss: "
            f"{final_train_loss:.4f}\n"
        )

        file.write(
            f"Final Test Accuracy: "
            f"{final_test_accuracy:.2f}%\n"
        )

        file.write(
            f"Final Test Loss: "
            f"{final_test_loss:.4f}\n"
        )

        file.write(
            f"Final Precision: "
            f"{final_precision:.4f}\n"
        )

        file.write(
            f"Final Recall: "
            f"{final_recall:.4f}\n"
        )

        file.write(
            f"Final F1 Score: "
            f"{final_f1:.4f}\n"
        )

    # ========================================================
    # SAVED FILES
    # ========================================================

    print()
    print("Results saved:")

    print(
        TRAIN_ACCURACY_FILE
    )

    print(
        TRAIN_LOSS_FILE
    )

    print(
        TEST_ACCURACY_FILE
    )

    print(
        TEST_LOSS_FILE
    )

    print(
        CONFUSION_MATRIX_FILE
    )

    print(
        CONFUSION_MATRIX_CSV
    )

    print(
        METRICS_CSV_FILE
    )

    print(
        CLASSIFICATION_REPORT_FILE
    )

    print(
        SUMMARY_FILE
    )

    print(
        CSV_FILE
    )

    print(
        MODEL_FILE
    )

    print()
    print("=" * 60)

    print(
        "SINGLE-CLIENT DELTA-SSM TRAINING COMPLETED"
    )

    print("=" * 60)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()