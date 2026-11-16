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
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# IMPORT DATASET LOADER
# ============================================================

from dataset.dataset_loader import get_data_loaders


# ============================================================
# IMPORT DELTA-LIF MODEL
# ============================================================

from model.delta_lif_snn import DeltaLIFSNN


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
# RESULTS DIRECTORY
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

TRAIN_LOSS_FILE = os.path.join(
    RESULTS_DIR,
    "delta_lif_snn_total_loss.png"
)

TRAIN_ACCURACY_FILE = os.path.join(
    RESULTS_DIR,
    "delta_lif_snn_total_accuracy.png"
)

TEST_ACCURACY_FILE = os.path.join(
    RESULTS_DIR,
    "delta_lif_snn_test_accuracy.png"
)

TEST_LOSS_FILE = os.path.join(
    RESULTS_DIR,
    "delta_lif_snn_test_loss.png"
)

CONFUSION_MATRIX_FILE = os.path.join(
    RESULTS_DIR,
    "delta_lif_snn_confusion_matrix.png"
)

CONFUSION_MATRIX_CSV = os.path.join(
    RESULTS_DIR,
    "delta_lif_snn_confusion_matrix.csv"
)

METRICS_CSV_FILE = os.path.join(
    RESULTS_DIR,
    "delta_lif_snn_metrics.csv"
)

CLASSIFICATION_REPORT_FILE = os.path.join(
    RESULTS_DIR,
    "delta_lif_snn_classification_report.txt"
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "delta_lif_snn_summary.txt"
)

CSV_FILE = os.path.join(
    RESULTS_DIR,
    "delta_lif_snn_training_log.csv"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "delta_lif_snn_model.pth"
)


# ============================================================
# LOSS FUNCTION
# ============================================================

def spike_rate_loss(
    output_spikes,
    labels
):

    # --------------------------------------------------------
    # Sum spikes over time
    # --------------------------------------------------------

    spike_counts = output_spikes.sum(
        dim=0
    )

    # --------------------------------------------------------
    # Classification loss
    # --------------------------------------------------------

    loss = nn.CrossEntropyLoss()(
        spike_counts,
        labels
    )

    return loss


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    test_loader,
    device
):

    model.eval()

    correct = 0

    total = 0

    total_loss = 0.0

    batches = 0

    all_labels = []

    all_predictions = []

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # Test progress bar
    # --------------------------------------------------------

    test_progress = tqdm(
        test_loader,
        desc="Testing",
        unit="batch",
        leave=True
    )


    with torch.no_grad():

        for data, labels in test_progress:

            data = data.to(
                device
            )

            labels = labels.to(
                device
            )


            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            output = model(
                data
            )


            # ------------------------------------------------
            # Sum spikes over time
            # ------------------------------------------------

            spike_counts = output.sum(
                dim=0
            )


            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                spike_counts,
                labels
            )


            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            predictions = spike_counts.argmax(
                dim=1
            )


            # ------------------------------------------------
            # Accuracy statistics
            # ------------------------------------------------

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            batches += labels.size(0)


            # ------------------------------------------------
            # Store labels and predictions
            # ------------------------------------------------

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )


            # ------------------------------------------------
            # Current test accuracy
            # ------------------------------------------------

            current_accuracy = (
                100.0
                * correct
                / total
            )


            test_progress.set_postfix(
                loss=f"{loss.item():.4f}",
                accuracy=f"{current_accuracy:.2f}%"
            )


    test_progress.close()


    # ========================================================
    # TEST METRICS
    # ========================================================

    average_loss = (
        total_loss
        / batches
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
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


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES))
    )


    return (
        average_loss,
        accuracy * 100.0,
        precision * 100.0,
        recall * 100.0,
        f1 * 100.0,
        cm,
        all_labels,
        all_predictions
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 70)

    print(
        "SINGLE-CLIENT DELTA-LIF SNN TRAINING"
    )

    print("=" * 70)


    # ========================================================
    # PROJECT INFORMATION
    # ========================================================

    print()

    print(
        "Project root:"
    )

    print(
        PROJECT_ROOT
    )


    print()

    print(
        "Execution device:"
    )

    print(
        DEVICE
    )


    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # ========================================================
    # LOAD N-MNIST
    # ========================================================

    print()

    print(
        "Loading N-MNIST..."
    )


    (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader
    ) = get_data_loaders(
        batch_size=BATCH_SIZE
    )


    print()

    print(
        "Dataset loaded successfully."
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


    # ========================================================
    # CREATE DELTA-LIF MODEL
    # ========================================================

    model = DeltaLIFSNN(

        input_channels=2,

        height=34,

        width=34,

        hidden_size=128,

        num_classes=NUM_CLASSES,

        threshold=1.0

    ).to(DEVICE)


    print()

    print(
        "Delta-LIF SNN model created successfully."
    )


    # ========================================================
    # MODEL PARAMETERS
    # ========================================================

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )


    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


    print()

    print(
        "Total parameters:",
        total_parameters
    )

    print(
        "Trainable parameters:",
        trainable_parameters
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    criterion = nn.CrossEntropyLoss()


    print()

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
    # TRAINING HISTORY
    # ========================================================

    epoch_losses = []

    epoch_accuracies = []

    test_losses = []

    test_accuracies = []

    test_precisions = []

    test_recalls = []

    test_f1_scores = []

    csv_rows = []


    # ========================================================
    # START TRAINING
    # ========================================================

    print()

    print(
        "Starting training..."
    )


    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        LOCAL_EPOCHS
    ):

        model.train()


        running_loss = 0.0

        correct = 0

        total = 0


        # ----------------------------------------------------
        # Training progress bar
        # ----------------------------------------------------

        train_progress = tqdm(

            train_loader,

            desc=(
                f"Epoch {epoch + 1}/"
                f"{LOCAL_EPOCHS}"
            ),

            unit="batch",

            leave=True

        )


        # ====================================================
        # BATCH LOOP
        # ====================================================

        for data, labels in train_progress:

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
            # Calculate loss
            # ------------------------------------------------

            loss = spike_rate_loss(
                output,
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
            # Statistics
            # ------------------------------------------------

            batch_size = labels.size(0)


            running_loss += (
                loss.item()
                * batch_size
            )


            spike_counts = (
                output.detach()
                .sum(
                    dim=0
                )
            )


            predictions = spike_counts.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += batch_size


            # ------------------------------------------------
            # Current training metrics
            # ------------------------------------------------

            current_loss = (
                running_loss
                / total
            )


            current_accuracy = (
                100.0
                * correct
                / total
            )


            train_progress.set_postfix(

                loss=f"{current_loss:.4f}",

                accuracy=f"{current_accuracy:.2f}%"

            )


        train_progress.close()


        # ====================================================
        # EPOCH TRAINING RESULTS
        # ====================================================

        epoch_loss = (
            running_loss
            / total
        )


        epoch_accuracy = (
            100.0
            * correct
            / total
        )


        epoch_losses.append(
            epoch_loss
        )


        epoch_accuracies.append(
            epoch_accuracy
        )


        # ====================================================
        # TEST AFTER EACH EPOCH
        # ====================================================

        print()

        print(
            f"Evaluating after Epoch "
            f"{epoch + 1}..."
        )


        (
            test_loss,
            test_accuracy,
            test_precision,
            test_recall,
            test_f1,
            confusion_mat,
            test_labels,
            test_predictions
        ) = evaluate(

            model,

            test_loader,

            DEVICE

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


        # ====================================================
        # SAVE EPOCH RESULTS
        # ====================================================

        csv_rows.append([

            epoch + 1,

            epoch_loss,

            epoch_accuracy,

            test_loss,

            test_accuracy,

            test_precision,

            test_recall,

            test_f1

        ])


        # ====================================================
        # PRINT EPOCH RESULTS
        # ====================================================

        print()

        print(
            f"Epoch [{epoch + 1}/"
            f"{LOCAL_EPOCHS}]"
        )

        print(
            f"Train Loss     : "
            f"{epoch_loss:.4f}"
        )

        print(
            f"Train Accuracy : "
            f"{epoch_accuracy:.2f}%"
        )

        print(
            f"Test Loss      : "
            f"{test_loss:.4f}"
        )

        print(
            f"Test Accuracy  : "
            f"{test_accuracy:.2f}%"
        )

        print(
            f"Precision      : "
            f"{test_precision:.2f}%"
        )

        print(
            f"Recall         : "
            f"{test_recall:.2f}%"
        )

        print(
            f"F1 Score       : "
            f"{test_f1:.2f}%"
        )

        print(
            "-" * 70
        )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    final_train_loss = (
        epoch_losses[-1]
    )

    final_train_accuracy = (
        epoch_accuracies[-1]
    )

    final_test_loss = (
        test_losses[-1]
    )

    final_test_accuracy = (
        test_accuracies[-1]
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
    # FINAL CONFUSION MATRIX
    # ========================================================

    final_confusion_matrix = confusion_mat


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

    ) as f:

        writer = csv.writer(f)


        writer.writerow([

            "Epoch",

            "Train Loss",

            "Train Accuracy",

            "Test Loss",

            "Test Accuracy",

            "Precision",

            "Recall",

            "F1 Score"

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

    ) as f:

        writer = csv.writer(f)


        writer.writerow([

            "Metric",

            "Value (%)"

        ])


        writer.writerow([
            "Accuracy",
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


    # ========================================================
    # SAVE CONFUSION MATRIX CSV
    # ========================================================

    with open(

        CONFUSION_MATRIX_CSV,

        "w",

        newline="",

        encoding="utf-8"

    ) as f:

        writer = csv.writer(f)


        writer.writerow([

            "Actual / Predicted",

            "0",

            "1",

            "2",

            "3",

            "4",

            "5",

            "6",

            "7",

            "8",

            "9"

        ])


        for i, row in enumerate(
            final_confusion_matrix
        ):

            writer.writerow(

                [i]
                + list(row)

            )


    # ========================================================
    # SAVE CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(

        test_labels,

        test_predictions,

        labels=list(range(NUM_CLASSES)),

        target_names=[
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9"
        ],

        digits=4,

        zero_division=0

    )


    with open(

        CLASSIFICATION_REPORT_FILE,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(
            "DELTA-LIF SNN CLASSIFICATION REPORT\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            report
        )


    # ========================================================
    # SAVE TRAINING LOSS GRAPH
    # ========================================================

    plt.figure()


    plt.plot(

        range(
            1,
            LOCAL_EPOCHS + 1
        ),

        epoch_losses,

        marker="o"

    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Training Loss"
    )


    plt.title(
        "Delta-LIF SNN Training Loss"
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
    # SAVE TRAINING ACCURACY GRAPH
    # ========================================================

    plt.figure()


    plt.plot(

        range(
            1,
            LOCAL_EPOCHS + 1
        ),

        epoch_accuracies,

        marker="o"

    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Training Accuracy (%)"
    )


    plt.title(
        "Delta-LIF SNN Training Accuracy"
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
    # SAVE TEST ACCURACY GRAPH
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
        "Delta-LIF SNN Test Accuracy"
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
    # SAVE TEST LOSS GRAPH
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
        "Delta-LIF SNN Test Loss"
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
    # SAVE CONFUSION MATRIX
    # LIGHT BLUE AND WHITE
    # ========================================================

    plt.figure(
        figsize=(8, 7)
    )


    plt.imshow(

        final_confusion_matrix,

        interpolation="nearest",

        cmap="Blues"

    )


    plt.title(
        "Delta-LIF SNN Confusion Matrix"
    )


    plt.colorbar()


    tick_marks = range(
        NUM_CLASSES
    )


    plt.xticks(

        tick_marks,

        [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9"
        ]

    )


    plt.yticks(

        tick_marks,

        [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9"
        ]

    )


    # --------------------------------------------------------
    # Add values inside confusion matrix
    # --------------------------------------------------------

    threshold = (
        final_confusion_matrix.max()
        / 2.0
    )


    for i in range(
        final_confusion_matrix.shape[0]
    ):

        for j in range(
            final_confusion_matrix.shape[1]
        ):

            plt.text(

                j,

                i,

                format(
                    final_confusion_matrix[i, j],
                    "d"
                ),

                ha="center",

                va="center",

                color=(
                    "white"
                    if final_confusion_matrix[i, j]
                    > threshold
                    else "black"
                )

            )


    plt.ylabel(
        "Actual Class"
    )


    plt.xlabel(
        "Predicted Class"
    )


    plt.tight_layout()


    plt.savefig(

        CONFUSION_MATRIX_FILE,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    with open(

        SUMMARY_FILE,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "SINGLE-CLIENT DELTA-LIF SNN TRAINING SUMMARY\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )


        f.write(
            f"Device: {DEVICE}\n"
        )


        if torch.cuda.is_available():

            f.write(
                "GPU: "
                + torch.cuda.get_device_name(0)
                + "\n"
            )


        f.write(
            "Dataset: N-MNIST\n"
        )


        f.write(
            "Clients: 1\n"
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
            "Optimizer: Adam\n"
        )


        f.write(
            f"Learning rate: "
            f"{LEARNING_RATE}\n"
        )


        f.write(
            f"Total parameters: "
            f"{total_parameters}\n"
        )


        f.write(
            f"Trainable parameters: "
            f"{trainable_parameters}\n\n"
        )


        # ----------------------------------------------------
        # Epoch results
        # ----------------------------------------------------

        f.write(
            "Epoch Results\n"
        )

        f.write(
            "-" * 70 + "\n"
        )


        for i in range(
            LOCAL_EPOCHS
        ):

            f.write(

                f"Epoch {i + 1}: "

                f"Train Loss = "
                f"{epoch_losses[i]:.4f}, "

                f"Train Accuracy = "
                f"{epoch_accuracies[i]:.2f}%, "

                f"Test Loss = "
                f"{test_losses[i]:.4f}, "

                f"Test Accuracy = "
                f"{test_accuracies[i]:.2f}%, "

                f"Precision = "
                f"{test_precisions[i]:.2f}%, "

                f"Recall = "
                f"{test_recalls[i]:.2f}%, "

                f"F1 Score = "
                f"{test_f1_scores[i]:.2f}%\n"

            )


        f.write(
            "\n"
        )


        # ----------------------------------------------------
        # Final results
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
            f"Final Train Loss: "
            f"{final_train_loss:.4f}\n"
        )


        f.write(
            f"Final Train Accuracy: "
            f"{final_train_accuracy:.2f}%\n"
        )


        f.write(
            f"Final Test Loss: "
            f"{final_test_loss:.4f}\n"
        )


        f.write(
            f"Final Test Accuracy: "
            f"{final_test_accuracy:.2f}%\n"
        )


        f.write(
            f"Final Precision: "
            f"{final_precision:.2f}%\n"
        )


        f.write(
            f"Final Recall: "
            f"{final_recall:.2f}%\n"
        )


        f.write(
            f"Final F1 Score: "
            f"{final_f1:.2f}%\n\n"
        )


        # ----------------------------------------------------
        # Classification report
        # ----------------------------------------------------

        f.write(
            "Classification Report\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            report
        )

        f.write(
            "\n\n"
        )


        # ----------------------------------------------------
        # Output files
        # ----------------------------------------------------

        f.write(
            "Output Files\n"
        )

        f.write(
            "-" * 70 + "\n"
        )


        f.write(
            f"Training Loss Graph: "
            f"{TRAIN_LOSS_FILE}\n"
        )


        f.write(
            f"Training Accuracy Graph: "
            f"{TRAIN_ACCURACY_FILE}\n"
        )


        f.write(
            f"Test Accuracy Graph: "
            f"{TEST_ACCURACY_FILE}\n"
        )


        f.write(
            f"Test Loss Graph: "
            f"{TEST_LOSS_FILE}\n"
        )


        f.write(
            f"Confusion Matrix: "
            f"{CONFUSION_MATRIX_FILE}\n"
        )


        f.write(
            f"Confusion Matrix CSV: "
            f"{CONFUSION_MATRIX_CSV}\n"
        )


        f.write(
            f"Metrics CSV: "
            f"{METRICS_CSV_FILE}\n"
        )


        f.write(
            f"Classification Report: "
            f"{CLASSIFICATION_REPORT_FILE}\n"
        )


        f.write(
            f"Training CSV Log: "
            f"{CSV_FILE}\n"
        )


        f.write(
            f"Model: "
            f"{MODEL_FILE}\n"
        )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "FINAL DELTA-LIF SNN RESULTS"
    )

    print(
        "=" * 70
    )


    print(
        f"Final Train Loss     : "
        f"{final_train_loss:.4f}"
    )


    print(
        f"Final Train Accuracy : "
        f"{final_train_accuracy:.2f}%"
    )


    print(
        f"Final Test Loss      : "
        f"{final_test_loss:.4f}"
    )


    print(
        f"Final Test Accuracy  : "
        f"{final_test_accuracy:.2f}%"
    )


    print(
        f"Final Precision      : "
        f"{final_precision:.2f}%"
    )


    print(
        f"Final Recall         : "
        f"{final_recall:.2f}%"
    )


    print(
        f"Final F1 Score       : "
        f"{final_f1:.2f}%"
    )


    print()

    print(
        "Final Confusion Matrix:"
    )

    print(
        final_confusion_matrix
    )


    print()

    print(
        "Results saved:"
    )


    print(
        TRAIN_LOSS_FILE
    )


    print(
        TRAIN_ACCURACY_FILE
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

    print(
        "=" * 70
    )

    print(
        "SINGLE-CLIENT DELTA-LIF TRAINING COMPLETED"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()