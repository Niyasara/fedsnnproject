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
# IMPORT DATASET
# ============================================================

from dataset.dataset_loader import get_data_loaders


# ============================================================
# IMPORT SSM MODEL
# ============================================================

from model.ssm_snn import SSMSNN


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

TRAIN_ACCURACY_FILE = os.path.join(
    RESULTS_DIR,
    "ssm_snn_total_train_accuracy.png"
)

TEST_ACCURACY_FILE = os.path.join(
    RESULTS_DIR,
    "ssm_snn_total_test_accuracy.png"
)

TRAIN_LOSS_FILE = os.path.join(
    RESULTS_DIR,
    "ssm_snn_total_train_loss.png"
)

TEST_LOSS_FILE = os.path.join(
    RESULTS_DIR,
    "ssm_snn_total_test_loss.png"
)

CONFUSION_MATRIX_FILE = os.path.join(
    RESULTS_DIR,
    "ssm_snn_confusion_matrix.png"
)

CONFUSION_MATRIX_CSV = os.path.join(
    RESULTS_DIR,
    "ssm_snn_confusion_matrix.csv"
)

METRICS_CSV_FILE = os.path.join(
    RESULTS_DIR,
    "ssm_snn_metrics.csv"
)

CLASSIFICATION_REPORT_FILE = os.path.join(
    RESULTS_DIR,
    "ssm_snn_classification_report.txt"
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "ssm_snn_summary.txt"
)

CSV_FILE = os.path.join(
    RESULTS_DIR,
    "ssm_snn_training_log.csv"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "ssm_snn_model.pth"
)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    model,
    test_loader,
    device
):

    model.eval()

    total_loss = 0.0

    correct = 0

    total = 0

    batches = 0

    all_labels = []

    all_predictions = []


    # ========================================================
    # TEST LOOP
    # ========================================================

    with torch.no_grad():

        for data, labels in tqdm(
            test_loader,
            desc="Testing",
            unit="batch",
            leave=True
        ):

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
            # Average output across time
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
            # Prediction
            # ------------------------------------------------

            predictions = logits.argmax(
                dim=1
            )


            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

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


            # ------------------------------------------------
            # Update progress bar
            # ------------------------------------------------

            test_progress_text = (
                f"loss={loss.item():.4f}, "
                f"accuracy={current_accuracy:.2f}%"
            )


    # ========================================================
    # AVERAGE TEST LOSS
    # ========================================================

    average_loss = (
        total_loss
        / batches
    )


    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    # ========================================================
    # PRECISION
    # ========================================================

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )


    # ========================================================
    # RECALL
    # ========================================================

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )


    # ========================================================
    # F1 SCORE
    # ========================================================

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
        "SINGLE-CLIENT SSM SNN TRAINING"
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
        "Python execution device:"
    )

    print(
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

    else:

        print(
            "GPU: Not available"
        )


    # ========================================================
    # LOAD DATASET
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
    # CREATE SSM MODEL
    # ========================================================

    model = SSMSNN(

        input_channels=2,

        height=34,

        width=34,

        hidden_size=128,

        num_classes=NUM_CLASSES

    ).to(DEVICE)


    print()

    print(
        "SSM SNN model created successfully."
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

    train_losses = []

    train_accuracies = []

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


        total_batches = len(
            train_loader
        )


        # ----------------------------------------------------
        # TRAINING PROGRESS BAR
        # ----------------------------------------------------

        progress_bar = tqdm(

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

        for data, labels in progress_bar:


            # ------------------------------------------------
            # Move data to device
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
            # Average output across time
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

            batch_size = labels.size(0)


            running_loss += (
                loss.item()
                * batch_size
            )


            predictions = logits.detach().argmax(
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


            progress_bar.set_postfix(

                loss=f"{current_loss:.4f}",

                accuracy=f"{current_accuracy:.2f}%"

            )


        progress_bar.close()


        # ====================================================
        # TRAINING METRICS
        # ====================================================

        train_loss = (
            running_loss
            / total
        )


        train_accuracy = (
            100.0
            * correct
            / total
        )


        # ====================================================
        # TESTING AFTER EACH EPOCH
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


        # ====================================================
        # STORE HISTORY
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


        # ====================================================
        # CSV ROW
        # ====================================================

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
            f"Train Loss     : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy : "
            f"{train_accuracy:.2f}%"
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
        train_losses[-1]
    )

    final_train_accuracy = (
        train_accuracies[-1]
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

    final_confusion_matrix = (
        confusion_mat
    )


    # ========================================================
    # FINAL RESULTS ON TERMINAL
    # ========================================================

    print()

    print("=" * 70)

    print(
        "FINAL RESULTS"
    )

    print("=" * 70)

    print()

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


        writer = csv.writer(
            f
        )


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


        writer = csv.writer(
            f
        )


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


        writer = csv.writer(
            f
        )


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

                [i] + list(row)

            )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(

        test_labels,

        test_predictions,

        labels=list(
            range(NUM_CLASSES)
        ),

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


    # ========================================================
    # SAVE CLASSIFICATION REPORT
    # ========================================================

    with open(

        CLASSIFICATION_REPORT_FILE,

        "w",

        encoding="utf-8"

    ) as f:


        f.write(
            "SSM SNN CLASSIFICATION REPORT\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
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
        "SSM SNN Training Accuracy"
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
        "SSM SNN Test Accuracy"
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
        "SSM SNN Training Loss"
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
        "SSM SNN Test Loss"
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
    # CONFUSION MATRIX
    # WHITE + LIGHT BLUE
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
        "SSM SNN Confusion Matrix"
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


    # ========================================================
    # ADD VALUES TO CONFUSION MATRIX
    # ========================================================

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
            "SINGLE-CLIENT SSM SNN TRAINING SUMMARY\n"
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


        else:

            f.write(
                "GPU: Not available\n"
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


        # ====================================================
        # EPOCH RESULTS
        # ====================================================

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
                f"{train_losses[i]:.4f}, "

                f"Train Accuracy = "
                f"{train_accuracies[i]:.2f}%, "

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


        # ====================================================
        # FINAL RESULTS
        # ====================================================

        f.write(
            "FINAL RESULTS\n"
        )

        f.write(
            "=" * 70 + "\n"
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


        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

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


        # ====================================================
        # OUTPUT FILES
        # ====================================================

        f.write(
            "Output Files\n"
        )

        f.write(
            "-" * 70 + "\n"
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

            f"Training Loss Graph: "
            f"{TRAIN_LOSS_FILE}\n"

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

            f"Training Log: "
            f"{CSV_FILE}\n"

        )


        f.write(

            f"Model: "
            f"{MODEL_FILE}\n"

        )


    # ========================================================
    # RESULTS SAVED
    # ========================================================

    print()

    print("=" * 70)

    print(
        "RESULTS SAVED"
    )

    print("=" * 70)

    print()

    print(
        "Training Accuracy PNG:"
    )

    print(
        TRAIN_ACCURACY_FILE
    )

    print()

    print(
        "Test Accuracy PNG:"
    )

    print(
        TEST_ACCURACY_FILE
    )

    print()

    print(
        "Training Loss PNG:"
    )

    print(
        TRAIN_LOSS_FILE
    )

    print()

    print(
        "Test Loss PNG:"
    )

    print(
        TEST_LOSS_FILE
    )

    print()

    print(
        "Confusion Matrix PNG:"
    )

    print(
        CONFUSION_MATRIX_FILE
    )

    print()

    print(
        "Confusion Matrix CSV:"
    )

    print(
        CONFUSION_MATRIX_CSV
    )

    print()

    print(
        "Metrics CSV:"
    )

    print(
        METRICS_CSV_FILE
    )

    print()

    print(
        "Classification Report:"
    )

    print(
        CLASSIFICATION_REPORT_FILE
    )

    print()

    print(
        "Summary:"
    )

    print(
        SUMMARY_FILE
    )

    print()

    print(
        "Training CSV:"
    )

    print(
        CSV_FILE
    )

    print()

    print(
        "Model:"
    )

    print(
        MODEL_FILE
    )

    print()

    print("=" * 70)

    print(
        "SINGLE-CLIENT SSM TRAINING COMPLETED"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()