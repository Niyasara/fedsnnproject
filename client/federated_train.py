import os
import sys
import csv
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import flwr as fl
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


# ============================================================
# IMPORT MODEL AND DATASET
# ============================================================

from model.lif_snn import LIFSNN

from dataset.dataset_loader import (
    NMNISTDataset,
    DATA_ROOT
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLIENTS = 5

NUM_ROUNDS = 50

LOCAL_EPOCHS = 5

BATCH_SIZE = 32

LEARNING_RATE = 0.001

TIME_STEPS = 10

NUM_CLASSES = 10

SEED = 42


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
    "phase1",
    "federated_standard_lif"
)

MODELS_DIR = os.path.join(
    RESULTS_DIR,
    "models"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

CSV_FILE = os.path.join(
    RESULTS_DIR,
    "federated_standard_lif_results.csv"
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "federated_standard_lif_summary.txt"
)

MODEL_FILE = os.path.join(
    MODELS_DIR,
    "global_standard_lif_model.pth"
)

TRAIN_ACCURACY_PLOT = os.path.join(
    RESULTS_DIR,
    "federated_standard_lif_train_accuracy.png"
)

TRAIN_LOSS_PLOT = os.path.join(
    RESULTS_DIR,
    "federated_standard_lif_train_loss.png"
)

TEST_ACCURACY_PLOT = os.path.join(
    RESULTS_DIR,
    "federated_standard_lif_test_accuracy.png"
)

TEST_LOSS_PLOT = os.path.join(
    RESULTS_DIR,
    "federated_standard_lif_test_loss.png"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

ROUND_RESULTS = []

TOTAL_UPLOADS = 0

LAST_GLOBAL_PARAMETERS = None


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("EXPERIMENT 1: STANDARD-LIF FEDERATED LEARNING")
print("=" * 70)

print("\nDevice:")
print(DEVICE)

print("\nConfiguration:")
print(f"Clients              : {NUM_CLIENTS}")
print(f"Local epochs         : {LOCAL_EPOCHS}")
print(f"Communication rounds : {NUM_ROUNDS}")
print(f"Batch size           : {BATCH_SIZE}")
print(f"Learning rate        : {LEARNING_RATE}")
print("Optimizer            : Adam")
print("Aggregation          : FedAvg")


# ============================================================
# DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING N-MNIST DATASET")
print("=" * 70)

FULL_TRAIN_DATASET = NMNISTDataset(
    DATA_ROOT,
    train=True
)

FULL_TEST_DATASET = NMNISTDataset(
    DATA_ROOT,
    train=False
)

print("\nTotal training samples:")
print(len(FULL_TRAIN_DATASET))

print("Total testing samples:")
print(len(FULL_TEST_DATASET))


# ============================================================
# SPLIT TRAINING DATA INTO CLIENTS
# ============================================================

total_samples = len(
    FULL_TRAIN_DATASET
)

samples_per_client = (
    total_samples // NUM_CLIENTS
)

CLIENT_INDICES = []


for client_id in range(NUM_CLIENTS):

    start = (
        client_id *
        samples_per_client
    )

    if client_id == NUM_CLIENTS - 1:

        end = total_samples

    else:

        end = (
            (client_id + 1)
            * samples_per_client
        )

    indices = list(
        range(
            start,
            end
        )
    )

    CLIENT_INDICES.append(
        indices
    )


print("\nClient data distribution:")

for client_id in range(NUM_CLIENTS):

    print(
        f"Client {client_id + 1}: "
        f"{len(CLIENT_INDICES[client_id])} samples"
    )


# ============================================================
# GLOBAL TEST DATALOADER
# ============================================================

GLOBAL_TEST_LOADER = DataLoader(
    FULL_TEST_DATASET,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOSS FUNCTION
# ============================================================

CRITERION = nn.CrossEntropyLoss()


# ============================================================
# MODEL PARAMETER FUNCTIONS
#
# IMPORTANT:
# Both functions use model.parameters().
#
# DO NOT use state_dict() here.
# This prevents the SNN parameter/buffer mismatch.
# ============================================================

def get_model_parameters(model):

    return [
        parameter.detach()
        .cpu()
        .numpy()
        .copy()
        for parameter in model.parameters()
    ]


def set_model_parameters(
    model,
    parameters
):

    model_parameters = list(
        model.parameters()
    )

    if len(model_parameters) != len(parameters):

        raise ValueError(
            "\nParameter count mismatch!\n"
            f"Model parameters : "
            f"{len(model_parameters)}\n"
            f"Received parameters: "
            f"{len(parameters)}"
        )

    for parameter, new_parameter in zip(
        model_parameters,
        parameters
    ):

        new_tensor = torch.tensor(
            new_parameter,
            dtype=parameter.dtype,
            device=parameter.device
        )

        if parameter.shape != new_tensor.shape:

            raise ValueError(
                "\nParameter shape mismatch!\n"
                f"Model shape    : "
                f"{tuple(parameter.shape)}\n"
                f"Received shape : "
                f"{tuple(new_tensor.shape)}"
            )

        parameter.data.copy_(
            new_tensor
        )


# ============================================================
# CHECK MODEL
# ============================================================

def print_model_parameters(model):

    print("\n" + "=" * 70)
    print("MODEL PARAMETER CHECK")
    print("=" * 70)

    for name, parameter in model.named_parameters():

        print(
            f"{name:30s} "
            f"{tuple(parameter.shape)}"
        )

    print("=" * 70)


# ============================================================
# TRAIN ONE CLIENT
# ============================================================

def train_local_model(
    model,
    train_loader,
    client_id,
    round_number
):

    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    final_loss = 0.0

    final_accuracy = 0.0

    total_samples = 0


    # --------------------------------------------------------
    # Local epochs
    # --------------------------------------------------------

    epoch_progress = tqdm(
        range(LOCAL_EPOCHS),
        desc=(
            f"Client {client_id} "
            f"Round {round_number}"
        ),
        leave=False
    )


    for epoch in epoch_progress:

        epoch_loss = 0.0

        epoch_correct = 0

        epoch_samples = 0


        # ----------------------------------------------------
        # Batches
        # ----------------------------------------------------

        for data, labels in train_loader:

            data = data.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad()


            # ------------------------------------------------
            # SNN forward
            #
            # Output:
            # [time, batch, classes]
            # ------------------------------------------------

            outputs = model(
                data
            )


            # ------------------------------------------------
            # Sum spikes over time
            #
            # [time, batch, classes]
            #        ↓
            # [batch, classes]
            # ------------------------------------------------

            output_sum = outputs.sum(
                dim=0
            )


            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = CRITERION(
                output_sum,
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

            epoch_loss += (
                loss.item()
                * batch_size
            )


            predictions = torch.argmax(
                output_sum,
                dim=1
            )


            epoch_correct += (
                (predictions == labels)
                .sum()
                .item()
            )


            epoch_samples += (
                batch_size
            )


        # ----------------------------------------------------
        # Epoch metrics
        # ----------------------------------------------------

        if epoch_samples > 0:

            epoch_loss = (
                epoch_loss
                / epoch_samples
            )

            epoch_accuracy = (
                epoch_correct
                / epoch_samples
                * 100.0
            )

        else:

            epoch_loss = 0.0

            epoch_accuracy = 0.0


        # ----------------------------------------------------
        # Progress display
        # ----------------------------------------------------

        epoch_progress.set_postfix(
            loss=f"{epoch_loss:.4f}",
            acc=f"{epoch_accuracy:.2f}%"
        )


        final_loss = epoch_loss

        final_accuracy = epoch_accuracy

        total_samples = epoch_samples


    epoch_progress.close()


    return (
        final_loss,
        final_accuracy,
        total_samples
    )


# ============================================================
# EVALUATE GLOBAL MODEL
# ============================================================

def evaluate_global_model(
    parameters
):

    model = LIFSNN(
        num_classes=NUM_CLASSES
    ).to(DEVICE)


    # --------------------------------------------------------
    # Load global parameters
    # --------------------------------------------------------

    set_model_parameters(
        model,
        parameters
    )


    model.eval()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0


    # --------------------------------------------------------
    # Testing
    # --------------------------------------------------------

    with torch.no_grad():

        for data, labels in GLOBAL_TEST_LOADER:

            data = data.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(
                data
            )


            output_sum = outputs.sum(
                dim=0
            )


            loss = CRITERION(
                output_sum,
                labels
            )


            batch_size = labels.size(0)


            total_loss += (
                loss.item()
                * batch_size
            )


            predictions = torch.argmax(
                output_sum,
                dim=1
            )


            total_correct += (
                (predictions == labels)
                .sum()
                .item()
            )


            total_samples += (
                batch_size
            )


    if total_samples == 0:

        return (
            0.0,
            0.0
        )


    average_loss = (
        total_loss
        / total_samples
    )


    accuracy = (
        total_correct
        / total_samples
        * 100.0
    )


    return (
        average_loss,
        accuracy
    )


# ============================================================
# FLOWER CLIENT
# ============================================================

class StandardLIFClient(
    fl.client.NumPyClient
):

    def __init__(
        self,
        client_id
    ):

        self.client_id = int(
            client_id
        )


        # ----------------------------------------------------
        # Client dataset
        # ----------------------------------------------------

        self.client_dataset = Subset(
            FULL_TRAIN_DATASET,
            CLIENT_INDICES[
                self.client_id
            ]
        )


        self.train_loader = DataLoader(
            self.client_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )


        # ----------------------------------------------------
        # Local model
        # ----------------------------------------------------

        self.model = LIFSNN(
            num_classes=NUM_CLASSES
        ).to(DEVICE)


    # ========================================================
    # GET PARAMETERS
    # ========================================================

    def get_parameters(
        self,
        config
    ):

        return get_model_parameters(
            self.model
        )


    # ========================================================
    # LOCAL TRAINING
    # ========================================================

    def fit(
        self,
        parameters,
        config
    ):

        server_round = int(
            config.get(
                "server_round",
                0
            )
        )


        # ----------------------------------------------------
        # Load global model
        # ----------------------------------------------------

        set_model_parameters(
            self.model,
            parameters
        )


        print(
            f"\n[Client {self.client_id + 1}] "
            f"Starting Round "
            f"{server_round}"
        )


        # ----------------------------------------------------
        # Local training
        # ----------------------------------------------------

        (
            train_loss,
            train_accuracy,
            num_samples
        ) = train_local_model(
            self.model,
            self.train_loader,
            self.client_id + 1,
            server_round
        )


        print(
            f"[Client {self.client_id + 1}] "
            f"Round {server_round} completed | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: "
            f"{train_accuracy:.2f}%"
        )


        return (
            get_model_parameters(
                self.model
            ),
            num_samples,
            {
                "train_loss": float(
                    train_loss
                ),
                "train_accuracy": float(
                    train_accuracy
                )
            }
        )


# ============================================================
# FLOWER CLIENT FUNCTION
# ============================================================

def client_fn(cid):

    return StandardLIFClient(
        cid
    ).to_client()


# ============================================================
# FIT CONFIGURATION
# ============================================================

def fit_config(
    server_round
):

    return {
        "server_round": server_round,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE
    }


# ============================================================
# AGGREGATE TRAINING METRICS
# ============================================================

def aggregate_fit_metrics(
    metrics
):

    if not metrics:

        return {}


    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )


    if total_examples == 0:

        return {}


    weighted_loss = sum(
        num_examples
        * metric.get(
            "train_loss",
            0.0
        )
        for num_examples, metric
        in metrics
    ) / total_examples


    weighted_accuracy = sum(
        num_examples
        * metric.get(
            "train_accuracy",
            0.0
        )
        for num_examples, metric
        in metrics
    ) / total_examples


    return {
        "train_loss": float(
            weighted_loss
        ),
        "train_accuracy": float(
            weighted_accuracy
        )
    }


# ============================================================
# CUSTOM FEDAVG STRATEGY
# ============================================================

class ProgressFedAvg(
    fl.server.strategy.FedAvg
):

    def aggregate_fit(
        self,
        server_round,
        results,
        failures
    ):

        global TOTAL_UPLOADS

        global LAST_GLOBAL_PARAMETERS


        # ----------------------------------------------------
        # Number of successful clients
        # ----------------------------------------------------

        successful_clients = len(
            results
        )


        # ----------------------------------------------------
        # Count uploads
        # ----------------------------------------------------

        TOTAL_UPLOADS += (
            successful_clients
        )


        # ----------------------------------------------------
        # Standard FedAvg
        # ----------------------------------------------------

        (
            aggregated_parameters,
            metrics
        ) = super().aggregate_fit(
            server_round,
            results,
            failures
        )


        # ----------------------------------------------------
        # Save global parameters
        # ----------------------------------------------------

        LAST_GLOBAL_PARAMETERS = (
            aggregated_parameters
        )


        # ----------------------------------------------------
        # Aggregate training metrics
        # ----------------------------------------------------

        if results:

            total_examples = sum(
                result.num_examples
                for _, result in results
            )


            if total_examples > 0:

                train_loss = sum(
                    result.num_examples
                    * result.metrics.get(
                        "train_loss",
                        0.0
                    )
                    for _, result in results
                ) / total_examples


                train_accuracy = sum(
                    result.num_examples
                    * result.metrics.get(
                        "train_accuracy",
                        0.0
                    )
                    for _, result in results
                ) / total_examples

            else:

                train_loss = 0.0

                train_accuracy = 0.0

        else:

            train_loss = 0.0

            train_accuracy = 0.0


        # ----------------------------------------------------
        # Save round result
        # ----------------------------------------------------

        ROUND_RESULTS.append(
            {
                "round": server_round,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": None,
                "test_accuracy": None,
                "uploads": successful_clients
            }
        )


        return (
            aggregated_parameters,
            metrics
        )


# ============================================================
# CENTRALIZED SERVER EVALUATION
# ============================================================

def centralized_evaluate(
    server_round,
    parameters,
    config
):

    test_loss, test_accuracy = (
        evaluate_global_model(
            parameters
        )
    )


    # --------------------------------------------------------
    # Store test metrics
    # --------------------------------------------------------

    for result in ROUND_RESULTS:

        if result["round"] == server_round:

            result["test_loss"] = (
                test_loss
            )

            result["test_accuracy"] = (
                test_accuracy
            )

            break


    # --------------------------------------------------------
    # Find training metrics
    # --------------------------------------------------------

    latest_train_loss = 0.0

    latest_train_accuracy = 0.0


    for result in ROUND_RESULTS:

        if result["round"] == server_round:

            latest_train_loss = (
                result["train_loss"]
            )

            latest_train_accuracy = (
                result["train_accuracy"]
            )

            break


    # --------------------------------------------------------
    # Display round results
    # --------------------------------------------------------

    print(
        "\n"
        + "-" * 70
    )

    print(
        f"ROUND {server_round}/{NUM_ROUNDS}"
    )

    print(
        f"Train Loss     : "
        f"{latest_train_loss:.4f}"
    )

    print(
        f"Train Accuracy : "
        f"{latest_train_accuracy:.2f}%"
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
        f"Total Uploads  : "
        f"{TOTAL_UPLOADS}"
    )

    print(
        "-" * 70
    )


    return (
        test_loss,
        {
            "test_accuracy": float(
                test_accuracy
            )
        }
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv():

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [
                "Round",
                "Train Loss",
                "Train Accuracy",
                "Test Loss",
                "Test Accuracy",
                "Uploads"
            ]
        )


        for result in ROUND_RESULTS:

            writer.writerow(
                [
                    result["round"],
                    result["train_loss"],
                    result["train_accuracy"],
                    result["test_loss"],
                    result["test_accuracy"],
                    result["uploads"]
                ]
            )


# ============================================================
# SAVE GRAPHS
# ============================================================

def save_graphs():

    if not ROUND_RESULTS:

        return


    rounds = [
        result["round"]
        for result in ROUND_RESULTS
    ]


    train_accuracy = [
        result["train_accuracy"]
        for result in ROUND_RESULTS
    ]


    train_loss = [
        result["train_loss"]
        for result in ROUND_RESULTS
    ]


    test_accuracy = [
        result["test_accuracy"]
        for result in ROUND_RESULTS
    ]


    test_loss = [
        result["test_loss"]
        for result in ROUND_RESULTS
    ]


    # --------------------------------------------------------
    # Train Accuracy
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        rounds,
        train_accuracy,
        marker="o",
        linewidth=2
    )

    plt.xlabel(
        "Communication Round"
    )

    plt.ylabel(
        "Training Accuracy (%)"
    )

    plt.title(
        "Standard-LIF Federated Training Accuracy"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        TRAIN_ACCURACY_PLOT,
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # Train Loss
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        rounds,
        train_loss,
        marker="o",
        linewidth=2
    )

    plt.xlabel(
        "Communication Round"
    )

    plt.ylabel(
        "Training Loss"
    )

    plt.title(
        "Standard-LIF Federated Training Loss"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        TRAIN_LOSS_PLOT,
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # Test Accuracy
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        rounds,
        test_accuracy,
        marker="o",
        linewidth=2
    )

    plt.xlabel(
        "Communication Round"
    )

    plt.ylabel(
        "Test Accuracy (%)"
    )

    plt.title(
        "Standard-LIF Federated Test Accuracy"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        TEST_ACCURACY_PLOT,
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # Test Loss
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        rounds,
        test_loss,
        marker="o",
        linewidth=2
    )

    plt.xlabel(
        "Communication Round"
    )

    plt.ylabel(
        "Test Loss"
    )

    plt.title(
        "Standard-LIF Federated Test Loss"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        TEST_LOSS_PLOT,
        dpi=300
    )

    plt.close()


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

def save_summary():

    if not ROUND_RESULTS:

        return


    final_result = (
        ROUND_RESULTS[-1]
    )


    # --------------------------------------------------------
    # Model parameter count
    # --------------------------------------------------------

    model = LIFSNN(
        num_classes=NUM_CLASSES
    )


    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )


    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


    # --------------------------------------------------------
    # Communication calculation
    #
    # Float32 = 4 bytes
    # --------------------------------------------------------

    model_bytes = (
        total_parameters * 4
    )


    total_upload_bytes = (
        model_bytes
        * TOTAL_UPLOADS
    )


    total_upload_mb = (
        total_upload_bytes
        / (1024 * 1024)
    )


    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "=" * 70
            + "\n"
        )

        file.write(
            "STANDARD-LIF FEDERATED LEARNING "
            "SUMMARY\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )


        file.write(
            f"Device: {DEVICE}\n"
        )

        file.write(
            "Dataset: N-MNIST\n"
        )

        file.write(
            f"Clients: {NUM_CLIENTS}\n"
        )

        file.write(
            f"Local epochs: {LOCAL_EPOCHS}\n"
        )

        file.write(
            f"Batch size: {BATCH_SIZE}\n"
        )

        file.write(
            f"Learning rate: {LEARNING_RATE}\n"
        )

        file.write(
            f"Communication rounds: {NUM_ROUNDS}\n"
        )

        file.write(
            "Optimizer: Adam\n"
        )

        file.write(
            "Aggregation: FedAvg\n\n"
        )


        file.write(
            f"Total model parameters: "
            f"{total_parameters}\n"
        )

        file.write(
            f"Trainable parameters: "
            f"{trainable_parameters}\n\n"
        )


        file.write(
            f"Final train loss: "
            f"{final_result['train_loss']:.4f}\n"
        )

        file.write(
            f"Final train accuracy: "
            f"{final_result['train_accuracy']:.2f}%\n"
        )

        file.write(
            f"Final test loss: "
            f"{final_result['test_loss']:.4f}\n"
        )

        file.write(
            f"Final test accuracy: "
            f"{final_result['test_accuracy']:.2f}%\n\n"
        )


        file.write(
            f"Total communication rounds: "
            f"{NUM_ROUNDS}\n"
        )

        file.write(
            f"Total client uploads: "
            f"{TOTAL_UPLOADS}\n"
        )

        file.write(
            f"Approximate model payload per upload: "
            f"{model_bytes} bytes\n"
        )

        file.write(
            f"Approximate total uploaded payload: "
            f"{total_upload_mb:.2f} MB\n\n"
        )


        file.write(
            "Standard-LIF Federated Learning "
            "completed successfully.\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    global LAST_GLOBAL_PARAMETERS


    print(
        "\n"
        + "=" * 70
    )

    print(
        "STARTING FLOWER FEDAVG"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Create initial model
    # --------------------------------------------------------

    initial_model = LIFSNN(
        num_classes=NUM_CLASSES
    ).to(DEVICE)


    # --------------------------------------------------------
    # Check model architecture
    # --------------------------------------------------------

    print_model_parameters(
        initial_model
    )


    # --------------------------------------------------------
    # Convert initial model parameters
    # --------------------------------------------------------

    initial_parameters = (
        ndarrays_to_parameters(
            get_model_parameters(
                initial_model
            )
        )
    )


    print(
        "\nInitial model parameters "
        "created successfully."
    )


    # --------------------------------------------------------
    # FedAvg strategy
    # --------------------------------------------------------

    strategy = ProgressFedAvg(

        fraction_fit=1.0,

        fraction_evaluate=0.0,

        min_fit_clients=NUM_CLIENTS,

        min_evaluate_clients=0,

        min_available_clients=NUM_CLIENTS,

        evaluate_fn=centralized_evaluate,

        on_fit_config_fn=fit_config,

        initial_parameters=initial_parameters,

        fit_metrics_aggregation_fn=(
            aggregate_fit_metrics
        )
    )


    print(
        "\nStarting federated training..."
    )

    print(
        f"Total communication rounds: "
        f"{NUM_ROUNDS}"
    )


    # --------------------------------------------------------
    # Start Flower simulation
    # --------------------------------------------------------

    history = fl.simulation.start_simulation(

        client_fn=client_fn,

        num_clients=NUM_CLIENTS,

        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),

        strategy=strategy,

        client_resources={
            "num_cpus": 1,
            "num_gpus": 0
        }
    )


    # --------------------------------------------------------
    # Save final global model
    # --------------------------------------------------------

    if LAST_GLOBAL_PARAMETERS is not None:

        final_model = LIFSNN(
            num_classes=NUM_CLASSES
        ).to(DEVICE)


        final_ndarrays = (
            parameters_to_ndarrays(
                LAST_GLOBAL_PARAMETERS
            )
        )


        set_model_parameters(
            final_model,
            final_ndarrays
        )


        torch.save(
            final_model.state_dict(),
            MODEL_FILE
        )


        print(
            "\nFinal global model saved to:"
        )

        print(
            MODEL_FILE
        )


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_csv()

    save_graphs()

    save_summary()


    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    if ROUND_RESULTS:

        final_result = (
            ROUND_RESULTS[-1]
        )


        print(
            "\n"
            + "=" * 70
        )

        print(
            "FINAL FEDERATED RESULTS"
        )

        print(
            "=" * 70
        )


        print(
            f"Final Train Loss     : "
            f"{final_result['train_loss']:.4f}"
        )

        print(
            f"Final Train Accuracy : "
            f"{final_result['train_accuracy']:.2f}%"
        )

        print(
            f"Final Test Loss      : "
            f"{final_result['test_loss']:.4f}"
        )

        print(
            f"Final Test Accuracy  : "
            f"{final_result['test_accuracy']:.2f}%"
        )

        print(
            f"Communication Rounds : "
            f"{NUM_ROUNDS}"
        )

        print(
            f"Total Client Uploads : "
            f"{TOTAL_UPLOADS}"
        )


    # --------------------------------------------------------
    # Results location
    # --------------------------------------------------------

    print(
        "\nResults saved in:"
    )

    print(
        RESULTS_DIR
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "STANDARD-LIF FEDAVG COMPLETED"
    )

    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()