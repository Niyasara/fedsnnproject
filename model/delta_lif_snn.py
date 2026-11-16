import os
import sys
import torch
import torch.nn as nn


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# DELTA-LIF NEURON
# ============================================================

class DeltaLIFNeuron(nn.Module):

    def __init__(
        self,
        delta_log_init=-1.0,
        gamma_log_init=0.0,
        threshold=1.0
    ):
        super().__init__()

        self.delta_log = nn.Parameter(
            torch.tensor(float(delta_log_init))
        )

        self.gamma_log = nn.Parameter(
            torch.tensor(float(gamma_log_init))
        )

        self.threshold = threshold

    def get_alpha(self):

        delta = torch.exp(self.delta_log)
        gamma = torch.exp(self.gamma_log)

        alpha = torch.exp(-delta * gamma)

        return alpha

    def forward(self, input_current, membrane):

        alpha = self.get_alpha()

        previous_spike = (
            membrane >= self.threshold
        ).float()

        membrane = (
            alpha
            * (
                membrane
                - self.threshold * previous_spike
            )
            + (1.0 - alpha) * input_current
        )

        spike = self._surrogate_spike(
            membrane - self.threshold
        )

        return spike, membrane

    @staticmethod
    def _surrogate_spike(x):

        hard_spike = (x >= 0).float()

        surrogate = torch.sigmoid(10.0 * x)

        return (
            hard_spike
            + surrogate
            - surrogate.detach()
        )


# ============================================================
# DELTA-LIF SNN CLASSIFIER
# ============================================================

class DeltaLIFSNN(nn.Module):

    def __init__(
        self,
        input_channels=2,
        height=34,
        width=34,
        hidden_size=128,
        num_classes=10,
        threshold=1.0
    ):
        super().__init__()

        input_size = (
            input_channels
            * height
            * width
        )

        # Hidden layer
        self.fc1 = nn.Linear(
            input_size,
            hidden_size
        )

        self.delta_lif1 = DeltaLIFNeuron(
            threshold=threshold
        )

        # Output layer
        self.fc2 = nn.Linear(
            hidden_size,
            num_classes
        )

        self.delta_lif2 = DeltaLIFNeuron(
            threshold=threshold
        )

    # ========================================================
    # FORWARD PASS
    # ========================================================

    def forward(self, x):

        batch_size = x.size(0)
        time_steps = x.size(1)

        device = x.device

        hidden_membrane = torch.zeros(
            batch_size,
            128,
            device=device
        )

        output_membrane = torch.zeros(
            batch_size,
            10,
            device=device
        )

        output_spikes = []

        for t in range(time_steps):

            # Current time step
            x_t = x[:, t]

            # Flatten image
            x_t = x_t.reshape(
                batch_size,
                -1
            )

            # Hidden layer
            hidden_current = self.fc1(x_t)

            hidden_spike, hidden_membrane = (
                self.delta_lif1(
                    hidden_current,
                    hidden_membrane
                )
            )

            # Output layer
            output_current = self.fc2(
                hidden_spike
            )

            output_spike, output_membrane = (
                self.delta_lif2(
                    output_current,
                    output_membrane
                )
            )

            output_spikes.append(
                output_spike
            )

        # (time, batch, classes)
        output_spikes = torch.stack(
            output_spikes,
            dim=0
        )

        return output_spikes


# ============================================================
# MODEL TEST
# ============================================================

def test_model():

    print("=" * 60)
    print("DELTA-LIF SNN MODEL TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = DeltaLIFSNN(
        input_channels=2,
        height=34,
        width=34,
        hidden_size=128,
        num_classes=10,
        threshold=1.0
    ).to(device)

    print()
    print("Model created successfully.")

    # --------------------------------------------------------
    # Parameter count
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
    # Test input
    # --------------------------------------------------------

    batch_size = 32
    time_steps = 10

    x = torch.rand(
        batch_size,
        time_steps,
        2,
        34,
        34,
        device=device
    )

    print()
    print("Input shape:")
    print(x.shape)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():
        output = model(x)

    print()
    print("Output shape:")
    print(output.shape)

    # --------------------------------------------------------
    # Expected output
    # --------------------------------------------------------

    expected_shape = (
        time_steps,
        batch_size,
        10
    )

    print()
    print("Expected output shape:")
    print(torch.Size(expected_shape))

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    passed = (
        tuple(output.shape)
        == expected_shape
    )

    print()

    if passed:
        print(
            "Output shape verification: PASSED"
        )
    else:
        print(
            "Output shape verification: FAILED"
        )

    # --------------------------------------------------------
    # Spike count
    # --------------------------------------------------------

    total_spikes = output.sum().item()

    print()
    print(
        "Total output spikes:",
        total_spikes
    )

    # --------------------------------------------------------
    # Delta-LIF parameters
    # --------------------------------------------------------

    hidden_alpha = (
        model.delta_lif1
        .get_alpha()
        .detach()
        .cpu()
        .item()
    )

    output_alpha = (
        model.delta_lif2
        .get_alpha()
        .detach()
        .cpu()
        .item()
    )

    print()
    print("Delta-LIF parameters:")

    print(
        "Hidden Delta_log:",
        model.delta_lif1.delta_log.item()
    )

    print(
        "Hidden Gamma_log:",
        model.delta_lif1.gamma_log.item()
    )

    print(
        "Hidden alpha:",
        hidden_alpha
    )

    print(
        "Output Delta_log:",
        model.delta_lif2.delta_log.item()
    )

    print(
        "Output Gamma_log:",
        model.delta_lif2.gamma_log.item()
    )

    print(
        "Output alpha:",
        output_alpha
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results_dir = os.path.join(
        PROJECT_ROOT,
        "results",
        "phase1"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    result_file = os.path.join(
        results_dir,
        "delta_lif_model_test.txt"
    )

    with open(
        result_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("=" * 60 + "\n")
        f.write("DELTA-LIF SNN MODEL TEST\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Device: {device}\n")

        f.write(
            f"Total parameters: "
            f"{total_parameters}\n"
        )

        f.write(
            f"Trainable parameters: "
            f"{trainable_parameters}\n"
        )

        f.write(
            f"Input shape: "
            f"{tuple(x.shape)}\n"
        )

        f.write(
            f"Output shape: "
            f"{tuple(output.shape)}\n"
        )

        f.write(
            f"Expected shape: "
            f"{expected_shape}\n"
        )

        f.write(
            "Output verification: "
            f"{'PASSED' if passed else 'FAILED'}\n"
        )

        f.write(
            f"Total output spikes: "
            f"{total_spikes}\n"
        )

        f.write("\nDelta-LIF parameters\n")

        f.write(
            f"Hidden Delta_log: "
            f"{model.delta_lif1.delta_log.item()}\n"
        )

        f.write(
            f"Hidden Gamma_log: "
            f"{model.delta_lif1.gamma_log.item()}\n"
        )

        f.write(
            f"Hidden alpha: "
            f"{hidden_alpha}\n"
        )

        f.write(
            f"Output Delta_log: "
            f"{model.delta_lif2.delta_log.item()}\n"
        )

        f.write(
            f"Output Gamma_log: "
            f"{model.delta_lif2.gamma_log.item()}\n"
        )

        f.write(
            f"Output alpha: "
            f"{output_alpha}\n"
        )

    print()
    print("Result saved to:")
    print(result_file)

    print()
    print("=" * 60)
    print("DELTA-LIF SNN MODEL TEST COMPLETED")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    test_model()