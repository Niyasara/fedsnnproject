import os
import torch
import torch.nn as nn
import snntorch as snn


# ============================================================
# LIF Spiking Neural Network
# ============================================================

class LIFSNN(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        # ----------------------------------------------------
        # Convolutional layer 1
        # Input: 2 x 34 x 34
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=2
        )

        self.lif1 = snn.Leaky(
            beta=0.9
        )

        # ----------------------------------------------------
        # Convolutional layer 2
        # ----------------------------------------------------

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool2 = nn.MaxPool2d(
            kernel_size=2
        )

        self.lif2 = snn.Leaky(
            beta=0.9
        )

        # ----------------------------------------------------
        # Fully connected layer
        #
        # 34 x 34
        #     ↓ MaxPool
        # 17 x 17
        #     ↓ MaxPool
        # 8 x 8
        #
        # 32 feature maps x 8 x 8
        # ----------------------------------------------------

        self.fc = nn.Linear(
            32 * 8 * 8,
            num_classes
        )

        self.lif_out = snn.Leaky(
            beta=0.9
        )

    # ========================================================
    # Forward Pass
    # ========================================================

    def forward(self, x):

        # x shape:
        # [batch, time, channels, height, width]

        batch_size = x.size(0)

        time_steps = x.size(1)

        # Initialize membrane potentials

        mem1 = self.lif1.init_leaky()

        mem2 = self.lif2.init_leaky()

        mem_out = self.lif_out.init_leaky()

        outputs = []

        # ----------------------------------------------------
        # Process each time step
        # ----------------------------------------------------

        for t in range(time_steps):

            current = x[:, t]

            # First convolution
            current = self.conv1(current)

            current = self.pool1(current)

            spk1, mem1 = self.lif1(
                current,
                mem1
            )

            # Second convolution
            current = self.conv2(spk1)

            current = self.pool2(current)

            spk2, mem2 = self.lif2(
                current,
                mem2
            )

            # Flatten
            current = spk2.reshape(
                batch_size,
                -1
            )

            # Fully connected layer
            current = self.fc(current)

            # Output LIF neuron
            spk_out, mem_out = self.lif_out(
                current,
                mem_out
            )

            outputs.append(spk_out)

        # ----------------------------------------------------
        # Stack outputs
        #
        # [time, batch, classes]
        # ----------------------------------------------------

        outputs = torch.stack(
            outputs,
            dim=0
        )

        return outputs


# ============================================================
# Model Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LIF SNN MODEL TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # CPU only
    # --------------------------------------------------------

    device = torch.device("cpu")

    print("\nDevice:", device)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = LIFSNN(
        num_classes=10
    ).to(device)

    print("Model created successfully.")

    # --------------------------------------------------------
    # Count parameters
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
    # Create test input
    #
    # Batch      = 32
    # Time steps = 10
    # Channels   = 2
    # Height     = 34
    # Width      = 34
    # --------------------------------------------------------

    x = torch.rand(
        32,
        10,
        2,
        34,
        34
    ).to(device)

    print("\nInput shape:")
    print(x.shape)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        output = model(x)

    print("\nOutput shape:")
    print(output.shape)

    # --------------------------------------------------------
    # Expected output
    # --------------------------------------------------------

    expected_shape = (
        10,
        32,
        10
    )

    print("\nExpected output shape:")
    print(
        f"torch.Size({expected_shape})"
    )

    # --------------------------------------------------------
    # Check output
    # --------------------------------------------------------

    if tuple(output.shape) == expected_shape:

        print(
            "\nOutput shape verification: PASSED"
        )

    else:

        print(
            "\nOutput shape verification: FAILED"
        )

    # --------------------------------------------------------
    # Calculate total spikes
    # --------------------------------------------------------

    total_spikes = output.sum().item()

    print(
        "\nTotal output spikes:",
        total_spikes
    )

    # --------------------------------------------------------
    # Create results directory
    # --------------------------------------------------------

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    results_dir = os.path.join(
        project_root,
        "results",
        "phase1"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save model test result
    # --------------------------------------------------------

    output_file = os.path.join(
        results_dir,
        "model_test.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("=" * 60 + "\n")
        f.write("LIF SNN MODEL TEST\n")
        f.write("=" * 60 + "\n\n")

        f.write(
            f"Device: {device}\n"
        )

        f.write(
            "Model: LIFSNN\n"
        )

        f.write(
            "Neuron: Leaky Integrate-and-Fire (LIF)\n"
        )

        f.write(
            "Number of classes: 10\n"
        )

        f.write(
            "Time steps: 10\n"
        )

        f.write(
            "Input channels: 2\n"
        )

        f.write(
            "Input size: 34 x 34\n\n"
        )

        f.write(
            f"Total parameters: "
            f"{total_parameters}\n"
        )

        f.write(
            f"Trainable parameters: "
            f"{trainable_parameters}\n\n"
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
            f"Expected output shape: "
            f"{expected_shape}\n"
        )

        f.write(
            f"Total output spikes: "
            f"{total_spikes}\n\n"
        )

        if tuple(output.shape) == expected_shape:

            f.write(
                "Output shape verification: PASSED\n"
            )

            f.write(
                "SNN model test completed successfully.\n"
            )

        else:

            f.write(
                "Output shape verification: FAILED\n"
            )

    print("\nResult saved to:")
    print(output_file)

    print("\n" + "=" * 60)
    print("SNN MODEL TEST COMPLETED")
    print("=" * 60)