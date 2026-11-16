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
# SSM LAYER
# ============================================================

class SSMBlock(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            hidden_size
        )

        self.state_projection = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.output_projection = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.activation = nn.Tanh()

    def forward(
        self,
        x,
        state
    ):

        input_value = self.input_projection(x)

        state_update = self.state_projection(
            state
        )

        state = self.activation(
            input_value + state_update
        )

        output = self.output_projection(
            state
        )

        return output, state


# ============================================================
# SSM SNN CLASSIFIER
# ============================================================

class SSMSNN(nn.Module):

    def __init__(
        self,
        input_channels=2,
        height=34,
        width=34,
        hidden_size=128,
        num_classes=10
    ):

        super().__init__()

        self.input_size = (
            input_channels
            * height
            * width
        )

        self.hidden_size = hidden_size

        self.num_classes = num_classes

        # ----------------------------------------------------
        # SSM block
        # ----------------------------------------------------

        self.ssm = SSMBlock(
            input_size=self.input_size,
            hidden_size=hidden_size
        )

        # ----------------------------------------------------
        # Classification layer
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        # Expected:
        # [batch, time, channels, height, width]

        batch_size = x.size(0)

        time_steps = x.size(1)

        # ----------------------------------------------------
        # Initial SSM state
        # ----------------------------------------------------

        state = torch.zeros(
            batch_size,
            self.hidden_size,
            device=x.device
        )

        outputs = []

        # ----------------------------------------------------
        # Process each time step
        # ----------------------------------------------------

        for t in range(time_steps):

            current_frame = x[:, t]

            current_frame = current_frame.reshape(
                batch_size,
                -1
            )

            output, state = self.ssm(
                current_frame,
                state
            )

            class_output = self.classifier(
                output
            )

            outputs.append(
                class_output
            )

        # ----------------------------------------------------
        # [time, batch, classes]
        # ----------------------------------------------------

        outputs = torch.stack(
            outputs,
            dim=0
        )

        return outputs


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SSM SNN MODEL TEST")
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

    model = SSMSNN(
        input_channels=2,
        height=34,
        width=34,
        hidden_size=128,
        num_classes=10
    ).to(device)

    print()
    print("Model created successfully.")

    # --------------------------------------------------------
    # Parameters
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

    test_input = torch.rand(
        32,
        10,
        2,
        34,
        34,
        device=device
    )

    print()
    print("Input shape:")
    print(test_input.shape)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            test_input
        )

    print()
    print("Output shape:")
    print(output.shape)

    # --------------------------------------------------------
    # Expected shape
    # --------------------------------------------------------

    expected_shape = (
        10,
        32,
        10
    )

    print()
    print("Expected output shape:")
    print(
        torch.Size(expected_shape)
    )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    if tuple(output.shape) == expected_shape:

        print()
        print(
            "Output shape verification: PASSED"
        )

    else:

        print()
        print(
            "Output shape verification: FAILED"
        )

    # --------------------------------------------------------
    # Output statistics
    # --------------------------------------------------------

    print()
    print(
        "Total output values:",
        output.numel()
    )

    print()
    print("=" * 60)
    print("SSM SNN MODEL TEST COMPLETED")
    print("=" * 60)