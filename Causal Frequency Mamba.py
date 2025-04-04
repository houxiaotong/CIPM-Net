import torch
import torch.nn as nn
from mambaIR import VSSBlock
from demo import PatchEmbed, PatchUnEmbed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Curved_state_space_model class
class Curved_state_space_model(nn.Module):
    def __init__(self, channel):
        super(Curved_state_space_model, self).__init__()
        # Adaptive average pooling layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.n_curve = 3
        # ReLU activation function
        self.relu = nn.ReLU(inplace=False)
        self.predict_a = nn.Sequential(
            nn.Conv2d(channel, channel, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 3, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Predict the value of a
        a = self.predict_a(x)
        # Apply ReLU activation function
        x = self.relu(x) - self.relu(x - 1)
        for i in range(self.n_curve):
            x = x + a[:, i:i + 1] * x * (1 - x)
        return x

# Causal_Frequency_Mamba class
class Causal_Frequency_Mamba(nn.Module):
    def __init__(self, in_channels=1):
        super(Causal_Frequency_Mamba, self).__init__()
        # First convolutional layer, change the input channel number to 2
        self.conv1 = nn.Conv2d(2, in_channels, kernel_size=3, padding=1)
        # PReLU activation function
        self.prelu = nn.PReLU()
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # Curved state space model layer
        self.curve_ca_layer = Curved_state_space_model(in_channels)
        # Third convolutional layer, change the output channel number to 1
        self.conv3 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        # VSSBlock layer
        self.VSS = VSSBlock(hidden_dim=32, drop_path=0.1, attn_drop_rate=0.1, d_state=2, is_light_sr=False)
        # Patch embedding layer
        self.patch = PatchEmbed()
        # Patch unembedding layer
        self.patch_embed = PatchUnEmbed()

    def forward(self, x):
        # Move the input tensor to the device
        x_input = x.to(device)
        # Perform 2D Fast Fourier Transform
        x = torch.fft.fft2(x_input, dim=(-2, -1))
        # Convert the complex tensor to a real tensor
        x = torch.view_as_real(x)  # Output shape: (batch, channel, height, width, 2)
        # Adjust the dimensions to fit the convolutional layer
        x_f = x.permute(0, 4, 1, 2, 3).contiguous().view(x.size(0), 2 * x.size(1), x.size(2), x.size(3))
        # Pass through the first convolutional layer, PReLU layer, and the second convolutional layer
        x = self.conv1(x_f)
        x = self.prelu(x)
        x = self.conv2(x)
        # Pass through the Curved state space model layer
        x = self.curve_ca_layer(x)
        # Pass through the third convolutional layer
        x = self.conv3(x)
        # Patch embedding
        x = self.patch(x)
        # Pass through the VSSBlock layer
        x = self.VSS(x, (8, 8))
        # Patch unembedding
        x = self.patch_embed(x)
        # Residual connection
        x = x + x_f
        # Restore the dimensions for inverse 2D Fast Fourier Transform
        x = x.view(x.size(0), 2, -1, x.size(2), x.size(3)).permute(0, 2, 3, 4, 1)
        # Ensure the stride of the last dimension is 1
        x = x.contiguous()
        # Convert the real tensor to a complex tensor
        x = torch.view_as_complex(x)
        # Perform inverse 2D Fast Fourier Transform
        x = torch.fft.ifft2(x, dim=(-2, -1))
        # Get the real part of the complex tensor
        x = torch.real(x)
        # Residual connection
        x = x + x_input
        return x

# Generate a random input tensor
input_tensor = torch.randn(2, 1, 256, 256).to(device)

# Initialize the processing module
processing_module = Causal_Frequency_Mamba().to(device)

# Perform forward propagation
output = processing_module(input_tensor)

# Print the shape of the output tensor
print("Output tensor shape:", output.shape)