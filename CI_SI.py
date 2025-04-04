import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SI(nn.Module):
    def __init__(self, in_channels=1):
        super(SI, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x
class CI(nn.Module):
    def __init__(self, in_channels=1):
        super(CI, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x

# input_tensor = torch.randn(2, 1, 256, 256).to(device)
# si_module = SI().to(device)
# si_output = si_module(input_tensor)
# print("SI output shape:", si_output.shape)
#
# ci_module = CI().to(device)
# ci_output = ci_module(input_tensor)
# print("CI output shape:", ci_output.shape)