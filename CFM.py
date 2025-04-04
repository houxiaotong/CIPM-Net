import torch
import torch.nn as nn
from mambaIR import VSSBlock
from demo import PatchEmbed, PatchUnEmbed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Curved_state_space_model(nn.Module):
    def __init__(self, channel):
        super(Curved_state_space_model, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.n_curve = 3
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
        a = self.predict_a(x)
        x = self.relu(x) - self.relu(x - 1)
        for i in range(self.n_curve):
            x = x + a[:, i:i + 1] * x * (1 - x)
        return x


#
class Causal_Frequency_Mamba(nn.Module):
    def __init__(self, in_channels=1):
        super(Causal_Frequency_Mamba, self).__init__()
        self.conv1 = nn.Conv2d(2, in_channels, kernel_size=3, padding=1)  # 输入通道数改为 2
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.curve_ca_layer = Curved_state_space_model(in_channels)
        self.conv3 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)  # 输出通道数改为 2
        self.VSS = VSSBlock(hidden_dim=32, drop_path=0.1, attn_drop_rate=0.1, d_state=2, is_light_sr=False)
        self.patch = PatchEmbed()
        self.patch_embed = PatchUnEmbed()

    def forward(self, x):
        x_input = x.to(device)
        x = torch.fft.fft2(x_input, dim=(-2, -1))
        x = torch.view_as_real(x)
        x_f = x.permute(0, 4, 1, 2, 3).contiguous().view(x.size(0), 2 * x.size(1), x.size(2), x.size(3))
        x = self.conv1(x_f)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.curve_ca_layer(x)
        x = self.conv3(x)
        x = self.patch(x)
        x = self.VSS(x, (8, 8))
        x = self.patch_embed(x)
        x = x + x_f
        x = x.view(x.size(0), 2, -1, x.size(2), x.size(3)).permute(0, 2, 3, 4, 1)
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.fft.ifft2(x, dim=(-2, -1))
        x = torch.real(x)
        x = x + x_input
        return x


input_tensor = torch.randn(2, 1, 256, 256).to(device)
processing_module = Causal_Frequency_Mamba().to(device)

output = processing_module(input_tensor)

print("Output tensor shape:", output.shape)