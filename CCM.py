import torch
import torch.nn as nn
from mambaIR import VSSBlock
from demo import PatchEmbed, PatchUnEmbed
from CI_SI import CI, SI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DEPTHWISECONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DEPTHWISECONV, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class CCM(nn.Module):
    def __init__(self):
        super(CCM, self).__init__()
        self.normalization = nn.BatchNorm2d(1)
        self.conv1x1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
        self.conv3   = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.depthwise_conv = DEPTHWISECONV(in_ch=1, out_ch=1)
        self.VSS = VSSBlock(hidden_dim=32, drop_path=0.1, attn_drop_rate=0.1, d_state=2, is_light_sr=False)
        self.patch = PatchEmbed()
        self.patch_embed = PatchUnEmbed()
        self.CI = CI()
        self.SI = SI()

    def forward(self, x_i):
        x_D1 = self.depthwise_conv(x_i)
        x_D = self.SI(x_D1)
        x = self.normalization(x_i)
        x = self.conv1x1(x)
        x1 = self.conv3x3(x)
        x = self.patch(x1)
        x = self.VSS(x, (8, 8))
        x_U1 = self.patch_embed(x)
        x_U1 = x_U1 + x_i
        x_U1 = self.conv3(x_U1) + x_U1
        x_U = self.CI(x_U1)
        x_D = x_D * x_U1
        x_U = x_D1 * x_U
        x = x_U + x_D
        return x

model = CCM().to(device)

input_tensor = torch.randn(2, 1, 256, 256).to(device)

output = model(input_tensor)

print("Output tensor shape:", output.shape)
