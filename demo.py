import torch.nn as nn
import torch
from mambaIR import VSSBlock

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=32, in_channels=1, embed_dim=32):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "Input image size doesn't match"

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=32, in_channels=1, embed_dim=32):
        super(PatchUnEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.unproj = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.num_patches, "Number of patches doesn't match"

        x = x.transpose(1, 2).view(B, self.embed_dim, int(self.img_size / self.patch_size),
                                   int(self.img_size / self.patch_size))
        x = self.unproj(x)
        return x


# Move model and input to GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)
# #  hidden_dim 为 32
# vss_block = VSSBlock(hidden_dim=32, drop_path=0.1, attn_drop_rate=0.1, d_state=16, is_light_sr=False).to(device)
#
# input_tensor = torch.randn(2, 1, 256, 256).to(device)
#
# # PatchEmbed
# patch_embed = PatchEmbed().to(device)
# patches = patch_embed(input_tensor)
# print("Patches shape:", patches.shape)
#
# x_size = (8, 8)
# output_image = vss_block(patches, x_size)
#
# # PatchUnEmbed
# patch_unembed = PatchUnEmbed().to(device)
# reconstructed_img = patch_unembed(patches)
# print("Reconstructed image shape:", reconstructed_img.shape)
