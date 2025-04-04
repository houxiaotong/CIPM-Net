import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.project = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        print(x.shape)
        x = self.project(x)  
        print(x.shape)
        x = x.flatten(2)  
        print(x.shape)
        x = x.transpose(1, 2)  
        print(x.shape)
        return x


if __name__ == "__main__":
    x = torch.rand([1, 3, 224, 224])

    model = PatchEmbed()
    y = model(x)
    print(y.shape)