"""
DCGAN model definitions: Generator, Discriminator, and weight initialization
following the original DCGAN paper (Radford et al., 2015).
"""
import torch.nn as nn


def weights_init_dcgan(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """Maps a latent vector z -> a 64x64 RGB image in [-1, 1]."""

    def __init__(self, nz: int, ngf: int, nc: int):
        super().__init__()
        self.main = nn.Sequential(
            # input Z: [B, nz, 1, 1]
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            # [B, ngf*8, 4, 4]
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            # [B, ngf*4, 8, 8]
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            # [B, ngf*2, 16, 16]
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            # [B, ngf, 32, 32]
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),  # -> [-1, 1], [B, nc, 64, 64]
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """Maps a 64x64 RGB image -> a single real/fake logit."""

    def __init__(self, ndf: int, nc: int):
        super().__init__()
        self.main = nn.Sequential(
            # [B, nc, 64, 64]
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            # [B, ndf, 32, 32]
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            # [B, ndf*2, 16, 16]
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            # [B, ndf*4, 8, 8]
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),
            # [B, ndf*8, 4, 4]
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        out = self.main(x)   # [B, 1, 1, 1]
        return out.view(-1)  # logits


def make_dcgan(nz: int, ngf: int, ndf: int, nc: int, device):
    """Build and initialize a fresh Generator/Discriminator pair."""
    G = Generator(nz, ngf, nc).to(device)
    D = Discriminator(ndf, nc).to(device)
    G.apply(weights_init_dcgan)
    D.apply(weights_init_dcgan)
    return G, D
