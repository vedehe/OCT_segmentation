
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two convolutional layers with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Basic 2.5D U-Net for retinal layer segmentation.

    Input:  (B, 5, H, W)
    Output: (B, 8, H, W)

    B = batch size
    5 input channels = five adjacent OCT B-scans
    8 output channels = eight retinal-layer classes
    """

    def __init__(
        self,
        in_channels=5,
        num_classes=8,
        base_channels=32,
    ):
        super().__init__()

        c = base_channels

        # Encoder
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, 2 * c)
        self.enc3 = DoubleConv(2 * c, 4 * c)
        self.enc4 = DoubleConv(4 * c, 8 * c)

        self.pool = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = DoubleConv(8 * c, 16 * c)

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            16 * c, 8 * c, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(16 * c, 8 * c)

        self.up3 = nn.ConvTranspose2d(
            8 * c, 4 * c, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(8 * c, 4 * c)

        self.up2 = nn.ConvTranspose2d(
            4 * c, 2 * c, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(4 * c, 2 * c)

        self.up1 = nn.ConvTranspose2d(
            2 * c, c, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(2 * c, c)

        # Per-pixel class scores
        self.output = nn.Conv2d(
            c, num_classes, kernel_size=1
        )

    @staticmethod
    def concatenate_skip(x, skip):
        """Check spatial dimensions before skip concatenation."""
        if x.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                f"Skip connection mismatch: "
                f"{x.shape[-2:]} versus {skip.shape[-2:]}"
            )

        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder and skip connections
        d4 = self.up4(b)
        d4 = self.concatenate_skip(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self.concatenate_skip(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.concatenate_skip(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.concatenate_skip(d1, e1)
        d1 = self.dec1(d1)

        return self.output(d1)


if __name__ == "__main__":
    model = UNet(
        in_channels=5,
        num_classes=8,
        base_channels=32,
    )

    x = torch.randn(1, 5, 640, 304)

    model.eval()
    with torch.no_grad():
        y = model(x)

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters()
            if p.requires_grad)
    )
