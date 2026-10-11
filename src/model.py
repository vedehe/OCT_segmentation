import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


class UNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        num_classes=11
    ):

        super().__init__()


        # ====================================================
        # ENCODER
        # ====================================================

        self.enc1 = DoubleConv(
            in_channels,
            64
        )

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.enc4 = DoubleConv(
            256,
            512
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2
        )


        # ====================================================
        # BOTTLENECK
        # ====================================================

        self.bottleneck = DoubleConv(
            512,
            1024
        )


        # ====================================================
        # DECODER
        # ====================================================

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )


        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )


        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )


        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )


        # ====================================================
        # OUTPUT
        # ====================================================

        self.output = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )


    def forward(self, x):

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        x1 = self.enc1(x)

        x2 = self.enc2(
            self.pool(x1)
        )

        x3 = self.enc3(
            self.pool(x2)
        )

        x4 = self.enc4(
            self.pool(x3)
        )


        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        x5 = self.bottleneck(
            self.pool(x4)
        )


        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        x = self.up4(x5)

        x = torch.cat(
            [x, x4],
            dim=1
        )

        x = self.dec4(x)


        x = self.up3(x)

        x = torch.cat(
            [x, x3],
            dim=1
        )

        x = self.dec3(x)


        x = self.up2(x)

        x = torch.cat(
            [x, x2],
            dim=1
        )

        x = self.dec2(x)


        x = self.up1(x)

        x = torch.cat(
            [x, x1],
            dim=1
        )

        x = self.dec1(x)


        # ----------------------------------------------------
        # Segmentation output
        # ----------------------------------------------------

        return self.output(x)