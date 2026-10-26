import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Multiclass Dice loss for semantic segmentation.

    Inputs:
        logits: (B, C, H, W), raw model outputs
        targets: (B, H, W), integer class IDs from 0 to C-1
    """

    def __init__(self, num_classes=8, smooth=1e-5, include_background=True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits, targets):
        probabilities = torch.softmax(logits, dim=1)

        targets = targets.long()
        one_hot_targets = F.one_hot(
            targets, num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        # Calculate overlap across batch and spatial dimensions.
        dims = (0, 2, 3)
        intersection = (probabilities * one_hot_targets).sum(dim=dims)
        denominator = (
            probabilities.sum(dim=dims)
            + one_hot_targets.sum(dim=dims)
        )

        dice_per_class = (
            2.0 * intersection + self.smooth
        ) / (denominator + self.smooth)

        if not self.include_background:
            dice_per_class = dice_per_class[1:]

        return 1.0 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    """
    Combines Cross-Entropy and Dice losses.
    """

    def __init__(
        self,
        num_classes=8,
        ce_weight=0.5,
        dice_weight=0.5,
        include_background=True,
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice = DiceLoss(
            num_classes=num_classes,
            include_background=include_background,
        )

    def forward(self, logits, targets):
        ce_loss = self.cross_entropy(logits, targets)
        dice_loss = self.dice(logits, targets)

        return (
            self.ce_weight * ce_loss
            + self.dice_weight * dice_loss
        )


def get_loss_function(
    num_classes=8,
    ce_weight=0.5,
    dice_weight=0.5,
):
    """Return the combined loss used to train the U-Net."""
    return CombinedLoss(
        num_classes=num_classes,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
    )


