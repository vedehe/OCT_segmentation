
import torch

from metrics import dice_score, iou_score, predictions_from_logits

num_classes = 8

# Simulated model output.
logits = torch.randn(2, num_classes, 64, 64)
prediction = predictions_from_logits(logits)

# Use the prediction itself as the target to test perfect agreement.
target = prediction.clone()

dice = dice_score(prediction, target, num_classes)
iou = iou_score(prediction, target, num_classes)

print("Prediction shape:", tuple(prediction.shape))
print("Dice score:", dice.item())
print("IoU score:", iou.item())

assert torch.isclose(dice, torch.tensor(1.0, device=dice.device))
assert torch.isclose(iou, torch.tensor(1.0, device=iou.device))

print("Metrics test passed.")
