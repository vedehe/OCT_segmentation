import torch


def _validate_inputs(prediction, target, num_classes):
    """Check segmentation predictions and ground-truth labels."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: prediction {prediction.shape}, "
            f"target {target.shape}"
        )

    if prediction.ndim < 2:
        raise ValueError("Expected batched segmentation tensors.")

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    if prediction.numel() == 0:
        raise ValueError("Prediction and target cannot be empty.")


def dice_score(prediction, target, num_classes, include_background=True):
    """
    Compute mean multiclass Dice score.

    prediction and target contain class IDs, not raw logits.
    Shape: (B, H, W)
    """
    _validate_inputs(prediction, target, num_classes)

    prediction = prediction.long().reshape(-1)
    target = target.long().reshape(-1)

    scores = []

    start_class = 0 if include_background else 1

    for class_id in range(start_class, num_classes):
        pred_class = prediction == class_id
        target_class = target == class_id

        intersection = (pred_class & target_class).sum().float()
        denominator = pred_class.sum() + target_class.sum()

        # Ignore classes absent from both prediction and target.
        if denominator.item() == 0:
            continue

        scores.append(2.0 * intersection / denominator.float())

    if not scores:
        return torch.tensor(
            1.0, device=prediction.device, dtype=torch.float32
        )

    return torch.stack(scores).mean()


def iou_score(prediction, target, num_classes, include_background=True):
    """
    Compute mean multiclass Intersection over Union.

    prediction and target contain class IDs, not raw logits.
    Shape: (B, H, W)
    """
    _validate_inputs(prediction, target, num_classes)

    prediction = prediction.long().reshape(-1)
    target = target.long().reshape(-1)

    scores = []

    start_class = 0 if include_background else 1

    for class_id in range(start_class, num_classes):
        pred_class = prediction == class_id
        target_class = target == class_id

        intersection = (pred_class & target_class).sum().float()
        union = (pred_class | target_class).sum().float()

        # Ignore classes absent from both prediction and target.
        if union.item() == 0:
            continue

        scores.append(intersection / union)

    if not scores:
        return torch.tensor(
            1.0, device=prediction.device, dtype=torch.float32
        )

    return torch.stack(scores).mean()


def predictions_from_logits(logits):
    """
    Convert U-Net outputs (B, C, H, W) into class predictions (B, H, W).
    """
    if logits.ndim != 4:
        raise ValueError(
            f"Expected logits shaped (B, C, H, W), got {logits.shape}"
        )

    return torch.argmax(logits, dim=1)
