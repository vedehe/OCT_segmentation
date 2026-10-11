import torch


def dice_score(
    prediction,
    target,
    num_classes
):

    prediction = prediction.view(-1)
    target = target.view(-1)

    dice_scores = []

    for class_id in range(num_classes):

        pred_class = (
            prediction == class_id
        )

        target_class = (
            target == class_id
        )

        intersection = (
            pred_class & target_class
        ).sum().float()

        denominator = (
            pred_class.sum()
            +
            target_class.sum()
        ).float()

        if denominator == 0:

            continue

        dice = (
            2 * intersection
            /
            denominator
        )

        dice_scores.append(dice)

    if len(dice_scores) == 0:

        return torch.tensor(1.0)

    return torch.stack(
        dice_scores
    ).mean()


def iou_score(
    prediction,
    target,
    num_classes
):

    prediction = prediction.view(-1)
    target = target.view(-1)

    iou_scores = []

    for class_id in range(num_classes):

        pred_class = (
            prediction == class_id
        )

        target_class = (
            target == class_id
        )

        intersection = (
            pred_class & target_class
        ).sum().float()

        union = (
            pred_class | target_class
        ).sum().float()

        if union == 0:

            continue

        iou = intersection / union

        iou_scores.append(iou)

    if len(iou_scores) == 0:

        return torch.tensor(1.0)

    return torch.stack(
        iou_scores
    ).mean()