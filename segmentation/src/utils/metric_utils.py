import numpy as np

def compute_dice_score(preds, targets, smooth=1e-6):
    """
    Computes the Dice score for binary segmentation tasks.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted binary mask (1 for foreground, 0 for background).
        targets (torch.Tensor or np.ndarray): Ground truth binary mask.
        smooth (float, optional): Smoothing factor to avoid division by zero. Default is 1e-6.

    Returns:
        float: The Dice score, a value between 0 and 1 where 1 indicates perfect overlap.
    """
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def compute_dice_score_samplewise(preds, targets, smooth=1e-6):
    """
    Computes the Dice score for each sample in a batch and averages the scores.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted binary masks of shape (batch_size, ...).
        targets (torch.Tensor or np.ndarray): Ground truth binary masks of shape (batch_size, ...).
        smooth (float, optional): Smoothing factor to avoid division by zero. Default is 1e-6.

    Returns:
        float: The average Dice score across all samples in the batch.
    """
    batch_size = preds.shape[0]
    dice = 0.0
    for i in range(batch_size):
        intersection = (preds[i] * targets[i]).sum()
        union = preds[i].sum() + targets[i].sum()
        dice += (2. * intersection + smooth) / (union + smooth)
    return (dice / batch_size).item()

def compute_dice_score_np(preds, targets, smooth=1e-6):
    """
    Computes the Dice score for binary segmentation tasks using NumPy arrays.

    Args:
        preds (np.ndarray): Predicted binary mask (1 for foreground, 0 for background).
        targets (np.ndarray): Ground truth binary mask.
        smooth (float, optional): Smoothing factor to avoid division by zero. Default is 1e-6.

    Returns:
        float: The Dice score, a value between 0 and 1 where 1 indicates perfect overlap.
    """
    intersection = np.sum(preds * targets)
    union = np.sum(preds) + np.sum(targets)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice
