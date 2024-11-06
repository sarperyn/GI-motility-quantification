import numpy as np


def compute_dice_score(preds, targets, smooth=1e-6):

    # Calculate Dice Score (2 * intersection / (union + smooth))
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def compute_dice_score_samplewise(preds, targets, smooth=1e-6):
    
    batch_size = preds.shape[0]
    dice = 0.0
    for i in range(batch_size):
        intersection = (preds[i] * targets[i]).sum()
        union = preds[i].sum() + targets[i].sum()
        dice += (2. * intersection + smooth) / (union + smooth)
    return (dice / batch_size).item()


def compute_dice_score_np(preds, targets, smooth=1e-6):

    # Calculate Dice Score (2 * intersection / (union + smooth))
    intersection = np.sum(preds * targets)
    union = np.sum(preds) + np.sum(targets)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice