import numpy as np


def compute_dice_score(preds, targets, smooth=1e-6):

    # Calculate Dice Score (2 * intersection / (union + smooth))
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def compute_dice_score_np(preds, targets, smooth=1e-6):

    # Calculate Dice Score (2 * intersection / (union + smooth))
    intersection = np.sum(preds * targets)
    union = np.sum(preds) + np.sum(targets)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice