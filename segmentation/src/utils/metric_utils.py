import numpy as np

def dice_coefficient(pred, target, threshold=0.5):

    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    pred = (pred > threshold).astype(np.uint8).flatten()
    target = (target > threshold).astype(np.uint8).flatten()
    
    intersection = np.sum(pred * target)
    return (2. * intersection) / (np.sum(pred) + np.sum(target) + 1e-6)

