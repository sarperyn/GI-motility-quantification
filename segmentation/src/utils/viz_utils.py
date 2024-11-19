import matplotlib.pyplot as plt
from itertools import product
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import glob
import torch
import os


def plot_results(imgs, recons, save_path, epoch, batch):
    """
    Plots original and reconstructed images side by side and saves the plot.

    Args:
        imgs (torch.Tensor): Original images.
        recons (torch.Tensor): Reconstructed images.
        save_path (str): Path to save the plot.
        epoch (int): Current epoch number.
        batch (int): Current batch number.

    Returns:
        None
    """
    bs = 8
    fig, axes = plt.subplots(nrows=2, ncols=bs, figsize=(bs, 15))

    for i, (row, col) in enumerate(product(range(2), range(bs))):
        if row == 0:
            axes[row][col].imshow(np.transpose(imgs[col].detach().cpu().numpy(), (1, 2, 0)))
            if col == 0:
                axes[row][col].set_ylabel('Original Image', fontsize=15, fontweight='bold')
        elif row == 1:
            axes[row][col].imshow(np.transpose(recons[col].detach().cpu().numpy(), (1, 2, 0)))
            if col == 0:
                axes[row][col].set_ylabel('Reconstructed Image', fontsize=15, fontweight='bold')

        axes[row][col].set_yticks([])
        axes[row][col].set_xticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join(save_path, f'fig_{epoch}_{batch}.jpg'), format='jpg', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.show()
    plt.close()


def visualize_samples(tensor, save_path, epoch, batch):
    """
    Visualizes a few samples from a tensor and saves the plot.

    Args:
        tensor (torch.Tensor): Input tensor of images.
        save_path (str): Path to save the visualization.
        epoch (int): Current epoch number.
        batch (int): Current batch number.

    Returns:
        None
    """
    tensor = tensor.cpu().numpy()
    tensor = tensor.squeeze(1)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(5):
        axes[i].imshow(tensor[i])
        axes[i].axis('off')
    plt.savefig(os.path.join(save_path, f'fig_{epoch}_{batch}.jpg'), format='jpg', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.show()
    plt.close()


def save_tensor_as_jpg(tensor, save_dir):
    """
    Saves each tensor image as a JPEG file in the specified directory.

    Args:
        tensor (torch.Tensor): Input tensor of images.
        save_dir (str): Directory to save the images.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    length_dir = len(glob.glob(os.path.join(save_dir, '*')))

    for idx, img in enumerate(tensor):
        transform = transforms.ToPILImage()
        image = transform(img)
        image.save(os.path.join(save_dir, f'{idx + length_dir}.jpeg'), 'JPEG')


def plot_from_dir(folder_path, num_sample=25):
    """
    Plots a grid of images randomly sampled from a directory.

    Args:
        folder_path (str): Path to the folder containing images.
        num_sample (int): Number of images to sample. Default is 25.

    Returns:
        None
    """
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpeg'))]
    if len(image_files) > num_sample:
        image_files = random.sample(image_files, num_sample)

    grid_size = int(num_sample ** 0.5)
    if grid_size ** 2 < num_sample:
        grid_size += 1

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for ax, img_file in zip(axes, image_files):
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path)
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    for ax in axes[len(image_files):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('figure-gray.jpg', format='jpg', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.show()


def visualize_tensor(tensor):
    """
    Visualizes a few slices of a tensor in grayscale.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        None
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()

    tensor = tensor.numpy()
    tensor = tensor.squeeze(1)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(5):
        axes[i].imshow(tensor[i], cmap='gray')
        axes[i].axis('off')
    plt.show()

def visualize_predictions(images, masks, outputs, save_path, epoch, batch_idx):
    """
    Visualizes original images, masks, and predicted outputs for a batch.

    Args:
        images (torch.Tensor): Batch of original images.
        masks (torch.Tensor): Ground truth masks.
        outputs (torch.Tensor): Model predictions (logits).
        save_path (str): Path to save the visualization.
        epoch (int): Current epoch number.
        batch_idx (int): Batch index.

    Returns:
        None
    """
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    outputs = torch.sigmoid(outputs).cpu().numpy()

    bs = min(images.shape[0], 5)  # Display up to 5 samples
    images, masks, outputs = images[:bs], masks[:bs], outputs[:bs]

    fig, axs = plt.subplots(3, bs, figsize=(10, 6))
    for i in range(bs):
        axs[0, i].imshow(images[i][0], cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(masks[i][0], cmap='gray')
        axs[1, i].axis('off')
        axs[2, i].imshow(outputs[i][0] > 0.5, cmap='gray')
        axs[2, i].axis('off')

    plt.suptitle(f'Batch {batch_idx} Predictions')
    plt.savefig(os.path.join(save_path, f'fig_{epoch}_{batch_idx}.jpg'), format='jpg', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.tight_layout()
    plt.show()


def visualize_predictions2(images, masks, outputs, save_path, batch_idx):
    """
    Visualizes original images, masks, and predictions using a custom sigmoid function.

    Args:
        images (np.ndarray): Batch of original images.
        masks (np.ndarray): Ground truth masks.
        outputs (np.ndarray): Model predictions (logits).
        save_path (str): Path to save the visualization.
        batch_idx (int): Batch index.

    Returns:
        None
    """
    sigmoid = lambda z: 1 / (1 + np.exp(-z)) 
    outputs = sigmoid(outputs)

    fig, axs = plt.subplots(3, 5, figsize=(10, 6))  # Fixed to 5 samples for display
    for i in range(5):
        axs[0, i].imshow(images[i], cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(masks[i], cmap='gray')
        axs[1, i].axis('off')
        axs[2, i].imshow(outputs[i] > 0.5, cmap='gray')
        axs[2, i].axis('off')

    plt.suptitle(f'Batch {batch_idx} Predictions')
    plt.savefig(os.path.join(save_path, f'fig_{batch_idx}.jpg'), format='jpg', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.tight_layout()
    plt.show()


def visualize_predictions_3d(images, masks, outputs, save_path, epoch, batch_idx, slice_indices=[10, 20, 30]):
    """
    Visualizes 3D images, masks, and predictions at specified slice indices.

    Args:
        images (torch.Tensor): Batch of original 3D images.
        masks (torch.Tensor): Ground truth 3D masks.
        outputs (torch.Tensor): Model predictions (logits).
        save_path (str): Path to save the visualization.
        epoch (int): Current epoch number.
        batch_idx (int): Batch index.
        slice_indices (list): Indices of slices to visualize.

    Returns:
        None
    """
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    outputs = torch.sigmoid(outputs).cpu().numpy()

    bs = min(images.shape[0], 5)  # Display up to 5 samples
    num_slices = len(slice_indices)

    fig, axs = plt.subplots(num_slices, bs * 3, figsize=(15, 6))
    for i in range(bs):
        for j, slice_idx in enumerate(slice_indices):
            axs[j, i * 3].imshow(images[i, 0, slice_idx], cmap='gray')
            axs[j, i * 3].axis('off')
            axs[j, i * 3 + 1].imshow(masks[i, 0, slice_idx], cmap='gray')
            axs[j, i * 3 + 1].axis('off')
            axs[j, i * 3 + 2].imshow(outputs[i, 0, slice_idx] > 0.5, cmap='gray')
            axs[j, i * 3 + 2].axis('off')

    plt.suptitle(f'Batch {batch_idx} Predictions - Slices {slice_indices}')
    plt.savefig(os.path.join(save_path, f'fig_{epoch}_{batch_idx}.jpg'), format='jpg', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.tight_layout()
    plt.show()


def visualize_predictions_mip(images, masks, outputs, save_path, epoch, batch_idx):
    """
    Visualizes maximum intensity projections (MIP) for images, masks, and predictions.

    Args:
        images (torch.Tensor): Batch of 3D images.
        masks (torch.Tensor): Ground truth 3D masks.
        outputs (torch.Tensor): Model predictions (logits).
        save_path (str): Path to save the visualization.
        epoch (int): Current epoch number.
        batch_idx (int): Batch index.

    Returns:
        None
    """
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    outputs = torch.sigmoid(outputs).cpu().numpy()

    bs = min(images.shape[0], 5)  # Display up to 5 samples
    fig, axs = plt.subplots(3, bs, figsize=(10, 6))
    for i in range(bs):
        axs[0, i].imshow(np.max(images[i][0], axis=0), cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(np.max(masks[i][0], axis=0), cmap='gray')
        axs[1, i].axis('off')
        axs[2, i].imshow(np.max(outputs[i][0] > 0.5, axis=0), cmap='gray')
        axs[2, i].axis('off')

    plt.suptitle(f'Batch {batch_idx} MIP Predictions')
    plt.savefig(os.path.join(save_path, f'fig_{epoch}_{batch_idx}_mip.jpg'), format='jpg', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.tight_layout()
    plt.show()


def plot_metric(x, label, plot_dir, args, metric):
    """
    Plots a metric curve (e.g., loss, accuracy) and saves it.

    Args:
        x (list): Metric values across epochs.
        label (str): Label for the metric (e.g., 'Loss').
        plot_dir (str): Directory to save the plot.
        args (Namespace): Training arguments (contains 'mode').
        metric (str): Name of the metric.

    Returns:
        None
    """
    plt.figure()
    plt.plot(x, label=label)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'{args.mode}_{metric}_curves.jpg'))
    plt.show()
    plt.close()


def plot_train_val_history(train_loss_history, val_loss_history, plot_dir, args):
    """
    Plots training and validation loss curves and saves the plot.

    Args:
        train_loss_history (list): Training loss values across epochs.
        val_loss_history (list): Validation loss values across epochs.
        plot_dir (str): Directory to save the plot.
        args (Namespace): Training arguments (contains 'exp_id').

    Returns:
        None
    """
    plt.figure()
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, args.exp_id, 'train_loss_curves.jpg'))
    plt.show()
    plt.close()
