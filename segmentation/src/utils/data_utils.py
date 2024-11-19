import pandas as pd
import cv2

def load_csv(filepath):
    """
    Loads a CSV file into a Pandas DataFrame.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the CSV file.
    """
    return pd.read_csv(filepath)

def save_csv(dataframe, filepath):
    """
    Saves a Pandas DataFrame to a CSV file.

    Args:
        dataframe (pd.DataFrame): The DataFrame to save.
        filepath (str): Path where the CSV file will be saved.

    Returns:
        None
    """
    dataframe.to_csv(filepath, index=False)

def read_image(filepath):
    """
    Reads an image from a file using OpenCV.

    Args:
        filepath (str): Path to the image file.

    Returns:
        np.ndarray: The image as a NumPy array.
    """
    return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

def min_max_normalize(tensor, min_val=0.0, max_val=1.0):
    """
    Applies min-max normalization to a tensor or NumPy array.

    Args:
        tensor (torch.Tensor or np.ndarray): The input data to normalize.
        min_val (float, optional): Minimum value of the normalized range. Default is 0.0.
        max_val (float, optional): Maximum value of the normalized range. Default is 1.0.

    Returns:
        torch.Tensor or np.ndarray: The normalized data with values scaled to the range [min_val, max_val].
    """
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * (max_val - min_val) + min_val
    return normalized_tensor
