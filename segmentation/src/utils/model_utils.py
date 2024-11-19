import torch
import numpy as np
import importlib

def save_model(model, save_path):
    """
    Saves the model's state dictionary to the specified path.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        save_path (str): Path to save the model's state dictionary.

    Returns:
        None
    """
    torch.save(model.state_dict(), save_path)

def load_model(model_class, model_path, device):
    """
    Loads a model's state dictionary and initializes it.

    Args:
        model_class (class): The model class to instantiate.
        model_path (str): Path to the model's state dictionary.
        device (torch.device): Device to map the model's parameters.

    Returns:
        torch.nn.Module: The loaded model instance.
    """
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def set_seed(seed):
    """
    Sets the seed for reproducibility in NumPy and PyTorch.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_class(config):
    """
    Retrieves the model class specified in the configuration.

    Args:
        config (dict): Configuration dictionary containing model class path.

    Returns:
        class: The model class specified in the configuration.
    """
    model_class_path = config['model_config']['class_path']
    module_name, class_name = model_class_path.rsplit(".", 1)
    model_module = importlib.import_module(module_name)
    model_class = getattr(model_module, class_name)

    return model_class

def get_dataset_class(config):
    """
    Retrieves the dataset class specified in the configuration.

    Args:
        config (dict): Configuration dictionary containing dataset class path.

    Returns:
        class: The dataset class specified in the configuration.
    """
    dataset_class_path = config['dataset_config']['class_path']
    module_name, class_name = dataset_class_path.rsplit(".", 1)
    dataset_module = importlib.import_module(module_name)
    dataset_class = getattr(dataset_module, class_name)

    return dataset_class

def get_optimizer_class(config):
    """
    Retrieves the optimizer class and initialization arguments from the configuration.

    Args:
        config (dict): Configuration dictionary containing optimizer details.

    Returns:
        tuple: Optimizer class and initialization arguments.
    """
    optimizer_config = config['optimizer_config']
    optimizer_class_path = optimizer_config['class_path']
    optimizer_init_args = optimizer_config['init_args']

    module_name, class_name = optimizer_class_path.rsplit('.', 1)
    optimizer_module = importlib.import_module(module_name)
    optimizer_class = getattr(optimizer_module, class_name)

    return optimizer_class, optimizer_init_args

def get_scheduler_class(config):
    """
    Retrieves the learning rate scheduler class specified in the configuration.

    Args:
        config (dict): Configuration dictionary containing scheduler class path.

    Returns:
        class: The scheduler class specified in the configuration.
    """
    scheduler_config = config['scheduler_config']
    scheduler_class_path = scheduler_config['class_path']

    module_name, class_name = scheduler_class_path.rsplit('.', 1)
    scheduler_module = importlib.import_module(module_name)
    scheduler_class = getattr(scheduler_module, class_name)

    return scheduler_class

def get_criterion_class(config):
    """
    Retrieves the loss function based on the configuration.

    Args:
        config (dict): Configuration dictionary containing the loss type.

    Returns:
        torch.nn.Module: The loss function module.
    """
    loss_type = config['loss_type']

    if loss_type == "classification":
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError("Loss type not implemented.")

def get_class(class_path):
    """
    Retrieves a class from its module and class name path.

    Args:
        class_path (str): Full path to the class (module.submodule.ClassName).

    Returns:
        class: The specified class.
    """
    module_name, class_name = class_path.rsplit('.', 1)
    module_ = importlib.import_module(module_name)
    class_ = getattr(module_, class_name)

    return class_

def train(model, X_train, y_train):
    """
    Trains a machine learning model on the given data.

    Args:
        model (object): The machine learning model instance.
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.

    Returns:
        object: The trained model.
    """
    model.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
    return model

def predict(model, X_test):
    """
    Makes predictions using a trained model.

    Args:
        model (object): The trained model instance.
        X_test (np.ndarray): Feature data for prediction.

    Returns:
        np.ndarray: Predicted values.
    """
    return model.predict(X_test.reshape(-1, 1))
