import torch
import numpy as np
import importlib

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def load_model(model_class, model_path, device):
    model = model_class()
    model = torch.load(model_path, map_location=device)
    return model

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_class(config):

    model_class_path = config['model_config']['class_path']
    module_name, class_name = model_class_path.rsplit(".", 1)
    model_module = importlib.import_module(module_name)
    model_class = getattr(model_module, class_name)

    return model_class

def get_dataset_class(config):

    dataset_class_path = config['dataset_config']['class_path']
    module_name, class_name = dataset_class_path.rsplit(".", 1)
    dataset_module = importlib.import_module(module_name)
    dataset_class  = getattr(dataset_module, class_name)

    return dataset_class

def get_optimizer_class(config):

    optimizer_config = config['optimizer_config']
    optimizer_class_path = optimizer_config['class_path']
    optimizer_init_args = optimizer_config['init_args']

    module_name, class_name = optimizer_class_path.rsplit('.', 1)
    optimizer_module = importlib.import_module(module_name)
    optimizer_class = getattr(optimizer_module, class_name)

    return optimizer_class, optimizer_init_args

def get_scheduler_class(config):

    scheduler_config = config['scheduler_config']
    scheduler_class_path = scheduler_config['class_path']

    module_name, class_name = scheduler_class_path.rsplit('.', 1)
    scheduler_module = importlib.import_module(module_name)
    scheduler_class = getattr(scheduler_module, class_name)

    return scheduler_class

def get_criterion_class(config):

    loss_type  = config['loss_type']

    if loss_type == "classification":
        return torch.nn.BCEWithLogitsLoss()
    
    else:
        raise NotImplementedError

def get_class(class_path):

    module_name, class_name = class_path.rsplit('.', 1)
    module_ = importlib.import_module(module_name)
    class_ = getattr(module_, class_name)

    return class_

def train(model, X_train, y_train):
    model.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
    return model

def predict(model, X_test):
    return model.predict(X_test.reshape(-1, 1))