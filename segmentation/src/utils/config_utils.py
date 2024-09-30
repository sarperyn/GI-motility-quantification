import yaml

def load_config(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config
