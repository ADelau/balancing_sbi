import yaml
from yaml.loader import FullLoader

def read_config(config_file_path):
    """Read config file

    Args:
        config_file_path (str): the path to the config file

    Returns:
        dict : the config
    """
    
    config = yaml.load(open(config_file_path), Loader=FullLoader)
    return config