import os
import json
import random
import numpy as np

import torch

def make_folder(path):
    """
    Create a folder at directory of the path
    """
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

def set_random_seed(random_seed):
    """
    Generalizing random seed
    """
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        # Important for reproducability
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    return

def get_timestamp():
    """
    :return: a string timestamp
    """
    return str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")

def load_config(dir):
    """
    Look for and load config.json in the save dir
    """
    config_path = os.path.join(dir, "cust_train_config.json")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def param_count(model):
    """
    Calculate the number of trainable and total parameters in the model.

    :param model: The model to count parameters for
    :type model: torch.nn.Module
    :returns: Total number of parameters, trainable parameters, and the fraction that is trainable
    :rtype: tuple
    """
    params = [(p.numel(), p.requires_grad) for p in model.parameters()]
    trainable = sum([count for count, trainable in params if trainable])
    total = sum([count for count, _ in params])
    frac = (trainable / total) * 100
    return total, trainable, frac

def save_losses(loss_dict, loss_dir):
    make_folder(loss_dir)
    for key, value in loss_dict.items():
        np.save(os.path.join(loss_dir, f'{key}.npy'), np.array(value))

def load_losses(loss_dir):
    # Get files in directory
    f = []
    for (dirpath, dirnames, filenames) in os.walk(loss_dir):
        f.extend(filenames)
        break

    loss_dict = {}
    for f in filenames:
        arr = np.load(os.path.join(loss_dir, f))
        loss_dict[f.split(".npy")[0]] = arr

    return loss_dict