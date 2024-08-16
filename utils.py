import os
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

