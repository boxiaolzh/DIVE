import csv
import os
import torch
import random
import math
import yaml
import logging
import numpy as np


def seed_set(seed):
    """
        fix the random seed
    """
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def load_config(file_path):
    """Load configuration parameters from a YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    logging.info("Configuration loaded from %s", file_path)
    return config


def get_bounds_from_config(config, experiment):
    """Extract parameter bounds from the configuration and convert any string values to floats."""
    bounds = []
    for k, v in config[experiment]['full_parameter_bounds'].items():
        # 转换每个边界值为浮点数
        bound = (float(v[0]), float(v[1])) if isinstance(v[1], str) else (float(v[0]), v[1])
        bounds.append(bound)
    return bounds
