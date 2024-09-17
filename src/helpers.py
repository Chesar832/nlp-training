
import os
from typing import Tuple

import numpy as np
import pandas as pd
import yaml


def load_config(filename: str = "config.yaml", base_path: str = "") -> dict:
    """
    Loads configuration from a YAML file and returns a dictionary with the data.

    Parameters
    ----------
    filename : str, optional
        Name of the YAML file. Default is 'config.yaml'.

    base_path : str, optional
        Base path where the YAML file is located. Default is ''.

    Returns
    -------
    config : dict
        Dictionary with the loaded configuration.
    """
    file_path = os.path.join(base_path, filename)
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config