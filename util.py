import os
import numpy as np


# Define the function to load the module
def load_params(path_str):
    import importlib.util

    # Construct the path to the set_params.py file
    params_path = os.path.join(path_str, "set_params.py")

    # Load the module
    spec = importlib.util.spec_from_file_location("set_params", params_path)
    set_params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(set_params)

    return set_params
