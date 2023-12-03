import numpy as np
import inspect
import random
import torch

def seed_everything(seed, deterministic_cudnn=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic_cudnn

def replace_with_defaults(param_dict, ref_func):
    result = {}
    ref_params = inspect.signature(ref_func).parameters
    
    for param_name, param_value in param_dict.items():
        if param_value is None and param_name in ref_params:
            param_value = ref_params[param_name].default

        result[param_name] = param_value
    
    return result

def get_device(model: torch.nn.Module):
    return next(model.parameters()).device
