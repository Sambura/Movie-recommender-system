import numpy as np
import random
import torch

def seed_everything(seed, deterministic_cudnn=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic_cudnn
