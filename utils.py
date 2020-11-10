import os
import math
import random
import numpy as np
import argparse
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def set_seed(seed):
    # for reproducibility. 
    # note that pytorch is not completely reproducible 
    # https://pytorch.org/docs/stable/notes/randomness.html  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.initial_seed() # dataloader multi processing 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def global_avgpool2d(x): #[N, C, H, W] -> [N, C]
    return torch.mean(x, dim=(2, 3))

def winner_take_all(x, sparsity_ratio):
    C = x.size(1)
    k = int(sparsity_ratio*C)
    val, idx = torch.topk(x, C-k, dim=-1, largest=False)
    winner_mask = torch.ones_like(x)

    return x.scatter_(1, idx, 0), winner_mask.scatter_(1, idx, 0)




