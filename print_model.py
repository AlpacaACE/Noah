import os
import platform
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from model.model import Transformer
from model.LMConfig_print import LMConfig_print
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_model():
    lm_config = LMConfig_print()
    device = "cpu"
    # model init
    model = Transformer(lm_config).to(device)
    
    # 解除注释，则为继续预训练
    # moe_path = '_moe' if lm_config.use_moe else ''
    # ckp = f'{save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'
    #
    # state_dict = torch.load(ckp, map_location=device)
    # unwanted_prefix = '_orig_mod.'
    # for k, v in list(state_dict.items()):
    #     if k.startswith(unwanted_prefix):
    #         state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    # model.load_state_dict(state_dict, strict=False)

    print(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万 = {count_parameters(model) / 1e9} B (Billion)')
    return model

if __name__ == "__main__":
    model = init_model()
    print(model)