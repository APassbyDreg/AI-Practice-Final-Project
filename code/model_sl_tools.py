import os
import torch

from torch.nn import Module


def save_ckpt(model: Module, ckpt_name, ckpt_dir):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    path = os.path.join(ckpt_dir, ckpt_name)
    torch.save(model.state_dict(), path)
    print(f"INFO: model saved to {path}")


def load_ckpt(model: Module, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    print(f"INFO: model loaded from {ckpt_path}")

