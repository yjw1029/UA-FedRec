from tqdm import tqdm
from data import get_dataset
import numpy as np
import random
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import get_dataset
from task.train.attack.adv_item import AdverItemTrainTask


class AdverItemNormTrainTask(AdverItemTrainTask):
    def restrict_mal_grad(
        self,
        mal_grad,
        mal_grad_norm_mean,
        mal_grad_norm_std,
        mal_grad_mean,
        mal_grad_std,
    ):
        # filter nan
        for name in mal_grad:
            grad_mask = torch.isnan(mal_grad[name])
            grad_mean = mal_grad_mean[name]
            grad_std = mal_grad_std[name]

            mal_grad[name][grad_mask] = (
                grad_mean - self.args.mal_factor1 * torch.sign(grad_mean) * grad_std
            )[grad_mask]

            # filter nan in mean - 3 std
            grad_mask = torch.isnan(mal_grad[name])
            mal_grad[name][grad_mask] = 0.0

        max_norm = mal_grad_norm_mean + 3 * mal_grad_norm_std
        gen_norm = torch.norm(
            torch.stack([torch.norm(g, p=2) for name, g in mal_grad.items()]),
            p=2,
            dim=0,
        )

        clip_coef = max_norm / (gen_norm + 1e-6)

        for name, p in mal_grad.items():
            mal_grad[name] = p * clip_coef.view(
                -1, *((1,) * len(mal_grad[name].shape[1:]))
            )

        return mal_grad