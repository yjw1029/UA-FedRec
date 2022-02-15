import random
import wandb

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from task.train.base import BaseTrainTask, process_grad
from data import get_dataset, get_collate_fn


class MalTrainTask(BaseTrainTask):
    def __init__(self, args, config, device):
        super().__init__(args, config, device)

        self.mal_users = set(
            random.sample(
                self.user_indices.keys(),
                int(len(self.user_indices) * args.mal_user_ratio),
            )
        )

    def train_on_step(self, step):
        users = random.sample(self.user_indices.keys(), self.args.user_num)
        train_ds = get_dataset(self.config["train_dataset_name"])(
            self.args,
            self.train_sam,
            users,
            self.user_indices,
            self.nid2index,
            self.news_index,
            mal_users=self.mal_users,
            news_ctr=self.news_ctr,
        )
        train_collate_fn = get_collate_fn(self.config["train_collate_fn_name"])
        train_dl = DataLoader(
            train_ds,
            collate_fn=train_collate_fn,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        user_sample = len(train_ds)

        self.model.load_state_dict(self.agg.model.state_dict())
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)

        self.model.train()
        loss = 0
        for cnt, data in enumerate(train_dl):
            sample_num = data["batch_label"].shape[0]

            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(self.device)

            bz_loss, y_hat = self.model(data)

            # compute gradients for user model and news representations
            loss += bz_loss.detach().cpu().numpy()

            optimizer.zero_grad()
            bz_loss.backward()

            model_grad = process_grad(
                self.model.named_parameters(), sample_num, user_sample
            )
            self.agg.collect(model_grad)

        loss = loss / (cnt + 1)
        self.agg.update()

        if "debug" not in self.args.run_name:
            wandb.log({"train loss": loss}, step=step + 1)
