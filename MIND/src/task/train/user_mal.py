import random
import wandb

import torch
from torch.utils.data import DataLoader

from task.train.user import UserTrainTask
from data import get_dataset, get_collate_fn


class UserMalTrainTask(UserTrainTask):
    '''Get per user gradients and aggregate them'''
    def train_on_step(self, step):
        users = random.sample(self.user_indices.keys(), self.args.user_num)

        # set mal user as {}, since this task is for robust baseline
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
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=train_collate_fn,
        )
        user_sample = len(train_ds)

        self.module._module.load_state_dict(self.agg.model.state_dict())

        self.module.train()
        loss = 0

        # get update of benigh users
        for cnt, data in enumerate(train_dl):
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(self.device)

            bz_loss, y_hat = self.module(data)

            loss += bz_loss.detach().cpu().numpy()
            bz_loss.backward()

            user_grad, user_sample_num = self.process_grad(self.module._module, data["user_mask_matrix"])
            self.agg.collect(user_grad, user_sample_num)
            self.module.zero_grad(set_to_none=True)

        loss = loss / (cnt + 1)
        self.agg.update(user_sample)

        if "debug" not in self.args.run_name:
            wandb.log({"train loss": loss}, step=step + 1)
