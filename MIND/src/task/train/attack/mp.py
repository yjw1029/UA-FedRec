import random
import wandb

import torch
from torch.utils.data import DataLoader

from task.train.user import UserTrainTask
from data import get_dataset, get_collate_fn


class ModelPoisonTrainTask(UserTrainTask):
    def process_mal_grad(self, model, user_mask_matrix):
        user_grad = {}
        user_sample_num = torch.sum(user_mask_matrix, dim=1).cpu()
        for name, param in model.named_parameters():
            if param.requires_grad:
                user_grad[name] = torch.einsum(
                    "ub,b...->u...", user_mask_matrix, param.grad_sample
                ).cpu()
        return user_grad, user_sample_num


    def accum_mal_grad(
        self, accu_mal_grads, mal_user_grad, accu_mal_sample_num, mal_user_sample_num
    ):
        for name in mal_user_grad:
            if name not in accu_mal_grads:
                accu_mal_grads[name] = mal_user_grad[name]
            else:
                accu_mal_grads[name] += mal_user_grad[name]

        if accu_mal_sample_num is None:
            accu_mal_sample_num = mal_user_sample_num
        else:
            accu_mal_sample_num += mal_user_sample_num
        return accu_mal_grads, accu_mal_sample_num

    def process_accu_mal_grads(self, accu_mal_grads, accu_mal_sample_num):
        mal_grad_mean = {}
        mal_grad_std = {}

        for name in accu_mal_grads:
            accu_mal_grads[name] = accu_mal_grads[name] / accu_mal_sample_num.view(
                -1, *((1,) * len(accu_mal_grads[name].shape[1:]))
            ).expand(-1, *accu_mal_grads[name].shape[1:])
            mal_grad_mean[name] = torch.mean(accu_mal_grads[name], dim=0)
            mal_grad_std[name] = torch.std(accu_mal_grads[name], dim=0)

        mal_sample_num_mean = torch.mean(accu_mal_sample_num)
        mal_sample_num_std = torch.std(accu_mal_sample_num)
        return mal_grad_mean, mal_grad_std, mal_sample_num_mean, mal_sample_num_std

    def gen_mal_grad(
        self,
        mal_grad_mean,
        mal_grad_std,
        mal_sample_num_mean,
        mal_sample_num_std,
        mal_user_num,
        mal_factor,
    ):
        mal_grad = {}

        # do not add more user sample
        mal_sample_num = int(mal_sample_num_mean + 0.0 * mal_sample_num_std)
        for name in mal_grad_mean:
            grad_mean = mal_grad_mean[name]
            grad_std = mal_grad_std[name]
            mal_grad[name] = (
                grad_mean - mal_factor * torch.sign(grad_mean) * grad_std
            ) * mal_sample_num
            # filter nan values
            mal_grad[name][torch.isnan(mal_grad[name])] = 0.0
            mal_grad[name] = (
                mal_grad[name]
                .unsqueeze(0)
                .expand(mal_user_num, *((-1,) * len(mal_grad[name].shape)))
            )
        return mal_grad, mal_sample_num

    def train_on_step(self, step):
        users = random.sample(self.user_indices.keys(), self.args.user_num)

        step_benign_users = set(users) - self.mal_users
        step_mal_users = set(users) & self.mal_users

        train_ds = get_dataset(self.config["train_dataset_name"])(
            self.args,
            self.train_sam,
            users,
            self.user_indices,
            self.nid2index,
            self.news_index,
            mal_users=step_mal_users,
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

            user_grad, user_sample_num = self.process_grad(
                self.module._module, data["user_mask_matrix"]
            )
            self.agg.collect(user_grad, user_sample_num)
            self.module.zero_grad(set_to_none=True)

        loss = loss / (cnt + 1)

        mal_mean, mal_std, mal_grad, mal_sample_num = None, None, None, None
        if len(step_mal_users) != 0:
            # use a small set of mal users to calculate mean and std
            sample_mal_users = random.sample(self.mal_users, self.args.user_num)
            train_ds_mal = get_dataset(self.config["train_dataset_name"])(
                self.args,
                self.train_sam,
                sample_mal_users,
                self.user_indices,
                self.nid2index,
                self.news_index,
                mal_users={},
            )
            train_dl_mal = DataLoader(
                train_ds_mal,
                batch_size=128,
                shuffle=False,
                num_workers=0,
                collate_fn=train_collate_fn,
            )

            accu_mal_grads = {}
            accu_mal_sample_num = None
            for cnt, data in enumerate(train_dl_mal):
                for key in data:
                    if torch.is_tensor(data[key]):
                        data[key] = data[key].to(self.device)

                bz_loss, y_hat = self.module(data)
                bz_loss.backward()

                mal_user_grad, mal_user_sample_num = self.process_mal_grad(
                    self.module._module, data["user_mask_matrix"]
                )
                accu_mal_grads, accu_mal_sample_num = self.accum_mal_grad(
                    accu_mal_grads,
                    mal_user_grad,
                    accu_mal_sample_num,
                    mal_user_sample_num,
                )

                self.module.zero_grad(set_to_none=True)

            (
                mal_mean,
                mal_std,
                mal_sample_num_mean,
                mal_sample_num_std,
            ) = self.process_accu_mal_grads(accu_mal_grads, accu_mal_sample_num)
            mal_grad, mal_sample_num = self.gen_mal_grad(
                mal_mean,
                mal_std,
                mal_sample_num_mean,
                mal_sample_num_std,
                len(step_mal_users),
                self.args.mal_factor,
            )

            user_sample += mal_sample_num * len(step_mal_users)
            self.agg.collect_by_uindex(
                mal_grad,
                (torch.ones(len(step_mal_users)) * mal_sample_num),
                [train_ds.tmp_user_dict[u] for u in step_mal_users],
            )

        self.agg.update(user_sample)

        if "debug" not in self.args.run_name:
            wandb.log({"train loss": loss}, step=step + 1)
