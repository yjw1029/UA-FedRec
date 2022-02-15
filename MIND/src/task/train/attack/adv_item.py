from tqdm import tqdm
from data import get_dataset
import numpy as np
import random
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import get_dataset, get_collate_fn
from task.train.attack.mp import ModelPoisonTrainTask
from data.adv_item import NewsAdvItemDataset
from model import get_model


class AdverItemTrainTask(ModelPoisonTrainTask):
    def __init__(self, args, config, device):
        super().__init__(args, config, device)

        # (candidate_idx, positive_idx, negative_idx)
        self.news_pairs = None

    def load_model(self):
        super().load_model()
        self.adv_loss_fn = nn.MSELoss()

    def process_accu_mal_grads(self, accu_mal_grads, accu_mal_sample_num):
        mal_grad_mean = {}
        mal_grad_std = {}

        for name in accu_mal_grads:
            accu_mal_grads[name] = accu_mal_grads[name] / accu_mal_sample_num.view(
                -1, *((1,) * len(accu_mal_grads[name].shape[1:]))
            ).expand(-1, *accu_mal_grads[name].shape[1:])
            mal_grad_mean[name] = torch.mean(accu_mal_grads[name], dim=0)
            mal_grad_std[name] = torch.std(accu_mal_grads[name], dim=0)

        mal_grad_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(g, p=2, dim=list(range(len(g.shape)))[1:])
                    for name, g in accu_mal_grads.items()
                    if name.startswith("text_encoder")
                ]
            ),
            p=2,
            dim=0,
        )
        mal_grad_norm_mean = torch.mean(mal_grad_norm)
        mal_grad_norm_std = torch.std(mal_grad_norm)

        mal_sample_num_mean = torch.mean(accu_mal_sample_num)
        mal_sample_num_std = torch.std(accu_mal_sample_num)
        return (
            mal_grad_mean,
            mal_grad_std,
            mal_sample_num_mean,
            mal_sample_num_std,
            mal_grad_norm_mean,
            mal_grad_norm_std,
        )

    def gen_mal_user_grad(
        self,
        mal_grad_mean,
        mal_grad_std,
        mal_sample_num_mean,
        mal_sample_num_std,
        mal_user_num
    ):
        mal_grad = {}

        # do not add more user sample
        mal_sample_num = int(mal_sample_num_mean + self.args.mal_factor3 * mal_sample_num_std)
        for name in mal_grad_mean:
            if name.startswith("user_encoder"):
                grad_mean = mal_grad_mean[name]
                grad_std = mal_grad_std[name]
                mal_grad[name] = (
                    grad_mean - self.args.mal_factor2 * torch.sign(grad_mean) * grad_std
                ) * mal_sample_num
                # filter nan values
                mal_grad[name][torch.isnan(mal_grad[name])] = 0.0
                mal_grad[name] = (
                    mal_grad[name]
                    .unsqueeze(0)
                    .expand(mal_user_num, *((-1,) * len(mal_grad[name].shape)))
                )
        return mal_grad, mal_sample_num

    def update_news_pair(self):
        # inference news_emb
        news_dataset = get_dataset(self.config["news_dataset_name"])(self.news_index)
        news_dl = DataLoader(news_dataset, batch_size=512, shuffle=False, num_workers=0)
        news_vecs = []
        self.agg.model.eval()
        with torch.no_grad():
            for news in tqdm(news_dl):
                news = news.to(self.device)
                news_vec = self.agg.model.text_encoder(news).detach().cpu().numpy()
                news_vecs.append(news_vec)
            news_vecs = np.concatenate(news_vecs)

        news_vecs = torch.FloatTensor(news_vecs).to(self.device)

        self.news_pairs = []
        with torch.no_grad():
            for idx in range(len(news_vecs)):
                candiate_news = news_vecs[idx]
                scores = torch.mm(news_vecs, candiate_news.unsqueeze(-1)).squeeze(-1)

                indices = torch.argsort(scores, descending=True)
                indices = indices[indices != idx]
                positive_news_idx = indices[0]
                negative_news_idx = indices[-1]
                self.news_pairs.append(
                    (
                        idx,
                        positive_news_idx.detach().cpu().numpy(),
                        negative_news_idx.detach().cpu().numpy(),
                    )
                )
    
    def restrict_mal_grad(
        self,
        mal_grad,
        mal_grad_norm_mean,
        mal_grad_norm_std,
        mal_grad_mean,
        mal_grad_std,
    ):
        # Do nothing
        return mal_grad

    def gen_mal_news_grad(
        self,
        mal_user_num,
        mal_grad_norm_mean,
        mal_grad_norm_std,
        mal_grad_mean,
        mal_grad_std,
        mal_sample_num_mean,
        mal_sample_num_std,
    ):

        adv_item_ds = NewsAdvItemDataset(self.news_index, self.news_pairs)
        adv_item_dl = DataLoader(
            adv_item_ds, batch_size=128, shuffle=True, num_workers=0
        )

        model_cls = get_model(self.config["model_name"])
        model = model_cls(self.args).to(self.device)
        model.load_state_dict(self.agg.model.state_dict())

        optimizer = optim.SGD(model.parameters(), lr=self.args.mal_adv_lr)
        iter_num = len(adv_item_ds) // 128
        for step, (candidate_nindex, pos_nindex, neg_nindex) in enumerate(adv_item_dl):
            candidate_nindex = candidate_nindex.to(self.device)
            pos_nindex = pos_nindex.to(self.device)
            neg_nindex = neg_nindex.to(self.device)

            canidate_news_emb = model.text_encoder(candidate_nindex)
            pos_news_emb = model.text_encoder(pos_nindex)
            neg_news_emb = model.text_encoder(neg_nindex)

            loss1 = self.adv_loss_fn(canidate_news_emb, pos_news_emb)
            loss2 = self.adv_loss_fn(canidate_news_emb, neg_news_emb)

            loss = (loss2 - loss1)
            loss.backward()

        optimizer.step()

        mal_grad = {}
        model_state_dict = model.state_dict()
        for name, param in self.agg.model.named_parameters():
            mal_grad[name] = (
                model_state_dict[name] - param
            ).detach().cpu() * self.args.adv_alpha

        mal_grad = self.restrict_mal_grad(
            mal_grad, mal_grad_norm_mean, mal_grad_norm_std, mal_grad_mean, mal_grad_std
        )

        mal_sample_num = int(mal_sample_num_mean + self.args.mal_factor * mal_sample_num_std)
        for name, param in self.agg.model.named_parameters():
            
            mal_grad[name] = (mal_grad[name] * mal_sample_num).expand(
                mal_user_num, *((-1,) * len(mal_grad[name].shape))
            )

        return mal_grad

    def gen_mal_grad(
        self,
        mal_grad_mean,
        mal_grad_std,
        mal_sample_num_mean,
        mal_sample_num_std,
        mal_grad_norm_mean,
        mal_grad_norm_std,
        mal_user_num
    ):
        mal_user_grad, mal_sample_num = self.gen_mal_user_grad(
            mal_grad_mean,
            mal_grad_std,
            mal_sample_num_mean,
            mal_sample_num_std,
            mal_user_num
        )
        mal_news_grad = self.gen_mal_news_grad(
            mal_user_num,
            mal_grad_norm_mean,
            mal_grad_norm_std,
            mal_grad_mean,
            mal_grad_std,
            mal_sample_num_mean,
            mal_sample_num_std,
        )

        for name in mal_news_grad:
            if name in mal_user_grad:
                mal_news_grad[name] = mal_news_grad[name] + mal_user_grad[name]

        return mal_news_grad, mal_sample_num

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
            batch_size=128,
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

        if self.news_pairs is None or step % self.args.adv_update_round == 0:
            self.update_news_pair()

        if len(step_mal_users) != 0:
            # use a small set of mal users to calculate mean and std
            sample_mal_users = random.sample(self.mal_users, min(self.args.user_num, len(self.mal_users)))
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
                mal_grad_norm_mean,
                mal_grad_norm_std,
            ) = self.process_accu_mal_grads(accu_mal_grads, accu_mal_sample_num)

            mal_grad, mal_sample_num = self.gen_mal_grad(
                mal_mean,
                mal_std,
                mal_sample_num_mean,
                mal_sample_num_std,
                mal_grad_norm_mean,
                mal_grad_norm_std,
                len(step_mal_users)
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
