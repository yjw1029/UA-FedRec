import numpy as np
import pickle
import wandb
import random
from tqdm import tqdm
import logging

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from task.base import BaseTask
from model import get_model
from agg import get_agg
from data import get_collate_fn, get_dataset
from metrics import evaluation_split


def process_grad(model_param, sample_num, user_sample):
    model_grad = {}
    for name, param in model_param:
        if param.requires_grad:
            model_grad[name] = param.grad * (sample_num / user_sample)
    return model_grad


class BaseTrainTask(BaseTask):
    """Train baseline using large batch size SGD. Cannot compute per-user gradient"""

    def __init__(self, args, config, device):
        super().__init__(args, config, device)

        if "debug" not in args.run_name:
            wandb.init(
                project=f"{args.project_name}-{args.data}",
                config=args,
                name=f"{args.run_name}-{args.data}",
            )
            logging.info("[-] finishing initing wandb.")

    def load_data(self):
        with open(self.data_path / "glove_nid2index.pkl", "rb") as f:
            self.nid2index = pickle.load(f)

        self.news_index = np.load(
            self.data_path / "glove_news_index.npy", allow_pickle=True
        )

        with open(self.data_path / "train_sam_uid.pkl", "rb") as f:
            self.train_sam = pickle.load(f)

        with open(self.data_path / "valid_sam_uid.pkl", "rb") as f:
            self.valid_sam = pickle.load(f)

        with open(self.data_path / "user_indices.pkl", "rb") as f:
            self.user_indices = pickle.load(f)

        with open(self.data_path / "news_ctr.pkl", "rb") as f:
            self.news_ctr = pickle.load(f)

        with open(self.data_path / "news_pop_class.pkl", "rb") as f:
            self.news_pop_class = pickle.load(f)

    def load_model(self):
        model_cls = get_model(self.config["model_name"])
        agg_cls = get_agg(self.config["agg_name"])
        self.agg = agg_cls(
            self.args,
            model_cls=model_cls,
            device=self.device,
        )
        self.model = model_cls(self.args).to(self.device)

    def train_on_step(self, step):
        users = random.sample(self.user_indices.keys(), self.args.user_num)
        train_ds = get_dataset(self.config["train_dataset_name"])(
            self.args,
            self.train_sam,
            users,
            self.user_indices,
            self.nid2index,
            self.news_index,
        )
        train_collate_fn = get_collate_fn(self.config["train_collate_fn_name"])
        train_dl = DataLoader(
            train_ds, collate_fn=train_collate_fn, batch_size=self.args.batch_size, shuffle=True, num_workers=0
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

    def validate(self, step):
        self.agg.model.eval()
        news_dataset = get_dataset(self.config["news_dataset_name"])(self.news_index)
        news_dl = DataLoader(news_dataset, batch_size=512, shuffle=False, num_workers=0)
        news_vecs = []
        for news in tqdm(news_dl):
            news = news.to(self.device)
            news_vec = self.agg.model.text_encoder(news).detach().cpu().numpy()
            news_vecs.append(news_vec)
        news_vecs = np.concatenate(news_vecs)

        user_dataset = get_dataset(self.config["user_dataset_name"])(
            self.args, self.valid_sam, news_vecs, self.nid2index
        )
        user_vecs = []
        user_dl = DataLoader(
            user_dataset, batch_size=4096, shuffle=False, num_workers=0
        )
        with torch.no_grad():
            for his in tqdm(user_dl):
                his = his.to(self.device)
                user_vec = self.agg.model.user_encoder(his).detach().cpu().numpy()
                user_vecs.append(user_vec)
        user_vecs = np.concatenate(user_vecs)

        val_scores = evaluation_split(
            news_vecs, user_vecs, self.valid_sam, self.nid2index
        )
        val_auc, val_mrr, val_ndcg, val_ndcg10 = [
            np.mean(i) for i in list(zip(*val_scores))
        ]

        if "debug" not in self.args.run_name:
            wandb.log(
                {
                    "valid auc": val_auc,
                    "valid mrr": val_mrr,
                    "valid ndcg@5": val_ndcg,
                    "valid ndcg@10": val_ndcg10,
                },
                step=step + 1,
            )

        logging.info(
            f"[{step}] round auc: {val_auc:.4f}, mrr: {val_mrr:.4f}, ndcg5: {val_ndcg:.4f}, ndcg10: {val_ndcg10:.4f}"
        )

        if val_auc > self.best_auc:
            self.best_auc = val_auc

            if "debug" not in self.args.run_name:
                wandb.run.summary["best_auc"] = self.best_auc
                wandb.run.summary["best_mrr"] = val_mrr
                wandb.run.summary["best_ndcg@5"] = val_ndcg
                wandb.run.summary["best_ndcg@10"] = val_ndcg10

            torch.save(
                {"model_state_dict": self.agg.model.state_dict()},
                self.out_model_path / f"{self.args.run_name}-{self.args.data}.pkl",
            )
            logging.info(f"[{step}] round save model")

    def start(self):
        self.best_auc = 0
        for step in range(self.args.max_train_steps):
            self.train_on_step(step)

            if (step + 1) % self.args.validation_steps == 0:
                self.validate(step)
