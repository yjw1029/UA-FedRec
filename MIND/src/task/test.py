import pickle
import logging
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data import get_dataset
from model import get_model
from task.base import BaseTask
from metrics import evaluation_split


class TestTask(BaseTask):
    def load_data(self):
        with open(self.data_path / "test_sam_uid.pkl", "rb") as f:
            self.test_sam = pickle.load(f)

        with open(self.data_path / "glove_test_nid2index.pkl", "rb") as f:
            self.test_nid2index = pickle.load(f)

        self.test_news_index = np.load(
            self.data_path / "glove_test_news_index.npy", allow_pickle=True
        )

    def load_model(self):
        self.model = get_model(self.config["model_name"])(self.args).cuda()
        ckpt = torch.load(
            self.out_model_path / f"{self.args.run_name}-{self.args.data}.pkl"
        )
        self.model.load_state_dict(ckpt["model_state_dict"])

    def start(self):
        self.model.eval()

        test_news_dataset = get_dataset(self.config["news_dataset_name"])(
            self.test_news_index
        )
        news_dl = DataLoader(
            test_news_dataset, batch_size=512, shuffle=False, num_workers=0
        )
        news_vecs = []
        with torch.no_grad():
            for news in tqdm(news_dl):
                news = news.cuda()
                news_vec = self.model.text_encoder(news).detach().cpu().numpy()
                news_vecs.append(news_vec)
        news_vecs = np.concatenate(news_vecs)

        user_dataset = get_dataset(self.config["user_dataset_name"])(
            self.args, self.test_sam, news_vecs, self.test_nid2index
        )
        user_vecs = []
        user_dl = DataLoader(
            user_dataset, batch_size=4096, shuffle=False, num_workers=0
        )

        with torch.no_grad():
            for his in tqdm(user_dl):
                his = his.cuda()
                user_vec = self.model.user_encoder(his).detach().cpu().numpy()
                user_vecs.append(user_vec)
        user_vecs = np.concatenate(user_vecs)

        test_scores = evaluation_split(
            news_vecs, user_vecs, self.test_sam, self.test_nid2index
        )
        test_auc, test_mrr, test_ndcg, test_ndcg10 = [
            np.mean(i) for i in list(zip(*test_scores))
        ]

        logging.info(
            f"test auc: {test_auc:.4f}, mrr: {test_mrr:.4f}, ndcg5: {test_ndcg:.4f}, ndcg10: {test_ndcg10:.4f}"
        )
