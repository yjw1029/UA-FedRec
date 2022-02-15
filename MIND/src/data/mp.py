import numpy as np

import torch
from torch.utils.data import Dataset

from data.base import newsample


def train_mp_collate_fn(data):
    batch_candidate_news, batch_his, batch_label, batch_uindex, batch_max_unum = zip(
        *data
    )
    max_unum = batch_max_unum[0]

    batch_candidate_news = np.stack(batch_candidate_news)
    batch_his = np.stack(batch_his)

    batch_candidate_news = torch.LongTensor(batch_candidate_news)
    batch_his = torch.LongTensor(batch_his)
    batch_label = torch.LongTensor(batch_label)

    user_mask_matrix = (
        torch.nn.functional.one_hot(
            torch.LongTensor(batch_uindex), num_classes=max_unum
        )
        .t()
        .float()
    )

    data = {
        "batch_candidate_news": batch_candidate_news,
        "batch_his": batch_his,
        "batch_label": batch_label,
        "user_mask_matrix": user_mask_matrix,
    }
    return data


class TrainMPDataset(Dataset):
    def __init__(
        self,
        args,
        samples,
        users,
        user_indices,
        nid2index,
        news_index,
        mal_users,
        *other_args,
        **kwargs
    ):
        self.args = args
        self.nid2index = nid2index
        self.news_index = news_index
        self.samples = []

        for user in users:
            if user not in mal_users:
                self.samples.extend([samples[i] for i in user_indices[user]])

        self.tmp_user_dict = {u: i for i, u in enumerate(users)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # pos, neg, his, neg_his
        _, pos, neg, his, uid = self.samples[idx]

        uindex = self.tmp_user_dict[uid]
        neg = newsample(neg, self.args.npratio)
        candidate_news = [pos] + neg
        candidate_news = self.news_index[[self.nid2index[n] for n in candidate_news]]
        his = [0] * (self.args.max_his_len - len(his)) + [
            self.nid2index[n] for n in his
        ]
        his = self.news_index[his]
        label = 0

        return candidate_news, his, label, uindex, len(self.tmp_user_dict)
