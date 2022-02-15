import random
import numpy as np

import torch
from torch.utils.data import Dataset


def newsample(nnn, ratio):
    if ratio > len(nnn):
        return nnn + ["<unk>"] * (ratio - len(nnn))
    else:
        return random.sample(nnn, ratio)


class TrainBaseDataset(Dataset):
    def __init__(
        self,
        args,
        samples,
        users,
        user_indices,
        nid2index,
        news_index,
        *other_args,
        **kwargs
    ):
        self.args = args
        self.nid2index = nid2index
        self.news_index = news_index
        self.samples = []

        for user in users:
            self.samples.extend([samples[i] for i in user_indices[user]])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # pos, neg, his, neg_his
        _, pos, neg, his, _ = self.samples[idx]
        neg = newsample(neg, self.args.npratio)
        candidate_news = [pos] + neg
        candidate_news = self.news_index[[self.nid2index[n] for n in candidate_news]]
        his = [0] * (self.args.max_his_len - len(his)) + [
            self.nid2index[n] for n in his
        ]
        his = self.news_index[his]
        label = 0

        return candidate_news, his, label


def train_base_collate_fn(data):
    batch_candidate_news, batch_his, batch_label = zip(*data)
    batch_candidate_news = np.stack(batch_candidate_news)
    batch_his = np.stack(batch_his)

    batch_candidate_news = torch.LongTensor(batch_candidate_news)
    batch_his = torch.LongTensor(batch_his)
    batch_label = torch.LongTensor(batch_label)

    batch_data = {
        "batch_candidate_news": batch_candidate_news,
        "batch_his": batch_his,
        "batch_label": batch_label,
    }

    return batch_data
