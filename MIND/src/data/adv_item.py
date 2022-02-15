import torch
from torch.utils.data import Dataset


class NewsAdvItemDataset(Dataset):
    def __init__(self, news_index, news_pairs):
        self.news_index = news_index
        self.news_pairs = news_pairs

    def __len__(self):
        return len(self.news_pairs)

    def __getitem__(self, idx):
        candidate_news_idx, positive_news_idx, negative_news_idx = self.news_pairs[idx]
        candidate_news_index = self.news_index[candidate_news_idx]

        positive_news_index = self.news_index[positive_news_idx]
        negative_news_index = self.news_index[negative_news_idx]

        return candidate_news_index, positive_news_index, negative_news_index
