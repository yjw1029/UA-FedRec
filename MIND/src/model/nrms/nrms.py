from pathlib import Path
import numpy as np

import torch
from torch import nn

from model.nrms.fast_nrms import UserEncoder, TextEncoder


class NRMS(nn.Module):
    def __init__(self, args):
        super().__init__()

        data_path = Path(args.data_path) / args.data
        embedding_matrix = np.load(data_path / "glove_embedding.npy")

        self.text_encoder = TextEncoder(embedding_matrix, freeze_embedding=args.freeze_embedding)
        self.user_encoder = UserEncoder()

        self.criterion = nn.CrossEntropyLoss()

    def get_news_vec(self, news_index):
        news_vector = self.text_encoder(news_index)
        return news_vector

    def get_user_vec(self, his_news_vector):
        # his_news_vector: batch_size, his_news_num, emb_dim
        user_vector = self.user_encoder(his_news_vector)
        return user_vector

    def get_click_score(self, news_emb, user_emb):
        score = (
            torch.bmm(news_emb.unsqueeze(-2), user_emb.unsqueeze(-1))
            .squeeze(dim=-1)
            .squeeze(dim=-1)
        )
        return score

    def forward(
        self,
        data,
        compute_loss=True,
    ):
        batch_size, npratio, word_num = data["batch_candidate_news"].shape

        candidate_vectors = []
        for i in range(npratio):
            candidate_vector = self.text_encoder(data["batch_candidate_news"][:, i, :])
            candidate_vectors.append(candidate_vector)
        candidate_vector = torch.stack(candidate_vectors, dim=1)

        batch_size, his_num, word_num = data["batch_his"].shape

        his_vectors = []
        for i in range(his_num):
            his_vector = self.text_encoder(data["batch_his"][:, i, :])
            his_vectors.append(his_vector)
        his_vector = torch.stack(his_vectors, dim=1)

        user_vector = self.user_encoder(his_vector)

        score = torch.bmm(candidate_vector, user_vector.unsqueeze(-1)).squeeze(dim=-1)

        if compute_loss:
            loss = self.criterion(score, data["batch_label"])
            return loss, score
        else:
            return score
