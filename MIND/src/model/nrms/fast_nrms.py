from pathlib import Path
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from model.layers import MultiHeadAttention, AdditiveAttention

class TextEncoder(nn.Module):
    def __init__(self, 
                 embedding_matrix,
                 freeze_embedding=False,
                 word_embedding_dim=300,
                 query_vector_dim = 200,
                 dropout_rate=0.2,
                 enable_gpu=True):
        super(TextEncoder, self).__init__()
        self.dropout_rate = 0.2
        pretrained_news_word_embedding = torch.from_numpy(embedding_matrix).float()
        
        self.word_embedding = nn.Embedding.from_pretrained(
            pretrained_news_word_embedding, freeze=False)
        
        self.additive_attention = AdditiveAttention(400, query_vector_dim)
        self.multihead_attention = MultiHeadAttention(300, 20, 20, 20)
        
    def forward(self, text):
        text_vector = F.dropout(self.word_embedding(text.long()),
                                p=self.dropout_rate,
                                training=self.training)
        context_text_vector = self.multihead_attention(text_vector, text_vector, text_vector)
        context_text_vector = F.dropout(context_text_vector,
                                          p=self.dropout_rate,
                                          training=self.training)
        # batch_size, word_embedding_dim
        text_vector = self.additive_attention(context_text_vector)
        return text_vector

class UserEncoder(nn.Module):
    def __init__(self,
                 news_embedding_dim=400,
                 query_vector_dim=200
                ):
        super(UserEncoder, self).__init__()
        self.dropout_rate = 0.2
        self.multihead_attention = MultiHeadAttention(news_embedding_dim,
                                              news_embedding_dim//20, 20, 20)
        self.additive_attention = AdditiveAttention(news_embedding_dim,
                                                    query_vector_dim)
                
    def forward(self, clicked_news_vecs):
        multi_clicked_vectors = self.multihead_attention(
                    clicked_news_vecs, clicked_news_vecs, clicked_news_vecs
                )
        user_vector =  F.dropout(self.additive_attention(multi_clicked_vectors), p=self.dropout_rate, training=self.training)
        return user_vector

class FastNRMS(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        data_path = Path(args.data_path) / args.data
        embedding_matrix = np.load(data_path / 'glove_embedding.npy')

        self.text_encoder = TextEncoder(embedding_matrix, freeze_embedding=args.freeze_embedding)
        self.user_encoder = UserEncoder()
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, data, compute_loss=True):
        batch_size, npratio, word_num = data["batch_candidate_news"].shape
        candidate_news = data["batch_candidate_news"].view(-1, word_num)
        candidate_vector = self.text_encoder(candidate_news).view(batch_size, npratio, -1)
        
        batch_size, clicked_news_num, word_num = data["batch_his"].shape
        clicked_news = data["batch_his"].view(-1, word_num)
        clicked_news_vecs = self.text_encoder(clicked_news).view(batch_size, clicked_news_num, -1)
        
        user_vector = self.user_encoder(clicked_news_vecs)
        
        score = torch.bmm(candidate_vector, user_vector.unsqueeze(-1)).squeeze(dim=-1)
        
        if compute_loss:
            loss = self.criterion(score, data["batch_label"])
            return loss, score
        else:
            return score