from torch import nn
import torch

class SimpleRegressorModelV1(nn.Module):
    '''
    Simple model for movie recommender system. Based on Embedding layers for users \
    and items and a single fully-connected layer. Predicts a single number, which
    is the predicted rating for a given movie by a given user.
    '''
    def __init__(self, user_count, item_count, user_embed_size=32, item_embed_size=32):
        super().__init__()
        self.user_emb = nn.Embedding(user_count, user_embed_size)
        self.item_emb = nn.Embedding(item_count, item_embed_size)
        self.regressor = nn.Linear(user_embed_size + item_embed_size, 1)

    def forward(self, users, items):
        u_emb = self.user_emb(users)
        i_emb = self.item_emb(items)
        return self.regressor(torch.cat([u_emb, i_emb], dim=1))
