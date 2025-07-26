import torch.nn as nn
import torch


class PureMF(nn.Module):
    def __init__(self,config,loader):
        super(PureMF, self).__init__()
        self.n_user = loader.n_users
        self.n_item = loader.n_items
        self.embed_dim = config.PureMF_dim
        self.U_emb = nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.embed_dim)
        self.V_emb = nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.embed_dim)
        self.f = nn.Sigmoid()

    def get_embed(self, u):
        return self.U_emb(u)

    def get_rating(self, user):
        item = self.V_emb.weight.t()
        user = self.U_emb(user)
        score = torch.matmul(user, item)
        score = self.f(score)
        return score[:,self.n_sitems:]

    def forward(self, u,i):
        user = self.U_emb(u)
        item = self.V_emb(i)
        score = torch.sum(user * item, dim=1)
        return score

    def bceloss(self, users, items, labels):
        prediction = self.forward(users, items)
        loss = torch.nn.BCEWithLogitsLoss()(prediction, labels.float())
        return loss

