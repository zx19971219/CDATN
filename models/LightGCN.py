import torch.nn as nn
import torch

class LightGCN(nn.Module):
    def __init__(self, dataset, LightGCN_n_layers=1, LightGCN_embed_dim=32, **kargs):
        super(LightGCN, self).__init__()

        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.n_layers = LightGCN_n_layers
        self.weight_decay = 0.001
        #self.keep_prob = keep_prob
        self.embed_dim = LightGCN_embed_dim
        self.embedding_user = nn.Embedding(self.n_users, self.embed_dim)
        self.embedding_item = nn.Embedding(self.n_items, self.embed_dim)
        self.f = nn.Sigmoid()
        self.Graph = dataset.getSparseGraph()

    def computer(self):
        user_embed = self.embedding_user.weight
        item_embed = self.embedding_item.weight
        all_emb = torch.cat([user_embed, item_embed])
        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        return users_emb, pos_emb, neg_emb

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        scores = torch.sum(users_emb * items_emb, dim=1)
        return users_emb,items_emb,scores

    def bprloss(self, users, pos, neg):
        users_emb,pos_emb,pos_scores = self.forward(users, pos)
        users_emb,neg_emb,neg_scores = self.forward(users, neg)
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss
        return loss

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
