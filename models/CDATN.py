import torch
import torch.nn as nn
from models import CustomModel

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class CDATN(CustomModel):
    def __init__(self, dataset, experiment_id=None, experiment_dir=None,
                 CDATN_embed_dim=32, CDATN_GCN_layers=2,
                 **kwargs):
        super().__init__(experiment_id, experiment_dir)
        self.n_users = dataset.n_users
        self.n_titems = dataset.n_items
        self.Graph = dataset.getSparseGraph()
        self.embed_dim = CDATN_embed_dim
        self.n_layers = CDATN_GCN_layers
        self.weight_decay = 0.0001
        self.embedding_user = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embed_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.n_titems, embedding_dim=self.embed_dim)
        self.a_embedding_user = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embed_dim)
        # self.auxiliary_user_embedding.weight.data.copy_(auxiliary_model.embedding_user.weight)
        # self.f = nn.Sigmoid()
        self.transfer_layer = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),nn.ReLU())


    def computer(self):
        user_embed = self.embedding_user.weight
        item_embed = self.embedding_item.weight
        all_emb = torch.cat([user_embed, item_embed])
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_titems])
        return users, items

    def attention(self,tu,au,i_embed):
        u = torch.stack([tu, au], dim=1)
        attn = torch.nn.functional.softmax(torch.sum(u * i_embed.unsqueeze(1), dim=2), dim=1)
        ans = torch.matmul(attn.unsqueeze(1), u).squeeze()
        return ans

    def forward(self, users, items):
        all_users, all_items = self.computer()
        t_user_emb = all_users[users]
        item_emb = all_items[items]
        a_user_embed = self.transfer_layer(self.a_embedding_user(users))
        user_emb = self.attention(t_user_emb,a_user_embed,item_emb)
        score = torch.sum(user_emb * item_emb, dim=1)
        return t_user_emb, item_emb, score

    def getUsersRating(self, users):
        h_su_embed, h_si_embed, h_tu_embed, h_ti_embed = self.computer()
        s_users_emb = h_su_embed[users.long()]
        t_users_emb = h_tu_embed[users.long()]
        transfer_score = torch.matmul(self.transfer_layer(s_users_emb)+t_users_emb, h_ti_embed.t())
        return transfer_score

    def bprloss(self, users, pos, neg):
        users_emb, pos_emb, pos_scores = self.forward(users, pos)
        users_emb, neg_emb, neg_scores = self.forward(users, neg)
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss
        return loss