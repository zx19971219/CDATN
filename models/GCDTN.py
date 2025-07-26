import torch
import torch.nn as nn
from models import CustomModel

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class GCDTN(CustomModel):
    def __init__(self, experiment_id=None, experiment_dir=None,
                 GCDTN_embed_dim=32, GCDTN_GCN_layers=2, tmp='None',
                 **kwargs):
        super().__init__(experiment_id, experiment_dir)
        self.n_users = tmp.n_users
        self.n_sitems = tmp.n_sitems
        self.n_titems = tmp.n_titems
        self.s_graph = tmp.s_graph.to(device)
        self.t_graph = tmp.t_graph.to(device)
        self.embed_dim = GCDTN_embed_dim
        self.GCN_layers = GCDTN_GCN_layers
        self.su_emb_table = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embed_dim)
        self.tu_emb_table = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embed_dim)
        self.si_emb_table = nn.Embedding(num_embeddings=self.n_sitems, embedding_dim=self.embed_dim)
        self.ti_emb_table = nn.Embedding(num_embeddings=self.n_titems, embedding_dim=self.embed_dim)
        # self.f = nn.Sigmoid()
        self.transfer_layer = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),nn.ReLU())
        self.mse_func = nn.MSELoss()

    def computer(self):
        s_embed = torch.cat([self.su_emb_table.weight, self.si_emb_table.weight])
        t_embed = torch.cat([self.tu_emb_table.weight, self.ti_emb_table.weight])
        h_s_embed = [s_embed]
        h_t_embed = [t_embed]
        for i in range(self.GCN_layers):
            s_embed = torch.mm(self.s_graph, s_embed)
            h_s_embed.append(s_embed)
            t_embed = torch.mm(self.t_graph, t_embed)
            h_t_embed.append(t_embed)
        h_s_embed = torch.mean(torch.stack(h_s_embed, dim=1), dim=1)
        h_t_embed = torch.mean(torch.stack(h_t_embed, dim=1), dim=1)
        h_su_embed, h_si_embed = torch.split(h_s_embed, [self.n_users, self.n_sitems])
        h_tu_embed, h_ti_embed = torch.split(h_t_embed, [self.n_users, self.n_titems])
        return h_su_embed, h_si_embed,h_tu_embed, h_ti_embed

    def forward(self, su, si, sy, tu, ti, ty):
        self.update_model_counter()
        h_su_embed, h_si_embed, h_tu_embed, h_ti_embed = self.computer()
        # source domain
        source_score = torch.sum(h_su_embed[su] * h_si_embed[si], dim=1)
        # target domain
        target_score = torch.sum(h_tu_embed[tu] * h_ti_embed[ti], dim=1)
        # transfer
        transfer_score = torch.sum((self.transfer_layer(h_su_embed[tu])+h_tu_embed[tu]) * h_ti_embed[ti], dim=1)
        mse_loss = self.mse_func(source_score.to(torch.float32), sy.to(torch.float32)) + \
                   self.mse_func(target_score.to(torch.float32), ty.to(torch.float32)) + \
                   self.mse_func(transfer_score.to(torch.float32), ty.to(torch.float32))
        return mse_loss

    def prediction(self, user, item):
        h_su_embed, h_si_embed, h_tu_embed, h_ti_embed = self.computer()
        transfer_score = torch.sum((self.transfer_layer(h_su_embed[user])+h_tu_embed[user]) * h_ti_embed[item], dim=1)
        return transfer_score

    def getUsersRating(self, users):
        h_su_embed, h_si_embed, h_tu_embed, h_ti_embed = self.computer()
        s_users_emb = h_su_embed[users.long()]
        t_users_emb = h_tu_embed[users.long()]
        transfer_score = torch.matmul(self.transfer_layer(s_users_emb)+t_users_emb, h_ti_embed.t())
        return transfer_score

    # def bceloss(self,user,item,label):
    #     score = self.prediction(user,item)
    #     loss = torch.nn.BCEWithLogitsLoss()(score, label.float())
    #     return loss
    #
    # def bpr_loss(self, users, pos_items, neg_items):
    #     pos_scores = self.forward(users,pos_items)
    #     neg_scores = self.forward(users,neg_items)
    #     loss = torch.mean(nn.functional.softplus(neg_scores-pos_scores))
    #     return loss