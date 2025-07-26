import torch
import torch.nn as nn
from models import CustomModel

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class CDATN(CustomModel):
    def __init__(self, experiment_id=None, experiment_dir=None,
                 CDATN_embed_dim=32, CDATN_GCN_layers=2, dataset='None',
                 **kwargs):
        super().__init__(experiment_id, experiment_dir)
        self.n_users = dataset.n_users
        self.n_items = dataset.n_sitems
        self.n_items = dataset.n_titems
        self.s_graph = dataset.s_graph.to(device)
        self.t_graph = dataset.t_graph.to(device)
        self.s_table = dataset.s_table
        self.t_table = dataset.t_table
        self.embed_dim = CDATN_embed_dim
        self.GCN_layers = CDATN_GCN_layers
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

    def forward(self, user):
        self.update_model_counter()
        h_su_embed, h_si_embed, h_tu_embed, h_ti_embed = self.computer()
        # source domain
        sitem = self.s_table[user]
        titem = self.t_table[user]
        if len(sitem)==0:
            source_score = 0
        else:
            source_score = torch.mean(h_su_embed[user] * h_si_embed[sitem], dim=1)

        # transfer
        if len(titem)==0:
            transfer_score = 0
        else:
            transfer_score = torch.mean((self.transfer_layer(h_su_embed[user])+h_tu_embed[user]) * h_ti_embed[titem], dim=1)
        mse_loss = self.mse_func(source_score.to(torch.float32), sy.to(torch.float32)) + \
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
