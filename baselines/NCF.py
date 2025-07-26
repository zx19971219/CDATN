from utils import dataloader,evaluate,dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import torch
from time import time
import numpy as np
import os


class NCF(nn.Module):
    def __init__(self,n_users, n_items, NCF_dim,NCF_layers):
        super(NCF, self).__init__()
        self.n_user = n_users
        self.n_item = n_items
        self.embed_dim = NCF_dim
        self.num_layers = NCF_layers
        self.U_emb = nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.embed_dim)
        self.V_emb = nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.embed_dim)

        self.embed_user_GMF = nn.Embedding(self.n_user, self.embed_dim)
        self.embed_item_GMF = nn.Embedding(self.n_item, self.embed_dim)
        self.embed_user_MLP = nn.Embedding(
            self.n_user, self.embed_dim * (2 ** (self.num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(
            self.n_item, self.embed_dim * (2 ** (self.num_layers - 1)))

        MLP_modules = []
        for i in range(self.num_layers):
            input_size = self.embed_dim * (2 ** (self.num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        predict_size = self.embed_dim * 2
        self.predict_layer = nn.Linear(predict_size, 1)
        self.f = nn.Sigmoid()


    def forward(self, user,item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)
        concat = torch.cat((output_GMF, output_MLP), -1)
        prediction = self.predict_layer(concat)
        return prediction.view(-1)

    def bceloss(self, users, items, labels):
        prediction = self.forward(users, items)
        loss = torch.nn.BCEWithLogitsLoss()(prediction, labels.float())
        return loss

    def bpr_loss(self, users, pos, neg):
        pos_scores = self.forward(users, pos)
        neg_scores = self.forward(users, neg)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        # users_emb = self.U_emb(users)
        # pos_emb = self.V_emb(pos)
        # neg_emb = self.V_emb(neg)
        # reg_loss = self.weight_decay * (1 / 2) * (users_emb.norm(2).pow(2) +
        #                       pos_emb.norm(2).pow(2) +
        #                       neg_emb.norm(2).pow(2)) / float(len(users))
        # loss = loss + reg_loss
        return loss


def parse_opt():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument('--dataset', type=str, default='amazon', help="Musical_Patio")
    parser.add_argument('--model', type=str, default='ATN', help="MF,EMCDR,NCF,MATN,CMF")
    parser.add_argument('--layers', type=int, default=1, help="l")
    parser.add_argument('--batch_size', type=int, default=2, help="b")
    parser.add_argument('--topk', type=int, default=10, help="tk")
    parser.add_argument('--out', type=bool, default=True, help="tk")
    parser.add_argument('--test_num_ng', type=int, default=99, help="b")
    parser.add_argument('--train_num_ng', type=int, default=4, help="b")
    parser.add_argument('--EPOCHs', type=int, default=4, help="b")

    parser.add_argument('--NCF_dim', type=int, default=16, help="b")
    parser.add_argument('--NCF_layers', type=int, default=3, help="b")
    parser.add_argument('--NCF_batch_size', type=int, default=256, help="b")
    parser.add_argument('--NCF_lr', type=float, default=0.001, help="b")
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    args.model_path = '../models/'
    args.read_model_path = '../models/Multi_MLP/'
    args.MF_S_path = args.read_model_path + 'mf_s.pth'
    args.MF_T_path = args.read_model_path + 'mf_t.pth'
    args.Mapping_path = args.read_model_path + 'mapping.pth'
    args.topk = [1,2,5,10]
    return args

args = parse_opt()

print('Choose Model: NCF!')
t_dataset = dataset.PairwiseDataset('target', args.train_num_ng)
t_dataloader = DataLoader(t_dataset, batch_size=args.NCF_batch_size, shuffle=True)
test_data = dataset.TestDataset()
TestLoader = DataLoader(test_data, batch_size=args.test_num_ng+1, shuffle=False)

n_users = max(t_dataset.n_users,max(test_data.users)+1)
n_items = max(t_dataset.n_items,max(test_data.items)+1)
model = NCF(n_users, n_items, args.NCF_dim, args.NCF_layers)
model = model.to(args.device)
opt = torch.optim.Adam(model.parameters(), lr=args.NCF_lr)

for epoch in range(args.EPOCHs):
    start_train_time = time()
    model.train()
    start_time = time()
    #print('ng sample start!')
    t_dataset.ng_sample()
    #print('ng sample over!')
    loss = 0.0
    temp_n = 0
    for batch in t_dataloader:
        model.zero_grad()
        users, pos_items, neg_items = map(lambda x: x, batch)
        batch_loss = model.bpr_loss(users.cuda(), pos_items.cuda(), neg_items.cuda())

        batch_loss.backward()
        opt.step()
        loss += batch_loss
        temp_n += 1
    loss /= temp_n
    if epoch%10==0:
        with torch.no_grad():
            results = evaluate.metrics(model,TestLoader, args.topk)
            print(results)
