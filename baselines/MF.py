from utils import dataloader,evaluate,dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import torch
from time import time
import numpy as np
import os

class MF(nn.Module):
    def __init__(self, n_user, n_item, dim):
        super(MF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.U_emb = nn.Embedding(num_embeddings=n_user, embedding_dim=dim)
        self.V_emb = nn.Embedding(num_embeddings=n_item, embedding_dim=dim)
        self.weight_decay = 0.001
        self.f = nn.Sigmoid()

    def get_embed(self, u):
        return self.U_emb(u)

    def get_item_embed(self, i):
        return self.V_emb(i)

    def get_rating(self, user,item):
        item = self.V_emb(item)
        score = torch.matmul(user.unsqueeze(1),item.permute(0,2,1)).squeeze()
        return score

    def forward(self, u,i):
        user = self.U_emb(u)
        item = self.V_emb(i)
        score = torch.sum(user * item, dim=1)
        return score

    def predict(self, users, items):
        user_emb = self.U_emb(users)
        item_emb = self.V_emb(items)
        score = torch.matmul(user_emb.unsqueeze(1),item_emb.permute(0,2,1)).squeeze()
        return score

    def bpr_loss(self, users, pos, neg):
        users_emb = self.U_emb(users)
        pos_emb = self.V_emb(pos)
        neg_emb = self.V_emb(neg)
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
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
    # parser.add_argument('--batchsize', type=int, default=2, help="b")
    parser.add_argument('--topk', type=int, default=10, help="tk")
    parser.add_argument('--out', type=bool, default=True, help="tk")
    parser.add_argument('--test_num_ng', type=int, default=99, help="b")
    parser.add_argument('--train_num_ng', type=int, default=4, help="b")
    parser.add_argument('--EPOCHs', type=int, default=30, help="b")

    parser.add_argument('--PureMF_batch_size', type=int, default=256, help="b")
    parser.add_argument('--PureMF_epochs', type=int, default=1, help="b")
    parser.add_argument('--PureMF_dim', type=int, default=32, help="b")
    parser.add_argument('--PureMF_lr', type=float, default=0.001, help="b")
    args = parser.parse_args()
    #args.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    args.device = 'cpu'
    args.model_path = '../models/'
    args.read_model_path = '../models/Multi_MLP/'
    args.MF_S_path = args.read_model_path + 'mf_s.pth'
    args.MF_T_path = args.read_model_path + 'mf_t.pth'
    args.Mapping_path = args.read_model_path + 'mapping.pth'
    args.topk = [1,2,5,10]
    return args

args = parse_opt()

print('Choose Model: PureMF!')
t_dataset = dataset.PairwiseDataset('target', args.train_num_ng)
t_dataloader = DataLoader(t_dataset, batch_size=args.PureMF_batch_size, shuffle=True)
test_data = dataset.TestDataset()
TestLoader = DataLoader(test_data, batch_size=args.test_num_ng+1, shuffle=False)

n_users = max(t_dataset.n_users, test_data.max_uid+1)
n_items = max(t_dataset.n_items, test_data.max_iid+1)
model = MF(n_users, n_items, args.PureMF_dim)
model = model.to(args.device)
opt = torch.optim.Adam(model.parameters(), lr=args.PureMF_lr)

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
        users, pos_items, neg_items = map(lambda x: x.to(args.device), batch)
        # batch_loss = mf.bpr_loss(users.cuda(), pos_items.cuda(), neg_items.cuda())
        batch_loss = model.bpr_loss(users, pos_items, neg_items)

        batch_loss.backward()
        opt.step()
        loss += batch_loss
        temp_n += 1
    loss /= temp_n
    print('Epoch %d train==[%.5f]' % (epoch, loss))

    if epoch%10==0:
        with torch.no_grad():
            results = evaluate.metrics(model,TestLoader, args.topk)
            print(results)
            #elapsed_time = time() - start_time
            #print("The time elapse of epoch {:03d}".format(epoch) + " is: " + '{}'.format(elapsed_time))
            #print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
