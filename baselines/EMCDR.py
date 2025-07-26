from utils import dataloader,evaluate,dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import torch
import numpy as np
import os
import pandas as pd


# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
device = torch.device('cpu')

def parse_opt():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument('--dataset', type=str, default='amazon', help="Musical_Patio")
    parser.add_argument('--model', type=str, default='ATN', help="MF,EMCDR,NCF,MATN,CMF")
    parser.add_argument('--layers', type=int, default=1, help="l")
    parser.add_argument('--batchsize', type=int, default=2, help="b")
    parser.add_argument('--topk', type=int, default=10, help="tk")
    parser.add_argument('--out', type=bool, default=True, help="tk")
    parser.add_argument('--preweight', type=bool, default=True, help="tk")
    parser.add_argument('--n_batch', type=int, default=200, help="b")
    parser.add_argument('--test_num_ng', type=int, default=99, help="b")
    parser.add_argument('--train_num_ng', type=int, default=4, help="b")

    parser.add_argument('--LF_model', type=str, default='MF', help="MF")
    parser.add_argument('--LF_dim', type=int, default=32, help="embedding")
    parser.add_argument('--LF_lr', type=float, default=0.001, help="lr")
    parser.add_argument('--LF_reg', type=float, default=0, help="r")
    parser.add_argument('--LF_epochs', type=int, default=100, help='e')
    parser.add_argument('--LF_batchsize', type=int, default=1024, help="b")

    parser.add_argument('--LS_model', type=str, default='MLP', help="MLP,Multi_MLP")
    parser.add_argument('--LS_dim', type=int, default=32, help="embedding")
    parser.add_argument('--LS_layers', type=int, default=1, help="l")
    parser.add_argument('--LS_lr', type=float, default=0.0005, help="lr")
    parser.add_argument('--LS_reg', type=float, default=0, help="r")
    parser.add_argument('--LS_epochs', type=int, default=1000, help='e')
    parser.add_argument('--LS_batchsize', type=int, default=32, help="b")
    parser.add_argument('--LS_overlapratio', type=float, default=0.5, help="b")
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    args.model_path = '../models/'
    args.read_model_path = '../models/Multi_MLP/'
    args.MF_S_path = args.read_model_path + 'mf_s.pth'
    args.MF_T_path = args.read_model_path + 'mf_t.pth'
    args.Mapping_path = args.read_model_path + 'mapping.pth'
    return args

class MF(nn.Module):
    def __init__(self, dataset, dim):
        super(MF, self).__init__()
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.U_emb = nn.Embedding(num_embeddings=self.n_users, embedding_dim=dim)
        self.V_emb = nn.Embedding(num_embeddings=self.n_items, embedding_dim=dim)
        self.weight_decay = 0.001
        self.f = nn.Sigmoid()

    def get_embed(self, u):
        return self.U_emb(u)

    def get_item_embed(self, i):
        return self.V_emb(i)

    def get_rating(self, user_embed,item):
        item_embed = self.V_emb(item)
        scores = torch.sum(user_embed * item_embed, dim=1)
        #score = torch.matmul(user.unsqueeze(1),item.permute(0,2,1)).squeeze()
        return scores

    def forward(self, u,i):
        user = self.U_emb(u)
        item = self.V_emb(i)
        score = torch.sum(user * item, dim=1)
        return score

    def bceloss(self, users, items, labels):
        prediction = self.forward(users, items)
        loss = torch.nn.BCEWithLogitsLoss()(prediction, labels.float())
        return loss

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

class PairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, filename, num_ng, is_training = None):
        super(PairwiseDataset, self).__init__()
        self.data = pd.read_csv(filename, header=0)
        with open('../data/movie-book/n_users.txt') as file:
            self.n_users = int(file.read().rstrip())
        self.n_items = max(self.data['i'])+1
        self.R = np.zeros((self.n_users, self.n_items))
        self.R[self.data['u'], self.data['i']] = 1
        self.num_ng = num_ng

    def ng_sample(self):
        users = []
        items_ps = []
        items_ng = []
        for (u,i) in zip(self.data['u'], self.data['i']):
            for _ in range(self.num_ng):
                j = np.random.randint(self.n_items)
                while (u, j) in self.R:
                    j = np.random.randint(self.n_items)
                users.append(u)
                items_ps.append(i)
                items_ng.append(j)
        self.final_data = (users, items_ps, items_ng)

    def __len__(self):
        return self.data.shape[0]*(self.num_ng)

    def __getitem__(self, idx):
        user = self.final_data[0][idx]
        item = self.final_data[1][idx]
        item_ng = self.final_data[2][idx]
        return user, item, item_ng

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, filename='/test_a_t.txt'):
        super(TestDataset, self).__init__()
        dataset = 'movie-book'
        self.test_data = self.read_data('../data/'+dataset+filename)

    def read_data(self, filename):
        tmp_data = []
        with open(filename, 'r') as fd:
            line = fd.readline()
            while line != None and line != '':
                arr = line.split(' ')
                if arr[-1] == '\n':
                    arr = arr[:-1]
                u = int(arr[0])
                for i in arr[1:]:
                    tmp_data.append([u, int(i)])
                line = fd.readline()
        return tmp_data

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        return self.test_data[idx][0], self.test_data[idx][1]

class MLP(nn.Module):
    def __init__(self, dim, layers):
        super(MLP, self).__init__()
        self.hidden = dim // 2
        self.layers = layers - 1
        MLP_modules = [nn.Linear(dim, self.hidden),nn.ReLU()]
        for i in range(self.layers):
            MLP_modules.append(nn.Linear(self.hidden, self.hidden))
            MLP_modules.append(nn.ReLU())
        MLP_modules.append(nn.Linear(self.hidden, dim))
        self.MLP_layers = nn.Sequential(*MLP_modules)

    def forward(self, u):
        x = self.MLP_layers(u)
        return x

def single_train(filename, savefname):
    dataset = PairwiseDataset(filename, args.train_num_ng)
    dataloader = DataLoader(dataset, batch_size=args.LF_batchsize, shuffle=True)
    mf = MF(dataset, args.LF_dim)
    mf = mf.to(device)
    opt = torch.optim.Adam(mf.parameters(), lr=args.LF_lr)

    for epoch in range(args.LF_epochs):
        mf.train()
        dataset.ng_sample()
        loss = 0.0
        temp_n = 0
        for batch in dataloader:
            mf.zero_grad()
            users, pos_items, neg_items = map(lambda x: x.to(device), batch)
            batch_loss = mf.bpr_loss(users, pos_items, neg_items)
            batch_loss.backward()
            opt.step()
            loss += batch_loss
            temp_n += 1
        loss /= temp_n
        print('Epoch %d train==[%.5f]' % (epoch, loss))

    mdir = 'pretrain/movie-book/EMCDR/'
    if not os.path.exists(mdir):
        os.makedirs(mdir, exist_ok=True)
    torch.save(mf.state_dict(), mdir+savefname)
    return mf


args = parse_opt()
train_single = True
print('\n================t Domain MF================')
filename = '../data/movie-book/target_train.csv'
savefname = 'MF_t.pth.tar'
if train_single:
    mf_t = single_train(filename,savefname)
else:
    dataset = PairwiseDataset(filename, args.train_num_ng)
    mf_t = MF(dataset, args.LF_dim)
    mf_t = mf_t.to(device)
    mf_t.load_state_dict(torch.load('pretrain/movie-book/EMCDR/MF_t.pth.tar'))


print('\n================s Domain MF================')
filename = '../data/movie-book/auxiliary.csv'
savefname = 'MF_s.pth.tar'
if train_single:
    mf_s = single_train(filename, savefname)
else:
    dataset = PairwiseDataset(filename, args.train_num_ng)
    mf_s = MF(dataset, args.LF_dim)
    mf_s = mf_s.to(device)
    mf_s.load_state_dict(torch.load('pretrain/movie-book/EMCDR/MF_s.pth.tar'))

print('\n================mapping================')
def overlap_user(filename):
    with open(filename, 'r') as f:
        users = [int(uid) for uid in f.read().strip().split()]
    return users

def batch_user(overlap_users, batch_size):
    for i in range(0, len(overlap_users), batch_size):
        yield list(overlap_users[i:min(i+batch_size, len(overlap_users))])

mapping = MLP(args.LS_dim, args.LS_layers)
mapping = mapping.to(device)
opt = torch.optim.Adam(mapping.parameters(), lr=args.LS_lr)

overlap_users = overlap_user('../data/movie-book/overlap_users.txt')
print(len(overlap_users))
mse_loss = nn.MSELoss()

for epoch in range(args.LS_epochs):
    loss_sum = 0
    for users in batch_user(overlap_users, 20):
        us = torch.tensor(users).long()
        us = us.to(device)
        u = mf_s.get_embed(us)
        y = mf_t.get_embed(us)
        out = mapping(u)
        loss = mse_loss(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.cpu().item()
    print('Epoch %d loss = %f' % (epoch, loss_sum))


# for epoch in range(args.LS_epochs):
#     loss_sum = 0
#     temp_n = 0
#     for batch in t_dataloader:
#         users, pos_items, neg_items = map(lambda x: x.cuda(), batch)
#         users = users.cuda()
#         user_embed = mapping(mf_s.get_embed(users))
#         pos_scores = mf_t.get_rating(user_embed, pos_items)
#         neg_scores = mf_t.get_rating(user_embed, neg_items)
#         # y = mf_t.get_embed(users)
#         batch_loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
#         opt.zero_grad()
#         batch_loss.backward()
#         opt.step()
#         loss_sum += batch_loss
#         temp_n += 1
#     loss_sum /= temp_n
#     print('Epoch %d train==[%.5f]' % (epoch, loss_sum))

test_data_a_t = TestDataset('/test_a_t.txt')
test_data_a_ts = TestDataset('/test_a_ts.txt')
test_data_as_t = TestDataset('/test_as_t.txt')
test_data_as_ts = TestDataset('/test_as_ts.txt')
testloader_a_t = DataLoader(test_data_a_t, batch_size=100, shuffle=False)
testloader_a_ts = DataLoader(test_data_a_ts, batch_size=100, shuffle=False)
testloader_as_t = DataLoader(test_data_as_t, batch_size=100, shuffle=False)
testloader_as_ts = DataLoader(test_data_as_ts, batch_size=100, shuffle=False)
testloader = [testloader_a_t, testloader_a_ts, testloader_as_t, testloader_as_ts]
for testloader_index in range(4):
    HR, NDCG = evaluate.EMCDR_metrics(mf_s, mf_t, mapping, testloader[testloader_index], args.topk)
    print('HR: {}, NDCG: {}'.format(HR, NDCG))

