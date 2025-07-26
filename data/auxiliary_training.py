import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import argparse
import logging
from tqdm import tqdm
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader

class PairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, num_ng, is_training = None):
        super(PairwiseDataset,self).__init__()
        dataset = 'movie-book'
        self.data = pd.read_csv('../data/'+dataset+'/auxiliary.csv',header=0)
        with open('../data/'+dataset+'/n_users.txt','r') as f:
            self.n_users = int(f.read().rstrip())
        self.n_items = max(self.data['i'])+1
        self.R = np.zeros((self.n_users, self.n_items))
        self.R[self.data['u'], self.data['i']] = 1
        # self.is_training = is_training
        self.num_ng = num_ng

    def ng_sample(self):
        # assert self.is_training, 'no need to sampling when testing'
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

    def getSparseGraph(self):
        print("loading adjacency matrix")
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = sp.csr_matrix(self.R)
        R = R.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.Graph = self.Graph.coalesce().to(device)
        print("don't split the matrix")
        return self.Graph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

class TestDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(TestDataset,self).__init__()
        dataset = 'movie-book'
        self.test_data = []
        with open('../data/'+dataset+'/auxiliary_test.txt', 'r') as fd:
            line = fd.readline()
            while line != None and line != '':
                arr = line.split(' ')
                u = eval(arr[0])
                self.test_data.append([u, eval(arr[1])])
                for i in arr[2:]:
                    self.test_data.append([u, int(i)])
                line = fd.readline()
        tmp=1

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        user = self.test_data[idx][0]
        item = self.test_data[idx][1]
        return user, item

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


def parse_opt():
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--disable_tqdm', action='store_true', default=False, help='')

    # Data
    # parser.add_argument('--dataset', type=str, default='amazon', help="Musical_Patio")
    parser.add_argument('--batch_size', type=int, default=256, help="b")
    parser.add_argument('--EPOCHS', type=int, default=200, help="b")
    parser.add_argument('--init_lr', type=float, default=0.001, help="b")
    parser.add_argument('--num_ng', type=int, default=4, help="b")

    parser.add_argument('--LightGCN_n_layers', type=int, default=2, help="b")
    parser.add_argument('--LightGCN_embed_dim', type=int, default=32, help="b")
    args = parser.parse_args()
    args.topKs = 10
    return args

def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0

def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0

def metrics(model, test_loader, top_k):
    HR, NDCG = [], []
    for user, item in test_loader:
        user = user.cuda()
        item = item.cuda()
        _, _, predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(
                item, indices).cpu().numpy().tolist()

        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

args = parse_opt()

dataset = PairwiseDataset(args.num_ng)
test_data = TestDataset()
model = LightGCN(dataset, **vars(args))
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
best_hr, best_ndcg, best_epoch = 0,0,0
for epoch in range(args.EPOCHS):
    # train
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
    TestLoader = DataLoader(test_data, batch_size=100, shuffle=False)
    if epoch%10==0:
       model.eval()
       HR, NDCG = metrics(model, TestLoader, args.topKs)

       print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

       if HR > best_hr:
           best_hr, best_ndcg, best_epoch = HR, NDCG, epoch


    model.train()
    torch.set_grad_enabled(model.training)
    dataloader.dataset.ng_sample()
    loss_train = 0
    n = 0
    for uid, iid_pos, iid_neg in tqdm(dataloader,
                                      desc='Epoch No.%i (training)' % epoch,
                                      leave=False, disable=args.disable_tqdm):
        uid, iid_pos, iid_neg = uid.to(device), iid_pos.to(device), iid_neg.to(device)
        batch_loss = model.bprloss(uid, iid_pos, iid_neg)
        loss_train += batch_loss
        n += 1
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    loss_train /= n
    print('Epoch No.%i train results: loss (%.4f)' % (epoch, loss_train))
