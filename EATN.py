# https://pypi.tuna.tsinghua.edu.cn/simple
import torch
import torch.nn as nn
from models import CustomModel
import os
import argparse
from datetime import datetime
import logging
from utils.evaluate import hit, ndcg
import coloredlogs
from tqdm import tqdm
import numpy as np
import torch
from utils.dataset import PairwiseDataset, TestDataset
from torch.utils.data import DataLoader
from models import setup_model
import warnings

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class EATN(CustomModel):
    def __init__(self, dataset, experiment_id=None, experiment_dir=None,
                 CDATN_embed_dim=32, CDATN_GCN_layers=2,
                 **kwargs):
        super().__init__(experiment_id, experiment_dir)
        self.n_users = dataset.n_users
        self.n_titems = dataset.n_items
        self.Graph = dataset.getSparseGraph()
        self.embed_dim = CDATN_embed_dim
        self.n_layers = CDATN_GCN_layers
        self.ratio = torch.Tensor(dataset.ratio).to(device)
        self.weight_decay = 0.0001
        self.embedding_user = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embed_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.n_titems, embedding_dim=self.embed_dim)
        self.a_embedding_user = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embed_dim)
        self.a_embedding_user.requires_grad_(False)
        # self.auxiliary_user_embedding.weight.data.copy_(auxiliary_model.embedding_user.weight)
        # self.f = nn.Sigmoid()
        self.transfer_layer = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),nn.ReLU())
        self.user_agg = nn.Sequential(nn.Linear(2*self.embed_dim, self.embed_dim),nn.ReLU())
        self.W_a = nn.Parameter(
            torch.normal(mean=0, std=0.01, size=(self.n_users,self.embed_dim),
                         requires_grad=True))
        self.mse_loss = nn.MSELoss()

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

    def m_forward(self, users, items):
        all_users, all_items = self.computer()
        t_user_emb = all_users[users]
        item_emb = all_items[items]
        a_user_emb = self.transfer_layer(self.a_embedding_user(users))
        # 1
        # user_emb_global = torch.add(self.W_a[users]*a_user_emb, (1-self.W_a[users])*t_user_emb)
        # user_emb_local = self.attention(self.ratio[users].repeat(self.embed_dim, 1).permute(1,0)*t_user_emb,a_user_emb,item_emb)
        # user_emb = self.user_agg(torch.cat([user_emb_global,user_emb_local],dim=1))
        # 2
        # user_emb = torch.add(self.W_a[users] * a_user_emb, (1 - self.W_a[users]) * t_user_emb)
        # 3
        #user_emb = self.attention(self.ratio[users].repeat(self.embed_dim, 1).permute(1,0)*t_user_emb,a_user_emb,item_emb)
        # 4 增加一个transfer mapping函数
        user_emb = a_user_emb
        score = torch.sum(user_emb * item_emb, dim=1)
        return t_user_emb, item_emb, score

    def forward(self, users, items):
        all_users, all_items = self.computer()
        t_user_emb = all_users[users]
        item_emb = all_items[items]
        a_user_emb = self.transfer_layer(self.a_embedding_user(users))
        # 1
        # user_emb_global = torch.add(self.W_a[users]*a_user_emb, (1-self.W_a[users])*t_user_emb)
        # user_emb_local = self.attention(self.ratio[users].repeat(self.embed_dim, 1).permute(1,0)*t_user_emb,a_user_emb,item_emb)
        # user_emb = self.user_agg(torch.cat([user_emb_global,user_emb_local],dim=1))
        # 2
        # user_emb = torch.add(self.W_a[users] * a_user_emb, (1 - self.W_a[users]) * t_user_emb)
        # 3
        user_emb = self.attention(self.ratio[users].repeat(self.embed_dim, 1).permute(1,0)*t_user_emb,a_user_emb,item_emb)
        # 4 增加一个transfer mapping函数
        # user_emb = a_user_emb
        score = torch.sum(user_emb * item_emb, dim=1)
        return t_user_emb, item_emb, score

    def t_forward(self,users,items):
        all_users, all_items = self.computer()
        user_emb = all_users[users]
        item_emb = all_items[items]
        score = torch.sum(user_emb * item_emb, dim=1)
        return user_emb, item_emb, score

    def mapping(self,users):
        au = self.a_embedding_user(users)
        tu = self.embedding_user(users)
        transfer_au = self.transfer_layer(au)
        loss = mse_loss(transfer_au, tu)
        return loss

    def getUsersRating(self, users):
        h_su_embed, h_si_embed, h_tu_embed, h_ti_embed = self.computer()
        s_users_emb = h_su_embed[users.long()]
        t_users_emb = h_tu_embed[users.long()]
        transfer_score = torch.matmul(self.transfer_layer(s_users_emb)+t_users_emb, h_ti_embed.t())
        return transfer_score

    def t_bprloss(self, users, pos, neg):
        users_emb, pos_emb, pos_scores = self.t_forward(users, pos)
        users_emb, neg_emb, neg_scores = self.t_forward(users, neg)
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss
        return loss

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

def parse_opt():
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--experiment_id', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'), help='')
    parser.add_argument('--experiment_name', type=str, default=None, help='')
    parser.add_argument('--model_name', type=str, default='CDATN', help='options: [GCDTN,MF,EMCDR,NCF,MATN,CMF,CDATN]')

    parser.add_argument('--data_name', type=str, default='movie-book', help="Musical_Patio")
    parser.add_argument('--batch_size', type=int, default=512, help="b")
    parser.add_argument('--EPOCHS', type=int, default=50, help="b")
    parser.add_argument('--init_lr', type=float, default=0.001, help="b")
    parser.add_argument('--test_num_ng', type=int, default=99, help="b")
    parser.add_argument('--train_num_ng', type=int, default=4, help="b")

    # CDATN
    parser.add_argument('--CDATN_embed_dim', type=int, default=32, help="b")
    parser.add_argument('--CDATN_GCN_layers', type=int, default=2, help="b")

    args = parser.parse_args()
    args.experiment_dir = os.path.join('log', args.model_name, args.experiment_id)
    args.experiment_name = str(args.experiment_id)
    args.topKs = [1, 2, 5, 10]
    return args

def overlap_user(filename):
    with open(filename, 'r') as f:
        users = [int(uid) for uid in f.read().strip().split()]
    return users

def batch_user(overlap_users, batch_size):
    for i in range(0, len(overlap_users), batch_size):
        yield list(overlap_users[i:min(i+batch_size, len(overlap_users))])

EPOCHS=100
args = parse_opt()

train_data = PairwiseDataset('data/movie-book/target_train.csv', args.train_num_ng)
test_data_a_t = TestDataset('data/movie-book/test_a_t.txt')
test_data_a_ts = TestDataset('data/movie-book/test_a_ts.txt')
test_data_as_t = TestDataset('data/movie-book/test_as_t.txt')
test_data_as_ts = TestDataset('data/movie-book/test_as_ts.txt')
best_HR, best_NDCG = np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])
n_test = (len(test_data_a_t)+len(test_data_a_ts)+len(test_data_as_ts)+len(test_data_as_t))/100
# for name, parameter in model.named_parameters():
#     print(name,parameter.requires_grad,parameter.is_leaf)
# pretrain t

model = EATN(train_data, **vars(args))
model.load_state_dict(torch.load(r'data/movie-book/auxiliary_model.pth'), strict=False)
model.to(device)
overlap_users = overlap_user('data/movie-book/overlap_users.txt')
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batchSize = 512




for epoch in range(EPOCHS):
    print("=" * 20 + "Epoch ", epoch, "=" * 20)
    #logging.info("=" * 20 + "Epoch ", epoch, "=" * 20)
    losses_A, losses_B = [], []
    dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    model.train()
    dataloader.dataset.ng_sample()
    loss, n = 0.0, 0
    for uid, iid_pos, iid_neg in tqdm(dataloader,
                                      desc='Epoch No.%i (training)' % epoch,
                                      leave=False, disable=True):
        uid, iid_pos, iid_neg = uid.to(device), iid_pos.to(device), iid_neg.to(device)
        batch_loss = model.t_bprloss(uid, iid_pos, iid_neg)
        loss += batch_loss
        n += 1
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    loss /= n
    print('recommend loss %.4f.' % (loss))
    if epoch and epoch % 5 == 0:
        testloader_a_t = DataLoader(test_data_a_t, batch_size=100, shuffle=False)
        testloader_a_ts = DataLoader(test_data_a_ts, batch_size=100, shuffle=False)
        testloader_as_t = DataLoader(test_data_as_t, batch_size=100, shuffle=False)
        testloader_as_ts = DataLoader(test_data_as_ts, batch_size=100, shuffle=False)
        testloader = [testloader_a_t, testloader_a_ts, testloader_as_t, testloader_as_ts]
        model.eval()
        tmp_HR, tmp_NDCG = np.array([0,0,0,0]), np.array([0,0,0,0])
        for testloader_index in range(4):
            HR, NDCG = [[] for _ in range(4)], [[] for _ in range(4)]
            for user, item in testloader[testloader_index]:
                user, item = user.to(device), item.to(device)
                _,_,rating = model.t_forward(user, item)
                _, indices = torch.topk(rating, k=max(args.topKs))
                recommends = torch.take(
                    item, indices).cpu().numpy().tolist()
                gt_item = item[0].item()
                for i in range(len(args.topKs)):
                    HR[i].append(hit(gt_item, recommends[:args.topKs[i]]))
                    NDCG[i].append(ndcg(gt_item, recommends[:args.topKs[i]]))

            for i in range(len(args.topKs)):
                HR[i] = round(np.mean(HR[i]), 3)
                NDCG[i] = round(np.mean(NDCG[i]), 3)

            print('HR: {}, NDCG: {}'.format(HR, NDCG))
            tmp_HR = tmp_HR+np.array(HR)*len(testloader[testloader_index])
            tmp_NDCG = tmp_NDCG+np.array(NDCG)*len(testloader[testloader_index])
        tmp_HR = tmp_HR/n_test
        tmp_NDCG = tmp_NDCG/n_test
        for i in range(len(args.topKs)):
            tmp_HR[i] = round(np.mean(tmp_HR[i]), 3)
            tmp_NDCG[i] = round(np.mean(tmp_NDCG[i]), 3)
        print('HR: {}, NDCG: {}'.format(tmp_HR, tmp_NDCG))
        if tmp_HR[-1]>best_HR[-1]:
            best_HR = tmp_HR
            best_NDCG = tmp_NDCG

for epoch in range(EPOCHS):
    print("=" * 20 + "Epoch ", epoch, "=" * 20)
    loss_sum, n = 0, 0
    for users in batch_user(overlap_users, 20):
        n += 1
        users = torch.tensor(users).long().to(device)
        mapping_loss = model.mapping(users)
        optimizer.zero_grad()
        mapping_loss.backward()
        optimizer.step()
        loss_sum += mapping_loss.cpu().item()
    loss_sum /= n
    print('mapping loss = %f' % (loss_sum))
    if epoch and epoch % 5 == 0:
        testloader_a_t = DataLoader(test_data_a_t, batch_size=100, shuffle=False)
        testloader_a_ts = DataLoader(test_data_a_ts, batch_size=100, shuffle=False)
        testloader_as_t = DataLoader(test_data_as_t, batch_size=100, shuffle=False)
        testloader_as_ts = DataLoader(test_data_as_ts, batch_size=100, shuffle=False)
        testloader = [testloader_a_t, testloader_a_ts, testloader_as_t, testloader_as_ts]
        model.eval()
        tmp_HR, tmp_NDCG = np.array([0,0,0,0]), np.array([0,0,0,0])
        for testloader_index in range(4):
            HR, NDCG = [[] for _ in range(4)], [[] for _ in range(4)]
            for user, item in testloader[testloader_index]:
                user, item = user.to(device), item.to(device)
                _,_,rating = model.m_forward(user, item)
                _, indices = torch.topk(rating, k=max(args.topKs))
                recommends = torch.take(
                    item, indices).cpu().numpy().tolist()
                gt_item = item[0].item()
                for i in range(len(args.topKs)):
                    HR[i].append(hit(gt_item, recommends[:args.topKs[i]]))
                    NDCG[i].append(ndcg(gt_item, recommends[:args.topKs[i]]))

            for i in range(len(args.topKs)):
                HR[i] = round(np.mean(HR[i]), 3)
                NDCG[i] = round(np.mean(NDCG[i]), 3)

            print('HR: {}, NDCG: {}'.format(HR, NDCG))
            tmp_HR = tmp_HR+np.array(HR)*len(testloader[testloader_index])
            tmp_NDCG = tmp_NDCG+np.array(NDCG)*len(testloader[testloader_index])
        tmp_HR = tmp_HR/n_test
        tmp_NDCG = tmp_NDCG/n_test
        for i in range(len(args.topKs)):
            tmp_HR[i] = round(np.mean(tmp_HR[i]), 3)
            tmp_NDCG[i] = round(np.mean(tmp_NDCG[i]), 3)
        print('HR: {}, NDCG: {}'.format(tmp_HR, tmp_NDCG))
        if tmp_HR[-1]>best_HR[-1]:
            best_HR = tmp_HR
            best_NDCG = tmp_NDCG

print('HR:{}, NDCG:{}'.format(best_HR, best_NDCG))


