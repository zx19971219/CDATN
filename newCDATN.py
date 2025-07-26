# https://pypi.tuna.tsinghua.edu.cn/simple
import torch
import torch.nn as nn
from models import CustomModel
import os
import argparse
from datetime import datetime
from utils.evaluate import hit, ndcg
import coloredlogs
from tqdm import tqdm
import numpy as np
import torch
from utils.dataset import PairwiseDataset, TestDataset
from torch.utils.data import DataLoader
from models import setup_model
import warnings
import random

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class CDATN(CustomModel):
    def __init__(self, dataset, experiment_id=None, experiment_dir=None,
                 CDATN_embed_dim=32, CDATN_GCN_layers=2, cw=4,
                 **kwargs):
        super().__init__(experiment_id, experiment_dir)
        self.n_users = dataset.n_users
        self.n_titems = dataset.n_items
        self.Graph = dataset.getSparseGraph()
        self.embed_dim = CDATN_embed_dim
        self.n_layers = CDATN_GCN_layers
        self.ratio = torch.Tensor(dataset.ratio).to(device)
        self.weight_decay = 0.0001
        self.combine_way = cw
        self.embedding_user = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embed_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.n_titems, embedding_dim=self.embed_dim)
        # self.a_embedding_user = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embed_dim)
        # self.a_embedding_user.requires_grad_(False)
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

    def forward(self, users, items):
        all_users, all_items = self.computer()
        user_emb = all_users[users]
        item_emb = all_items[items]
        score = torch.sum(user_emb * item_emb, dim=1)
        return user_emb, item_emb, score

    def mapping(self, users, items, a_user_emb):
        tu = self.get_embed(users)
        transfer_a_emb, _, _ = self.transfer_block(users, items, a_user_emb)
        loss = mse_loss(transfer_a_emb, tu)
        return loss

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


    def transfer_block(self, users, items, au_embed):
        all_users, all_items = self.computer()
        t_user_emb = all_users[users]
        item_emb = all_items[items]
        a_user_emb = self.transfer_layer(au_embed)
        if self.combine_way == 1:
            user_emb_global = torch.add(self.W_a[users] * a_user_emb, (1 - self.W_a[users]) * t_user_emb)
            user_emb_local = self.attention(self.ratio[users].repeat(self.embed_dim, 1).permute(1, 0)
                                            * t_user_emb, a_user_emb, item_emb)
            user_emb = self.user_agg(torch.cat([user_emb_global, user_emb_local], dim=1))
        elif self.combine_way == 2:
            # 2
            user_emb = torch.add(self.W_a[users] * a_user_emb, (1 - self.W_a[users]) * t_user_emb)
        elif self.combine_way == 3:
            # 3
            user_emb = self.attention(self.ratio[users].repeat(self.embed_dim, 1).permute(1, 0) * t_user_emb,
                                      a_user_emb, item_emb)
        elif self.combine_way == 5:
            user_emb = t_user_emb + a_user_emb
        else:
            # 4 增加一个transfer mapping函数
            user_emb = a_user_emb
        score = torch.sum(user_emb * item_emb, dim=1)
        return user_emb, item_emb, score

    def transfer_bprloss(self, users, pos, neg, au_embed):
        users_emb, pos_emb, pos_scores = self.transfer_block(users, pos, au_embed)
        users_emb, neg_emb, neg_scores = self.transfer_block(users, neg, au_embed)
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss
        return loss

    def get_embed(self, users):
        all_users, all_items = self.computer()
        user_emb = all_users[users]
        return user_emb


def parse_opt():
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--experiment_id', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'), help='')
    parser.add_argument('--experiment_name', type=str, default=None, help='')
    parser.add_argument('--model_name', type=str, default='CDATN', help='options: [GCDTN,MF,EMCDR,NCF,MATN,CMF,CDATN]')

    parser.add_argument('--data_name', type=str, default='movie-book', help="Musical_Patio")
    parser.add_argument('--batch_size', type=int, default=512, help="b")
    parser.add_argument('--EPOCHS', type=int, default=200, help="b")
    parser.add_argument('--init_lr', type=float, default=0.001, help="b")
    parser.add_argument('--test_num_ng', type=int, default=99, help="b")
    parser.add_argument('--train_num_ng', type=int, default=4, help="b")

    # CDATN
    parser.add_argument('--CDATN_embed_dim', type=int, default=32, help="b")
    parser.add_argument('--CDATN_GCN_layers', type=int, default=2, help="b")
    parser.add_argument('--mapping', type=bool, default=False, help="b")
    parser.add_argument('--cw', type=int, default=3, help="b")#combine way
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


def batch_user_item(overlap_users, dataset):
    batch_size = 32
    tmp_data = dataset.data[dataset.data.u.isin(overlap_users)].index
    tmp_index = random.sample(list(tmp_data), min(len(tmp_data),batch_size*len(overlap_users)))
    tmp_test_data = dataset.data.loc[tmp_index]
    for i in range(0,len(tmp_test_data),batch_size):
        yield list(tmp_test_data['u'].iloc[i:min(i+batch_size, len(tmp_test_data))]),\
            list(tmp_test_data['i'].iloc[i:min(i+batch_size, len(tmp_test_data))])



def single_train(filename, savefname):
    train_data = PairwiseDataset(filename, args.train_num_ng)
    dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    model = CDATN(train_data, **vars(args))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.init_lr)

    for epoch in range(100):
        model.train()
        train_data.ng_sample()
        loss = 0.0
        temp_n = 0
        for batch in dataloader:
            model.zero_grad()
            users, pos_items, neg_items = map(lambda x: x.to(device), batch)
            batch_loss = model.bprloss(users, pos_items, neg_items)
            batch_loss.backward()
            opt.step()
            loss += batch_loss
            temp_n += 1
        loss /= temp_n
        print('Epoch %d train==[%.5f]' % (epoch, loss))

    mdir = 'pretrain/movie-book/CDATN/'
    if not os.path.exists(mdir):
        os.makedirs(mdir, exist_ok=True)
    torch.save(model.state_dict(), mdir+savefname)
    return model


EPOCHS=100
args = parse_opt()
train_data = PairwiseDataset('data/movie-book/balance_target_train.csv', args.train_num_ng)
test_data_a_t = TestDataset('data/movie-book/test_a_t.txt')
test_data_a_ts = TestDataset('data/movie-book/test_a_ts.txt')
test_data_as_t = TestDataset('data/movie-book/test_as_t.txt')
test_data_as_ts = TestDataset('data/movie-book/test_as_ts.txt')
best_HR, best_NDCG = np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])
n_test = (len(test_data_a_t)+len(test_data_a_ts)+len(test_data_as_ts)+len(test_data_as_t))/100
# for name, parameter in model.named_parameters():
#     print(name,parameter.requires_grad,parameter.is_leaf)
# pretrain t
train_single = False
print('\n================t Domain================')
filename = 'data/movie-book/target_train.csv'
savefname = 'model_t.pth.tar'
if train_single:
    model_t = single_train(filename,savefname)
else:
    dataset = PairwiseDataset(filename, args.train_num_ng)
    model_t = CDATN(dataset, **vars(args))
    model_t = model_t.to(device)
    model_t.load_state_dict(torch.load('pretrain/movie-book/CDATN/model_t.pth.tar'))


print('\n================a Domain================')
filename = 'data/movie-book/auxiliary.csv'
savefname = 'model_a.pth.tar'
if train_single:
    model_a = single_train(filename, savefname)
else:
    dataset = PairwiseDataset(filename, args.train_num_ng)
    model_a = CDATN(dataset, **vars(args))
    model_a = model_a.to(device)
    model_a.load_state_dict(torch.load('pretrain/movie-book/CDATN/model_a.pth.tar'))

for name, param in model_a.named_parameters():
    param.requires_grad = False


print('\n================train================')
cw_list = ['','W and attention','W','attention','direct','add','else(direct)']
print('combine way:',cw_list[args.cw])

overlap_users = overlap_user('data/movie-book/overlap_users.txt')
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model_t.parameters(), lr=0.001)
batchSize = 512


for epoch in range(EPOCHS):
    print("=" * 20 + "Epoch ", epoch, "=" * 20)
    losses_A, losses_B = [], []
    dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    model_t.train()
    dataloader.dataset.ng_sample()
    if args.mapping:
        loss_sum,n = 0,0
        for users, items in batch_user_item(overlap_users, train_data):
            n += 1
            users = torch.tensor(users).long().to(device)
            au = model_a.get_embed(users)
            mapping_loss = model_t.mapping(users, items, au)
            optimizer.zero_grad()
            mapping_loss.backward()
            optimizer.step()
            loss_sum += mapping_loss.cpu().item()
        loss_sum /= n
        print('mapping loss = %f' % (loss_sum))

    loss, n = 0.0, 0
    for uid, iid_pos, iid_neg in tqdm(dataloader,
                                      desc='Epoch No.%i (training)' % epoch,
                                      leave=False, disable=True):
        uid, iid_pos, iid_neg = uid.to(device), iid_pos.to(device), iid_neg.to(device)
        au = model_a.get_embed(uid)
        batch_loss = model_t.transfer_bprloss(uid, iid_pos, iid_neg, au)
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
        model_t.eval()
        tmp_HR, tmp_NDCG = np.array([0,0,0,0]), np.array([0,0,0,0])
        for testloader_index in range(4):
            HR, NDCG = [[] for _ in range(4)], [[] for _ in range(4)]
            for user, item in testloader[testloader_index]:
                user, item = user.to(device), item.to(device)

                _,_,rating = model_t.transfer_block(user, item, model_a.get_embed(user))
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
