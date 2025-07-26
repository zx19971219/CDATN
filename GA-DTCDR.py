import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader
from utils.dataset import PointwiseDataset,PairwiseDataset, TestDataset
import argparse
from utils.evaluate import hit, ndcg
from tqdm import tqdm
import scipy.sparse as sp
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def parse_opt():
    parser = argparse.ArgumentParser()
    # Experiment
    # parser.add_argument('--experiment_id', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'), help='')
    # parser.add_argument('--experiment_name', type=str, default=None, help='')
    args = parser.parse_args()

    # Specific settings
    # args.experiment_dir = os.path.join('log', args.model_name, args.experiment_id)
    # args.experiment_name = str(args.experiment_id)
    args.topKs = [1, 2, 5, 10]
    return args


class GA_DTCDR(nn.Module):
    def __init__(self, n_users, adata, tdata):
        super(GA_DTCDR, self).__init__()
        # print("--------------------------auxiliary---------------------------")
        # self.adata = Dataset('../data/movie-book/auxiliary.csv')
        #
        # print("--------------------------target--------------------------")
        # self.tdata = Dataset('../data/movie-book/target_train.csv')
        self.n_users = n_users
        self.n_aitems = adata.n_items
        self.n_titems = tdata.n_items
        self.embed_dim = 32
        self.a_embedding_user = nn.Embedding(self.n_users, self.embed_dim)
        self.t_embedding_user = nn.Embedding(self.n_users, self.embed_dim)
        self.a_embedding_item = nn.Embedding(self.n_aitems, self.embed_dim)
        self.t_embedding_item = nn.Embedding(self.n_titems, self.embed_dim)
        self.f = nn.Sigmoid()
        self.aGraph = adata.sp_Graph
        self.tGraph = tdata.sp_Graph

        self.MLP_t_users = self.build_MLP()
        self.MLP_t_items = self.build_MLP()
        self.MLP_a_users = self.build_MLP()
        self.MLP_a_items = self.build_MLP()

        self.W_a = nn.Parameter(
            torch.normal(mean=0, std=0.01, size=(self.n_users,self.embed_dim),
                         requires_grad=True))
        self.W_b = nn.Parameter(
            torch.normal(mean=0, std=0.01, size=(self.n_users,self.embed_dim),
                         requires_grad=True))

        self.crossloss = nn.MSELoss()

    def build_MLP(self):
        MLP_layer = []
        input_size = [self.embed_dim, 2 * self.embed_dim, self.embed_dim]
        for i in range(2):
            MLP_layer.append(torch.nn.Linear(input_size[i], input_size[i + 1]))
            MLP_layer.append(torch.nn.ReLU())
        MLP = torch.nn.Sequential(*MLP_layer)
        for i in MLP:
            if isinstance(i, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(i.weight)
        return MLP

    def computer(self,embedding_user,embedding_item,Graph,n_items):
        user_embed = embedding_user.weight
        item_embed = embedding_item.weight
        all_emb = torch.cat([user_embed, item_embed])
        embs = [all_emb]
        g_droped = Graph

        for layer in range(2):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, n_items])
        return users, items

    def multi_layer_feedforward(self, users, aitems, titems):
        all_users, all_items = self.computer(self.a_embedding_user, self.a_embedding_item,
                                             self.aGraph, self.adata.n_items)
        a_users_emb = all_users[users]
        a_items_emb = all_items[aitems]
        all_users, all_items = self.computer(self.t_embedding_user, self.t_embedding_item,
                                             self.tGraph, self.tdata.n_items)
        t_users_emb = all_users[users]
        t_items_emb = all_items[titems]
        pre_aue, pre_tue = a_users_emb, t_users_emb
        a_users_emb = torch.matmul(pre_aue,self.W_a)+torch.matmul(pre_tue,1-self.W_a)
        t_users_emb = torch.matmul(pre_aue,self.W_b)+torch.matmul(pre_tue,1-self.W_b)

        a_users_emb = self.MLP_a_users(a_users_emb)
        a_items_emb = self.MLP_a_items(a_items_emb)
        t_users_emb = self.MLP_t_users(t_users_emb)
        t_items_emb = self.MLP_t_items(t_items_emb)

        z_data_1 = torch.sum(a_users_emb * a_items_emb, dim=1)
        z_data_2 = torch.sum(t_users_emb * t_items_emb, dim=1)
        return self.f(z_data_1), self.f(z_data_2)

    def t_predict(self, users, titems):
        all_users, all_items = self.computer(self.a_embedding_user, self.a_embedding_item,
                                             self.aGraph, self.n_aitems)
        a_users_emb = all_users[users]
        all_users, all_items = self.computer(self.t_embedding_user, self.t_embedding_item,
                                             self.tGraph, self.n_titems)
        t_users_emb = all_users[users]
        t_items_emb = all_items[titems]
        pre_aue, pre_tue = a_users_emb, t_users_emb
        t_users_emb = torch.matmul(pre_aue, self.W_b)+torch.matmul(pre_tue,1-self.W_b)
        # t_users_emb = self.MLP_t_users(t_users_emb)
        # t_items_emb = self.MLP_t_items(t_items_emb)

        scores = torch.sum(t_users_emb * t_items_emb, dim=1)

        return self.f(scores)
    def predict(self, users, items):
        a_user_embed = self.a_embedding_user(users)
        t_user_embed = self.t_embedding_user(users)
        t_item_embed = self.t_embedding_item(items)
        # Element-wise Attention for common users
        final_tu_embed = torch.add(self.W_b[users] * a_user_embed, (1-self.W_b[users])*t_user_embed)
        final_tu_embed = self.MLP_t_users(final_tu_embed)
        final_ti_embed = self.MLP_t_items(t_item_embed)
        # norm_user_output_t = torch.sqrt(torch.sum(torch.square(final_tu_embed), dim=1))
        # norm_item_output_t = torch.sqrt(torch.sum(torch.square(final_ti_embed), dim=1))

        # t_scores = torch.sum(final_tu_embed * final_ti_embed, dim=1) / (norm_user_output_t * norm_item_output_t)
        t_scores = torch.sum(final_tu_embed * final_ti_embed, dim=1)
        tmp_0 = torch.Tensor((t_scores.shape)).fill_(1e-6).to(device)
        t_scores = torch.maximum(tmp_0, t_scores)
        return t_scores

    def forward(self, inputs):
        ausers, aitems, aratings, tusers, titems, tratings = (torch.LongTensor(x).to(device) for x in inputs)
        a_user_embed = self.a_embedding_user(ausers)
        t_user_embed = self.t_embedding_user(tusers)
        a_item_embed = self.a_embedding_item(aitems)
        t_item_embed = self.t_embedding_item(titems)
        # Element-wise Attention for common users
        final_au_embed = torch.add(self.W_a[ausers]*a_user_embed, (1-self.W_a[tusers])*t_user_embed)
        final_tu_embed = torch.add(self.W_b[ausers]*a_user_embed, (1-self.W_b[tusers])*t_user_embed)
        final_au_embed = self.MLP_a_users(final_au_embed)
        final_tu_embed = self.MLP_t_users(final_tu_embed)
        final_ai_embed = self.MLP_a_items(a_item_embed)
        final_ti_embed = self.MLP_t_items(t_item_embed)
        # norm_user_output_a = torch.sqrt(torch.sum(torch.square(final_au_embed), dim=1))
        # norm_item_output_a = torch.sqrt(torch.sum(torch.square(final_ai_embed), dim=1))
        # norm_user_output_t = torch.sqrt(torch.sum(torch.square(final_tu_embed), dim=1))
        # norm_item_output_t = torch.sqrt(torch.sum(torch.square(final_ti_embed), dim=1))
        # regularizer_a = torch.norm(final_au_embed) ** 2 / 2+ torch.norm(final_ai_embed) ** 2 / 2
        # regularizer_t = torch.norm(final_tu_embed) ** 2 / 2+ torch.norm(final_ti_embed) ** 2 / 2
        #
        # a_scores = torch.sum(final_au_embed * final_ai_embed, dim=1) / (norm_user_output_a * norm_item_output_a)
        # t_scores = torch.sum(final_tu_embed * final_ti_embed, dim=1) / (norm_user_output_t * norm_item_output_t)
        #
        a_scores = torch.sum(final_au_embed * final_ai_embed, dim=1)
        t_scores = torch.sum(final_tu_embed * final_ti_embed, dim=1)
        tmp_0 = torch.Tensor((a_scores.shape)).fill_(1e-6).to(device)
        a_scores = torch.maximum(tmp_0, a_scores)
        t_scores = torch.maximum(tmp_0, t_scores)
        loss_a = torch.sum(self.crossloss(a_scores, aratings.float()))
        loss_t = torch.sum(self.crossloss(t_scores, tratings.float()))
        return loss_a, loss_t




EPOCHS=100
args = parse_opt()
test_data_a_t = TestDataset('data/movie-book/test_a_t.txt')
test_data_a_ts = TestDataset('data/movie-book/test_a_ts.txt')
test_data_as_t = TestDataset('data/movie-book/test_as_t.txt')
test_data_as_ts = TestDataset('data/movie-book/test_as_ts.txt')
best_HR, best_NDCG = np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])
n_test = (len(test_data_a_t)+len(test_data_a_ts)+len(test_data_as_ts)+len(test_data_as_t))/100
# for name, parameter in model.named_parameters():
#     print(name,parameter.requires_grad,parameter.is_leaf)
# pretrain t
a_train_data = PointwiseDataset('data/movie-book/auxiliary.csv', 4)
t_train_data = PointwiseDataset('data/movie-book/target_train.csv', 4)
n_users = max(a_train_data.n_users, t_train_data.n_users)
a_train_data.n_users, t_train_data.n_users = n_users, n_users
a_train_data.build_graph()
t_train_data.build_graph()
model = GA_DTCDR(n_users, a_train_data, t_train_data)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batchSize = 512
for epoch in range(EPOCHS):
    print("=" * 20 + "Epoch ", epoch, "=" * 20)
    losses_A, losses_B = [], []
    a_train_data.ng_sample()
    t_train_data.ng_sample()
    a_train_u, a_train_i, a_train_r = np.array(a_train_data.final_data[0]), \
        np.array(a_train_data.final_data[1]), np.array(a_train_data.final_data[2])
    a_train_len = len(a_train_u)
    shuffled_idx_A = np.random.permutation(np.arange(a_train_len))
    a_train_u, a_train_i, a_train_r = a_train_u[shuffled_idx_A], a_train_i[shuffled_idx_A], a_train_r[shuffled_idx_A]

    t_train_u, t_train_i, t_train_r = np.array(t_train_data.final_data[0]), \
        np.array(t_train_data.final_data[1]), np.array(t_train_data.final_data[2])
    t_train_len = len(t_train_u)
    shuffled_idx_t = np.random.permutation(np.arange(t_train_len))
    t_train_u, t_train_i, t_train_r = t_train_u[shuffled_idx_t], t_train_i[shuffled_idx_t], t_train_r[shuffled_idx_t]


    n_a_batch, n_t_batch = a_train_len // batchSize + 1, t_train_len // batchSize + 1
    max_batch = max(n_a_batch, n_t_batch)

    auid,aiid,arating,tuid,tiid,trating = 0,0,0,0,0,0
    for i in range(max_batch):
        min_idx = i * batchSize
        max_idx_A = np.min([a_train_len, (i + 1) * batchSize])
        max_idx_B = np.min([t_train_len, (i + 1) * batchSize])
        if i==0:
            auid,aiid,arating = a_train_u[min_idx:max_idx_A],a_train_i[min_idx:max_idx_A],a_train_r[min_idx:max_idx_A]
            tuid,tiid,trating = t_train_u[min_idx:max_idx_B],t_train_i[min_idx:max_idx_B],t_train_r[min_idx:max_idx_B]

        if min_idx+batchSize < a_train_len:  # the training for domain A has not completed
            auid,aiid,arating = a_train_u[min_idx:max_idx_A],a_train_i[min_idx:max_idx_A],a_train_r[min_idx:max_idx_A]

            optimizer.zero_grad()
            tmp_aloss, _ = model((auid,aiid,arating,tuid,tiid,trating))
            tmp_aloss.backward()
            optimizer.step()
            losses_A.append(tmp_aloss.cpu().item())
        if min_idx+batchSize < t_train_len:  # the training for domain B has not completed
            tuid,tiid,trating = t_train_u[min_idx:max_idx_B],t_train_i[min_idx:max_idx_B],t_train_r[min_idx:max_idx_B]
            optimizer.zero_grad()
            _, tmp_tloss = model((auid,aiid,arating,tuid,tiid,trating))
            tmp_tloss.backward()
            optimizer.step()
            losses_B.append(tmp_tloss.cpu().item())
    loss_A = np.mean(losses_A)
    loss_B = np.mean(losses_B)
    print("epoch {} training aloss : {:.4f}, tloss : {:.4f}".format(epoch, loss_A, loss_B))

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
                rating = model.predict(user, item)
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
            tmp_NDCG[i] = round(np.mean(tmp_NDCG[i]), 3)
            tmp_NDCG[i] = round(np.mean(tmp_NDCG[i]), 3)
        print('HR: {}, NDCG: {}'.format(tmp_HR, tmp_NDCG))
        if tmp_HR[-1]>best_HR[-1]:
            best_HR = tmp_HR
            best_NDCG = tmp_NDCG