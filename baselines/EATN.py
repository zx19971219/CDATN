import torch
import torch.nn as nn
from utils import dataloader, evaluate, dataset
from torch.utils.data import DataLoader
import argparse
from time import time
import numpy as np
import os


class EATN(nn.Module):
    def __init__(self, preweight, MATN_embed_dim, MATN_layers,
                 n_users, n_sitems, n_titems,
                 mf_s_model = None,mf_t_model=None,mapping_model = None):
        super(EATN, self).__init__()
        self.preweight = preweight
        if self.preweight:
            self.mf_t_model = mf_t_model
            self.mf_s_model = mf_s_model
            self.MLP_model = mapping_model
        self.n_user = n_users
        self.n_sitem = n_sitems
        self.n_titem = n_titems
        self.embed_dim = MATN_embed_dim
        self.layers = MATN_layers - 1
        self.hidden = MATN_embed_dim // 2
        self.__layers__()
        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        if self.preweight:
            self.su_emb.weight.data.copy_(
                self.mf_s_model.U_emb.weight)
            self.si_emb.weight.data.copy_(
                self.mf_s_model.V_emb.weight)
            self.tu_emb.weight.data.copy_(
                self.mf_t_model.U_emb.weight)
            self.ti_emb.weight.data.copy_(
                self.mf_t_model.V_emb.weight)

            for (m1, m2) in zip(
                    self.MLP1_layers, self.MLP_model.MLP1_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)
            for (m1, m2) in zip(
                    self.MLP2_layers, self.MLP_model.MLP2_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)
            for (m1, m2) in zip(
                    self.MLP3_layers, self.MLP_model.MLP3_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

    def __layers__(self):
        self.su_emb = nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.embed_dim)
        self.tu_emb = nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.embed_dim)
        self.si_emb = nn.Embedding(num_embeddings=self.n_sitem, embedding_dim=self.embed_dim)
        self.ti_emb = nn.Embedding(num_embeddings=self.n_titem, embedding_dim=self.embed_dim)
        self.f = nn.Sigmoid()
        self.mse_loss = torch.nn.MSELoss()
        MLP1_modules = [nn.Linear(self.embed_dim, self.hidden),nn.ReLU()]
        for i in range(self.layers):
            MLP1_modules.append(nn.Linear(self.hidden, self.hidden))
            MLP1_modules.append(nn.ReLU())
        MLP1_modules.append(nn.Linear(self.hidden, self.embed_dim))
        self.MLP1_layers = nn.Sequential(*MLP1_modules)

        MLP2_modules = [nn.Linear(self.embed_dim, self.hidden),nn.ReLU()]
        for i in range(self.layers):
            MLP2_modules.append(nn.Linear(self.hidden, self.hidden))
            MLP2_modules.append(nn.ReLU())
        MLP2_modules.append(nn.Linear(self.hidden, self.embed_dim))
        self.MLP2_layers = nn.Sequential(*MLP2_modules)

        MLP3_modules = [nn.Linear(self.embed_dim, self.hidden),nn.ReLU()]
        for i in range(self.layers):
            MLP3_modules.append(nn.Linear(self.hidden, self.hidden))
            MLP3_modules.append(nn.ReLU())
        MLP3_modules.append(nn.Linear(self.hidden, self.embed_dim))
        self.MLP3_layers = nn.Sequential(*MLP3_modules)

        self.embedding_key = torch.nn.Sequential(torch.nn.Linear(self.embed_dim, self.embed_dim),torch.nn.ReLU())
        self.embedding_value = torch.nn.Sequential(torch.nn.Linear(self.embed_dim, self.embed_dim),torch.nn.ReLU())


    def mlp(self,user):
        x1 = self.MLP1_layers(user)
        x2 = self.MLP2_layers(user)
        x3 = self.MLP3_layers(user)
        x = 1/3*(x1+x2+x3)
        return x

    def multi_mlp(self,user):
        x1 = self.MLP1_layers(user)
        x2 = self.MLP2_layers(user)
        x3 = self.MLP3_layers(user)
        return x1,x2,x3

    def mapping(self,user,item):
        su_embed,tu_embed = self.su_emb(user), self.tu_emb(user)
        i_embed = self.ti_emb(item)
        x1,x2,x3 = self.multi_mlp(su_embed)
        fu_embed = self.attention(x1,x2,x3,i_embed)
        loss = self.mse_loss(fu_embed, tu_embed)
        return loss

    def attention(self,x1,x2,x3,i_embed):
        x = torch.stack([x1, x2, x3], dim=1)
        attn = torch.nn.functional.softmax(torch.sum(x * i_embed.unsqueeze(1), dim=2), dim=1)
        ans = torch.matmul(attn.unsqueeze(1), x).squeeze()
        return ans

    def prediction(self,user,item):
        u_embed = self.su_emb(user)
        i_embed = self.ti_emb(item)
        x1,x2,x3 = self.multi_mlp(u_embed)
        fu_embed = self.attention(x1,x2,x3,i_embed)
        score = torch.sum(fu_embed * i_embed, dim=1)
        return score

    '''
    def get_rating(self, user):
        item = self.V_emb.weight.t()
        score = torch.matmul(user, item)
        return self.f(score)
    '''

    def mf_s(self,user,item,label):
        user = self.su_emb(user)
        item = self.si_emb(item)
        score = torch.sum(user * item, dim=1)
        loss = torch.nn.BCEWithLogitsLoss()(score, label.float())
        return loss

    def mf_t(self,user,item,label):
        user = self.tu_emb(user)
        item = self.ti_emb(item)
        score = torch.sum(user * item, dim=1)
        loss = torch.nn.BCEWithLogitsLoss()(score, label.float())
        return loss


    def bceloss(self,user,item,label):
        score = self.prediction(user,item)
        loss = torch.nn.BCEWithLogitsLoss()(score, label.float())
        return loss

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = self.forward(users,pos_items)
        neg_scores = self.forward(users,neg_items)
        loss = torch.mean(nn.functional.softplus(neg_scores-pos_scores))
        return loss

def parse_opt():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument('--dataset', type=str, default='amazon', help="Musical_Patio")
    parser.add_argument('--model', type=str, default='ATN', help="MF,EMCDR,NCF,MATN,CMF")
    parser.add_argument('--batch_size', type=int, default=2, help="b")
    parser.add_argument('--topk', type=int, default=10, help="tk")
    parser.add_argument('--out', type=bool, default=True, help="tk")
    parser.add_argument('--test_num_ng', type=int, default=99, help="b")
    parser.add_argument('--train_num_ng', type=int, default=4, help="b")
    parser.add_argument('--EPOCHs', type=int, default=4, help="b")
    parser.add_argument('--sd_batch_size', type=int, default=256, help="b")
    parser.add_argument('--cd_batch_size', type=int, default=256, help="b")
    parser.add_argument('--preweight', type=bool, default=True, help="b")
    parser.add_argument('--overlapratio', type=float, default=0.5, help="b")


    parser.add_argument('--MATN_loss_weight', type=float, default=0.001, help="b")
    parser.add_argument('--MATN_lr', type=float, default=0.001, help="b")
    parser.add_argument('--MATN_embed_dim', type=int, default=16, help="b")
    parser.add_argument('--MATN_layers', type=int, default=1, help="l")

    parser.add_argument('--BTN_loss_weight', type=float, default=0.001, help="b")
    parser.add_argument('--BTN_lr', type=float, default=0.001, help="b")
    parser.add_argument('--BTN_embed_dim', type=int, default=16, help="b")

    parser.add_argument('--MTN_loss_weight', type=float, default=0.001, help="b")
    parser.add_argument('--MTN_lr', type=float, default=0.001, help="b")
    parser.add_argument('--MTN_embed_dim', type=int, default=16, help="b")

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    args.model_path = '../models/'
    args.read_model_path = '../models/Multi_MLP/'
    args.MF_S_path = args.read_model_path + 'mf_s.pth'
    args.MF_T_path = args.read_model_path + 'mf_t.pth'
    args.Mapping_path = args.read_model_path + 'mapping.pth'
    args.topk = [1,2,5,10]
    return args


print('Choose EATN!')
args = parse_opt()
s_dataset = dataset.PairwiseDataset('source', args.train_num_ng)
s_dataloader = DataLoader(s_dataset, batch_size=args.sd_batch_size, shuffle=True)
t_dataset = dataset.PairwiseDataset('target', args.train_num_ng)
t_dataloader = DataLoader(t_dataset, batch_size=args.sd_batch_size, shuffle=True)
overlap_dataset = dataset.CDPairwiseDataset(t_dataset, args.train_num_ng, args.overlapratio)
overlap_dataloader = DataLoader(overlap_dataset, batch_size=args.sd_batch_size, shuffle=True)
test_data = dataset.TestDataset()
TestLoader = DataLoader(test_data, batch_size=args.test_num_ng+1, shuffle=False)

n_users = t_dataset.n_users
n_titems = t_dataset.n_items
n_sitems = s_dataset.n_items

if args.preweight == True:
    assert os.path.exists(args.MF_S_path), 'lack of MF_S model'
    assert os.path.exists(args.MF_T_path), 'lack of MF_T model'
    assert os.path.exists(args.Mapping_path), 'lack of MF_T model'
    mf_s_model = torch.load(args.MF_S_path)
    mf_t_model = torch.load(args.MF_T_path)
    mapping_model = torch.load(args.Mapping_path)
    model = EATN(args.preweight, args.MATN_embed_dim, args.MATN_layers,
                 n_users, n_sitems, n_titems,mf_s_model, mf_t_model, mapping_model)

else:
    model = EATN(args.preweight, args.MATN_embed_dim, args.MATN_layers,
                 n_users, n_sitems, n_titems)


model = model.to(args.device)
opt = torch.optim.Adam(model.parameters(), lr=args.MATN_lr)

for epoch in range(args.EPOCHs):
    start_train_time = time()
    model.train()
    start_time = time()
    # print('ng sample start!')
    t_dataset.ng_sample()
    s_dataset.ng_sample()
    overlap_dataset.ng_sample()
    Loss, lossS, lossT, lossScore, lossMapping = 0.0, 0.0, 0.0, 0.0, 0.0
    temp_n = 0
    # print('ng sample over!')
    for i,(batchT, batchS, batchOver) in enumerate(zip(t_dataloader, s_dataloader, overlap_dataloader)):
        users, pos_items, neg_items = map(lambda x: x.cuda(), batchT)
        batch_lossT = model.mf_t(users, pos_items, neg_items)
        users, pos_items, neg_items = map(lambda x: x.cuda(), batchS)
        batch_lossS = model.mf_s(users, pos_items, neg_items)
        users, pos_items, neg_items = map(lambda x: x.cuda(), batchOver)
        batch_lossScore = model.bpr_loss(users, pos_items, neg_items)
        #batch_lossMapping = model.mapping(users, pos_items, neg_items)
        loss = (batch_lossT + batch_lossS) + args.MATN_loss_weight * (batch_lossScore)
        opt.zero_grad()
        loss.backward()
        opt.step()
        Loss += loss.cpu().detach()
        temp_n += 1
        lossS += batch_lossS.cpu().detach()
        lossT += batch_lossT.cpu().detach()
        #lossMapping += batch_lossMapping.cpu().detach()
        lossScore += batch_lossScore.cpu().detach()

    lossS /= temp_n
    lossT /= temp_n
    lossScore /= temp_n
    # lossMapping /= temp_n
    Loss /= temp_n

    print('MF_S: Epoch %d train==[%.5f]' % (epoch, lossS))
    print('MF_T: Epoch %d train==[%.5f]' % (epoch, lossT))
    print('MappingScore: Epoch %d train==[%.5f]' % (epoch, lossScore))
    # print('Mapping: Epoch %d train==[%.5f]' % (epoch, lossMapping))
    print('Loss: Epoch %d train==[%.5f]' % (epoch, Loss))


    if epoch % 1 == 0:
        print('================test===============')
        with torch.no_grad():
            HR, NDCG = evaluate.EMCDRETE_metrics(model, TestLoader, args.topks)
            print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
