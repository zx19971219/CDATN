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
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
device = torch.device('cpu')

def load_file(file_name):
    df = pd.read_csv(file_name, header=0)
    actual_dict = {}
    for user, sf in df.groupby("u"):
        actual_dict[user] = list(sf["i"])

    data = df[["u", "i"]].to_numpy(dtype=int).tolist()
    return data, actual_dict, set(df["u"]), set(df["i"])

class Dataset:

    def __init__(self, filename):
        self.data, self.train_dict, train_user_set, train_item_set = load_file(filename)
        self.user_set = train_user_set
        self.item_set = train_item_set
        self.n_users = len(train_user_set)
        self.n_items = len(train_item_set)
        self.train_size = len(self.data)
        self.train_neg_dict = self.get_dicts()

        print("Train size:", self.train_size)
        print("Number of user:", self.n_users)
        print("Number of item:", self.n_items)
        print("Data Sparsity: {:.1f}%".format(
            100 * (self.n_users * self.n_items - self.train_size) / (self.n_users * self.n_items)))
        print()

    def get_dicts(self):
        train_actual_dict = self.train_dict
        train_neg_dict = {}
        for user in list(self.user_set):
            train_neg_dict[user] = list(self.item_set - set(train_actual_dict[user]))
        return train_neg_dict

    def neg_sampling(self, num):
        item_dict = self.train_neg_dict
        user_list = []
        item_list = []

        for user in list(self.user_set):
            items = random.sample(item_dict[user], 50)
            item_list += items
            user_list += [user] * len(items)
        result = np.transpose(np.array([user_list, item_list]))
        return random.sample(result.tolist(), num)

    def get_train(self):
        neg = self.neg_sampling(num=self.train_size)
        pos = self.data
        labels = [1] * len(pos) + [0] * len(neg)
        return pos + neg, labels

def training(adata_ui, tdata_ui, adata_label, tdata_label):
    df_adata = pd.DataFrame(adata_ui, columns=["auser", "aitem"])
    df_tdata = pd.DataFrame(tdata_ui, columns=["tuser", "titem"])
    df_adata["alabel"] = adata_label
    df_tdata["tlabel"] = tdata_label
    train_frames = []
    for user, sf_data_1 in df_adata.groupby("auser"):
        df = pd.DataFrame()
        sf_data_2 = df_tdata.loc[df_tdata["tuser"] == user]
        l = min(len(sf_data_1), len(sf_data_2))

        df["user"] = l * [user]

        sample_data_1 = sf_data_1.sample(n=l)

        sample_data_2 = sf_data_2.sample(n=l)

        df["aitem"] = sample_data_1["aitem"].values
        df["titem"] = sample_data_2["titem"].values
        df["alabel"] = sample_data_1["alabel"].values
        df["tlabel"] = sample_data_2["tlabel"].values
        train_frames.append(df)
    frame = pd.concat(train_frames)
    data = frame[["user", "aitem", "titem"]].values.astype(np.int).tolist()
    labels = frame[["alabel", "tlabel"]].values.astype(np.int).tolist()
    return data, labels


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


class CoNet(nn.Module):
    def __init__(self, n_users):
        super(CoNet, self).__init__()
        print("--------------------------auxiliary---------------------------")
        self.adata = Dataset('../data/movie-book/auxiliary.csv')

        print("--------------------------target--------------------------")
        self.tdata = Dataset('../data/movie-book/target_train.csv')
        self.n_users = n_users
        self.reg = 0.0001
        self.batch_size = 32
        self.cross_layer = 2

        self.edim = 32
        self.lr = 0.001

        self.std = 0.01
        self.initialise_neural_net()

        self.U = nn.Embedding(self.n_users, self.edim)
        self.V_adata = nn.Embedding(self.adata.n_items, self.edim)
        self.V_tdata = nn.Embedding(self.tdata.n_items, self.edim)
        self.f = nn.Sigmoid()


    def create_neural_net(self, layers):
        weights = {}
        biases = {}
        for l in range(len(self.layers) - 1):
            weights[l] = torch.normal(mean=0, std=self.std, size=(layers[l], layers[l + 1]), requires_grad=True)
            biases[l] = torch.normal(mean=0, std=self.std, size=(layers[l + 1],), requires_grad=True)
        return weights, biases

    def initialise_neural_net(self):
        edim = 2 * self.edim
        i = 0
        self.layers = [edim]
        while edim > 8:
            i += 1
            edim /= 2
            self.layers.append(int(edim))

        assert (self.cross_layer <= i)

        self.weights_biases_data_1()
        self.weights_biases_data_2()

        self.weights_shared = nn.ParameterList()
        for l in range(self.cross_layer):
            temp = nn.Parameter(torch.empty(self.layers[l], self.layers[l + 1]))
            nn.init.normal(temp)
            self.weights_shared.append(temp)

    def weights_biases_data_1(self):

        weights_data_1, biases_data_1 = self.create_neural_net(self.layers)
        self.weights_data_1 = nn.ParameterList(
            [nn.Parameter(weights_data_1[i]) for i in range(len(self.layers) - 1)])
        self.biases_data_1 = nn.ParameterList(
            [nn.Parameter(biases_data_1[i]) for i in range(len(self.layers) - 1)])
        self.W_data_1 = nn.Parameter(
            torch.normal(mean=0, std=self.std, size=(self.layers[-1], 1), requires_grad=True))
        self.b_data_1 = nn.Parameter(
            torch.normal(mean=0, std=self.std, size=(1,), requires_grad=True))

    def weights_biases_data_2(self):

        weights_data_2, biases_data_2 = self.create_neural_net(self.layers)
        self.weights_data_2 = nn.ParameterList(
            [nn.Parameter(weights_data_2[i]) for i in range(len(self.layers) - 1)])
        self.biases_data_2 = nn.ParameterList(
            [nn.Parameter(biases_data_2[i]) for i in range(len(self.layers) - 1)])
        self.W_data_2 = nn.Parameter(
            torch.normal(mean=0, std=self.std, size=(self.layers[-1], 1), requires_grad=True))
        self.b_data_2 = nn.Parameter(
            torch.normal(mean=0, std=self.std, size=(1,), requires_grad=True))

    def multi_layer_feedforward(self, user, item_data_1, item_data_2):
        user_emb = self.U(user)
        item_emb_data_1 = self.V_adata(item_data_1)
        item_emb_data_2 = self.V_tdata(item_data_2)
        cur_data_1 = torch.cat((user_emb, item_emb_data_1), 1)
        cur_data_2 = torch.cat((user_emb, item_emb_data_2), 1)
        pre_data_1 = cur_data_1
        pre_data_2 = cur_data_2
        for l in range(len(self.layers) - 1):
            cur_data_1 = torch.add(torch.matmul(pre_data_1, self.weights_data_1[l]), self.biases_data_1[l])
            cur_data_2 = torch.add(torch.matmul(pre_data_2, self.weights_data_2[l]), self.biases_data_2[l])
            if (l < self.cross_layer):
                cur_data_1 = torch.add(cur_data_1, torch.matmul(pre_data_2, self.weights_shared[l]))
                cur_data_2 = torch.add(cur_data_2, torch.matmul(pre_data_1, self.weights_shared[l]))
            cur_data_1 = nn.functional.relu(cur_data_1)
            cur_data_2 = nn.functional.relu(cur_data_2)
            pre_data_1 = cur_data_1
            pre_data_2 = cur_data_2
        z_data_1 = torch.matmul(cur_data_1, self.W_data_1) + self.b_data_1
        z_data_2 = torch.matmul(cur_data_2, self.W_data_2) + self.b_data_2
        return self.f(z_data_1), self.f(z_data_2)

    def t_predict(self, user, item):
        user_emb = self.U(user)
        item_emb = self.V_tdata(item)
        cur_data = torch.cat((user_emb, item_emb), 1)
        for l in range(len(self.layers) - 1):
            cur_data = torch.add(torch.matmul(cur_data, self.weights_data_2[l]), self.biases_data_2[l])
            cur_data = nn.functional.relu(cur_data)
        z_data = torch.matmul(cur_data, self.W_data_2) + self.b_data_2
        return torch.squeeze(self.f(z_data))

    def a_predict(self, user, item):
        user_emb = self.U(user)
        item_emb = self.V_adata(item)
        cur_data = torch.cat((user_emb, item_emb), 1)
        for l in range(len(self.layers) - 1):
            cur_data = torch.add(torch.matmul(cur_data, self.weights_data_1[l]), self.biases_data_2[l])
            cur_data = nn.functional.relu(cur_data)
        z_data = torch.matmul(cur_data, self.W_data_1) + self.b_data_1
        return z_data


EPOCHS=100
args = parse_opt()
test_data_a_t = TestDataset('../data/movie-book/test_a_t.txt')
test_data_a_ts = TestDataset('../data/movie-book/test_a_ts.txt')
test_data_as_t = TestDataset('../data/movie-book/test_as_t.txt')
test_data_as_ts = TestDataset('../data/movie-book/test_as_ts.txt')
best_HR, best_NDCG = [[0, 0, 0, 0] for _ in range(4)], [[0, 0, 0, 0] for _ in range(4)]
# for name, parameter in model.named_parameters():
#     print(name,parameter.requires_grad,parameter.is_leaf)
# pretrain t
a_train_data = PointwiseDataset('../data/movie-book/auxiliary.csv', 4)
t_train_data = PointwiseDataset('../data/movie-book/target_train.csv', 4)
n_users = max(a_train_data.n_users,t_train_data.n_users)
a_train_data.n_users, t_train_data.n_users = n_users, n_users
model = CoNet(n_users)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
best_HR, best_NDCG = np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])
n_test = len(test_data_a_t)+len(test_data_a_ts)+len(test_data_as_ts)+len(test_data_as_t)

def pretrain(train_data, mode='t'):
    for k_ep in range(10):
        dataloader = DataLoader(train_data, shuffle=True, batch_size=512)
        testloader_a_t = DataLoader(test_data_a_t, batch_size=100, shuffle=False)
        testloader_a_ts = DataLoader(test_data_a_ts, batch_size=100, shuffle=False)
        testloader_as_t = DataLoader(test_data_as_t, batch_size=100, shuffle=False)
        testloader_as_ts = DataLoader(test_data_as_ts, batch_size=100, shuffle=False)
        testloader = [testloader_a_t,testloader_a_ts,testloader_as_t,testloader_as_ts]
        model.train()
        dataloader.dataset.ng_sample()
        loss, n = 0.0, 0
        for uid, iid, label in tqdm(dataloader,
                                          desc='Epoch No.%i (training)' % k_ep,
                                          leave=False, disable=True):
            uid, iid, label = uid.to(device), iid.to(device), label.to(device)
            if mode=='t':
                predict = model.t_predict(uid, iid)
            else:
                predict = model.a_predict(uid, iid)
            batch_loss = criterion(label.float(), torch.squeeze(predict))
            loss += batch_loss
            n += 1
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        loss /= n
        print('EPOCH %d: loss %.4f.' % (k_ep, loss))

        if k_ep%10==0:
            for testloader_index in range(4):
                model.eval()
                HR, NDCG = [[] for _ in range(4)], [[] for _ in range(4)]
                for uid, iid in testloader[testloader_index]:
                    uid, iid = uid.to(device), iid.to(device)
                    if mode=='t':
                        predict = model.t_predict(uid, iid)
                    else:
                        predict = model.a_predict(uid, iid)
                    _, indices = torch.topk(predict, k=max(args.topKs))
                    recommends = torch.take(
                        iid, indices).cpu().numpy().tolist()
                    gt_item = iid[0].item()
                    for i in range(len(args.topKs)):
                        HR[i].append(hit(gt_item, recommends[:args.topKs[i]]))
                        NDCG[i].append(ndcg(gt_item, recommends[:args.topKs[i]]))

                for i in range(len(args.topKs)):
                    HR[i] = round(np.mean(HR[i]), 3)
                    NDCG[i] = round(np.mean(NDCG[i]), 3)
                print('EPOCH {}: HR: {}, NDCG: {}'.format(k_ep, HR, NDCG))
                if HR[-1]>best_HR[testloader_index][-1]:
                    best_HR[testloader_index] = HR
                    best_NDCG[testloader_index] = NDCG

#pretrain(a_train_data,'a')
pretrain(t_train_data,'t')
adata_ui, adata_label = model.adata.get_train()
tdata_ui, tdata_label = model.tdata.get_train()
data, labels = training(adata_ui, tdata_ui, adata_label, tdata_label)
train_data = torch.tensor(data).to(device)
labels = torch.tensor(labels).to(device)
alabel, tlabel = labels[:, 0], labels[:, 1]
user, aitem, titem = train_data[:, 0], train_data[:, 1], train_data[:, 2]
for epoch in range(EPOCHS):
    losses_A, losses_B = [], []
    permut = torch.randperm(user.shape[0])
    for batch in range(0, user.shape[0], 512):
        optimizer.zero_grad()
        idx = permut[batch: min(batch + 512, user.shape[0])]
        pred_data_1, pred_data_2 = model.multi_layer_feedforward(user[idx], aitem[idx], titem[idx])
        aloss = criterion(alabel[idx].float(), torch.squeeze(pred_data_1))
        tloss = criterion(tlabel[idx].float(), torch.squeeze(pred_data_2))
        losses_A.append(aloss.cpu().item())
        losses_B.append(tloss.cpu().item())
        loss = aloss + tloss
        loss.backward()
        optimizer.step()
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
            tmp_HR += np.array(HR)*len(testloader[testloader_index])
            tmp_NDCG += np.array(NDCG)*len(testloader[testloader_index])
        tmp_HR /= n_test
        tmp_NDCG /= n_test
        print('HR: {}, NDCG: {}'.format(tmp_HR, tmp_NDCG))
        if tmp_HR[-1]>best_HR[-1]:
            best_HR = tmp_HR
            best_NDCG = tmp_NDCG