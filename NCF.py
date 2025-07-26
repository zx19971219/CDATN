# https://pypi.tuna.tsinghua.edu.cn/simple
import os
import argparse
from datetime import datetime
import logging
from utils.evaluate import hit, ndcg
import coloredlogs
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from utils.dataset import PairwiseDataset, TestDataset
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class NCF(nn.Module):
    def __init__(self, dataset, NCF_dim=32, NCF_layers=2, ** kargs):
        super(NCF, self).__init__()
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.embed_dim = NCF_dim
        self.num_layers = NCF_layers
        self.U_emb = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embed_dim)
        self.V_emb = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.embed_dim)

        self.embed_user_GMF = nn.Embedding(self.n_users, self.embed_dim)
        self.embed_item_GMF = nn.Embedding(self.n_items, self.embed_dim)
        self.embed_user_MLP = nn.Embedding(
            self.n_users, self.embed_dim * (2 ** (self.num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(
            self.n_items, self.embed_dim * (2 ** (self.num_layers - 1)))

        MLP_modules = []
        for i in range(self.num_layers):
            input_size = self.embed_dim * (2 ** (self.num_layers - i))
            # MLP_modules.append(nn.Dropout(p=self.dropout))
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

    def bprloss(self, users, pos, neg):
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
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--experiment_id', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'), help='')
    parser.add_argument('--experiment_name', type=str, default=None, help='')
    parser.add_argument('--model_name', type=str, default='NCF')

    # Data
    parser.add_argument('--data_name', type=str, default='movie-book', help="Musical_Patio")
    parser.add_argument('--batch_size', type=int, default=512, help="b")
    parser.add_argument('--EPOCHS', type=int, default=400, help="b")
    parser.add_argument('--init_lr', type=float, default=0.001, help="b")
    parser.add_argument('--test_num_ng', type=int, default=99, help="b")
    parser.add_argument('--train_num_ng', type=int, default=4, help="b")
    parser.add_argument('--layers', type=int, default=1, help="l")

    parser.add_argument('--NCF_dim', type=int, default=32, help="b")
    parser.add_argument('--NCF_layers', type=int, default=2, help="b")
    parser.add_argument('--NCF_lr', type=float, default=0.001, help="b")

    args = parser.parse_args()
    args.model_path = 'models/'

    # Specific settings
    args.experiment_dir = os.path.join('baselines/log', args.model_name, args.experiment_id)
    args.experiment_name = str(args.experiment_id)
    args.topKs = [1, 2, 5, 10]
    return args


def predict(model, test_dataloader):
    model.eval()
    HR, NDCG = [[] for _ in range(4)], [[] for _ in range(4)]
    for user, item in test_dataloader:
        user,item = user.to(device), item.to(device)
        _, _, rating = model(user, item)
        _, indices = torch.topk(rating, k=max(args.topKs))
        recommends = torch.take(
            item, indices).cpu().numpy().tolist()
        gt_item = item[0].item()
        for i in range(len(args.topKs)):
            HR[i].append(hit(gt_item, recommends[:args.topKs[i]]))
            NDCG[i].append(ndcg(gt_item, recommends[:args.topKs[i]]))

    for i in range(len(args.topKs)):
        HR[i] = round(np.mean(HR[i]),3)
        NDCG[i] = round(np.mean(NDCG[i]),3)
    return HR, NDCG

args = parse_opt()
if not os.path.isdir(args.experiment_dir):
    os.makedirs(args.experiment_dir)
logging.basicConfig(filename='%s.log' % args.experiment_dir, format='[%(levelname)s] %(asctime)s %(message)s ')
coloredlogs.install(level='INFO', fmt='[%(levelname)s] %(message)s %(asctime)s')
logging.info('Experiment ID.%s start ......' % args.experiment_id)
logging.info(args)

train_data = PairwiseDataset('data/movie-book/target_train.csv', args.train_num_ng)
test_data_a_t = TestDataset('data/movie-book/test_a_t.txt')
test_data_a_ts = TestDataset('data/movie-book/test_a_ts.txt')
test_data_as_t = TestDataset('data/movie-book/test_as_t.txt')
test_data_as_ts = TestDataset('data/movie-book/test_as_ts.txt')
best_HR, best_NDCG = np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])
n_test = (len(test_data_a_t)+len(test_data_a_ts)+len(test_data_as_ts)+len(test_data_as_t))/100

model = NCF(train_data, **vars(args))
model.to(device)
# print("Total number of param in {} is {}".format(args.model_name,sum(x.numel() for x in model.parameters())))
optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, threshold=0.01,factor=0.1, min_lr=1e-6)
for epoch in range(args.EPOCHS):
    dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    testloader_a_t = DataLoader(test_data_a_t, batch_size=100, shuffle=False)
    testloader_a_ts = DataLoader(test_data_a_ts, batch_size=100, shuffle=False)
    testloader_as_t = DataLoader(test_data_as_t, batch_size=100, shuffle=False)
    testloader_as_ts = DataLoader(test_data_as_ts, batch_size=100, shuffle=False)
    testloader = [testloader_a_t,testloader_a_ts,testloader_as_t,testloader_as_ts]
    model.train()
    dataloader.dataset.ng_sample()
    loss, n = 0.0, 0
    for uid, iid_pos, iid_neg in tqdm(dataloader,
                                      desc='Epoch No.%i (training)' % epoch,
                                      leave=False, disable=True):
        uid, iid_pos, iid_neg = uid.to(device), iid_pos.to(device), iid_neg.to(device)
        batch_loss = model.bprloss(uid, iid_pos, iid_neg)
        loss += batch_loss
        n += 1
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    loss /= n
    logging.info('EPOCH %d: loss %.4f.' % (epoch, loss))

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
                rating = model(user, item)
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
print('best HR:{}, best NDCG:{}'.format(best_HR,best_NDCG))
with open('results/'+args.model_name+'/%s.csv' % args.experiment_name, 'a') as f:
    f.write('\t'.join([args.experiment_id,
                       '%.4f'%best_HR[0][0],  '%.4f'%best_HR[0][1],  '%.4f'%best_HR[0][2],  '%.4f'%best_HR[0][3],
                       '%.4f'%best_NDCG[0][0],  '%.4f'%best_NDCG[0][1],  '%.4f'%best_NDCG[0][2],  '%.4f'%best_NDCG[0][3],
                       '%.4f' % best_HR[1][0], '%.4f' % best_HR[1][1], '%.4f' % best_HR[1][2],'%.4f' % best_HR[1][3],
                       '%.4f' % best_NDCG[1][0], '%.4f' % best_NDCG[1][1], '%.4f' % best_NDCG[1][2],'%.4f' % best_NDCG[1][3],
                       '%.4f' % best_HR[2][0], '%.4f' % best_HR[2][1], '%.4f' % best_HR[2][2],'%.4f' % best_HR[2][3],
                       '%.4f' % best_NDCG[2][0], '%.4f' % best_NDCG[2][1], '%.4f' % best_NDCG[2][2],'%.4f' % best_NDCG[2][3],
                       '%.4f' % best_HR[3][0], '%.4f' % best_HR[3][1], '%.4f' % best_HR[3][2],'%.4f' % best_HR[3][3],
                       '%.4f' % best_NDCG[3][0], '%.4f' % best_NDCG[3][1], '%.4f' % best_NDCG[3][2],'%.4f' % best_NDCG[3][3],
                       ]) + '\n')

logging.info('Experiment finished.')
