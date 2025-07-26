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
from utils.dataset import PairwiseDataset, TestDataset
from torch.utils.data import DataLoader
from models import setup_model

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

import warnings
warnings.filterwarnings('ignore')

def parse_opt():
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--experiment_id', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'), help='')
    parser.add_argument('--experiment_name', type=str, default=None, help='')
    parser.add_argument('--model_name', type=str, default='LightGCN', help='options: [GCDTN,MF,EMCDR,NCF,MATN,CMF,CDATN,LightGCN]')
    parser.add_argument('--disable_tqdm', action='store_true', default=False, help='')
    parser.add_argument('--gpu_id', type=int, default=0, help='')

    # Data
    parser.add_argument('--data_name', type=str, default='movie-book', help="Musical_Patio")
    parser.add_argument('--batch_size', type=int, default=512, help="b")
    parser.add_argument('--EPOCHS', type=int, default=200, help="b")
    parser.add_argument('--init_lr', type=float, default=0.001, help="b")
    parser.add_argument('--test_num_ng', type=int, default=99, help="b")
    parser.add_argument('--train_num_ng', type=int, default=4, help="b")
    # GCDTN
    parser.add_argument('--GCDTN_embed_dim', type=int, default=32, help="b")
    parser.add_argument('--GCDTN_GCN_layers', type=int, default=2, help="b")


    # CDATN
    parser.add_argument('--CDATN_embed_dim', type=int, default=32, help="b")
    parser.add_argument('--CDATN_GCN_layers', type=int, default=2, help="b")
    # LightGCN
    parser.add_argument('--LightGCN_embed_dim', type=int, default=32, help="b")
    parser.add_argument('--LightGCN_n_layers', type=int, default=2, help="b")


    #
    # parser.add_argument('--cl_reference_length', type=int, default=100,
    #                     help='')

    args = parser.parse_args()

    # Specific settings
    args.experiment_dir = os.path.join('log', args.model_name, args.experiment_id)
    args.experiment_name = str(args.experiment_id)
    args.topKs = [1, 2, 5, 10]
    return args

def train():
    train_data = PairwiseDataset('data/movie-book/target_train.csv', args.train_num_ng)
    test_data_a_t = TestDataset('data/movie-book/test_a_t.txt')
    test_data_a_ts = TestDataset('data/movie-book/test_a_ts.txt')
    test_data_as_t = TestDataset('data/movie-book/test_as_t.txt')
    test_data_as_ts = TestDataset('data/movie-book/test_as_ts.txt')
    model = setup_model(args, train_data)
    model.to(device)
    # print("Total number of param in {} is {}".format(args.model_name,sum(x.numel() for x in model.parameters())))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, threshold=0.01,factor=0.1, min_lr=1e-6)
    best_HR, best_NDCG = [[0, 0, 0, 0] for _ in range(4)], [[0, 0, 0, 0] for _ in range(4)]
    for k_ep in range(args.EPOCHS):
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
                                          desc='Epoch No.%i (training)' % k_ep,
                                          leave=False, disable=args.disable_tqdm):
            uid, iid_pos, iid_neg = uid.to(device), iid_pos.to(device), iid_neg.to(device)
            batch_loss = model.bprloss(uid, iid_pos, iid_neg)
            loss += batch_loss
            n += 1
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        loss /= n
        logging.info('EPOCH %d: loss %.4f.' % (k_ep, loss))

        if k_ep%10==0:
            for testloader_index in range(4):
                HR, NDCG = predict(model, testloader[testloader_index])
                logging.info('EPOCH {}: HR: {}, NDCG: {}'.format(k_ep, HR, NDCG))
                if HR[-1]>best_HR[testloader_index][-1]:
                    best_HR[testloader_index] = HR
                    best_NDCG[testloader_index] = NDCG

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

#
# def predict(model, dataset):
#     model.eval()
#     results = {'precision': np.zeros(len(args.topKs)),
#                'HR': np.zeros(len(args.topKs)),
#                'NDCG': np.zeros(len(args.topKs))}
#
#     u_batch_size = 100
#
#     with torch.no_grad():
#         users = dataset.users
#         users_list = []
#         rating_list = []
#         groundTrue_list = []
#         total_batch = len(users) // u_batch_size + 1
#         for batch_users in minibatch(users, batch_size=u_batch_size):
#             allPos = [dataset.train_dict[u] for u in batch_users]
#             groundTrue = [dataset.test_dict[u] for u in batch_users]
#             batch_users = torch.Tensor(batch_users).long().to(device)
#
#             rating = model.getUsersRating(batch_users)
#             exclude_index = []
#             exclude_items = []
#             for range_i, items in enumerate(allPos):
#                exclude_index.extend([range_i] * len(items))
#                exclude_items.extend(items)
#             rating[exclude_index, exclude_items] = -(1 << 10)
#             _, rating_K = torch.topk(rating, k=max(args.topKs))
#             rating = rating.cpu().numpy()
#             users_list.append(batch_users)
#             rating_list.append(rating_K.cpu())
#             groundTrue_list.append(groundTrue)
#         assert total_batch == len(users_list)
#         X = zip(rating_list, groundTrue_list)
#         pre_results = []
#         for x in X:
#             pre_results.append(test_one_batch(x,args.topKs))
#         # scale = float(u_batch_size / len(users))
#         for result in pre_results:
#             results['HR'] += result['recall']
#             results['precision'] += result['precision']
#             results['NDCG'] += result['ndcg']
#         results['HR'] /= float(len(users))
#         results['precision'] /= float(len(users))
#         results['NDCG'] /= float(len(users))
#         print(results)
#         return results

if __name__ == '__main__':
    args = parse_opt()
    if not os.path.isdir(args.experiment_dir):
        os.makedirs(args.experiment_dir)
    batch_time = -1
    logging.basicConfig(filename='%s.log' % args.experiment_dir, format='[%(levelname)s] %(asctime)s %(message)s ')
    coloredlogs.install(level='INFO', fmt='[%(levelname)s] %(message)s %(asctime)s')
    logging.info('Experiment ID.%s start ......' % args.experiment_id)
    logging.info(args)
    train()
    logging.info('Experiment finished.')
