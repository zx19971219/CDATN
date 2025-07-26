import pandas as pd
import numpy as np
import os
from time import time
import random


dataset = 'amazon'
path = '%s/' % (dataset)

def build_dataset():
    sdata = pd.read_csv(path + 'ori_source.csv', names=['u', 'i', 'r', 't'])
    tdata = pd.read_csv(path + 'ori_target.csv', names=['u', 'i', 'r', 't'])
    n_users = sdata['u'].value_counts().shape[0]
    n_tusers = tdata['u'].value_counts().shape[0]
    n_sitems = sdata['i'].value_counts().shape[0]
    n_titems = tdata['i'].value_counts().shape[0]
    n_items = n_sitems + n_titems
    print('New Dataset Choose Over!\n users: {},tusers: {}, sitems: {}, titems: {}\n s records: {}, t records: {}'.format(
        n_users, n_tusers, n_sitems, n_titems, sdata.shape[0], tdata.shape[0]
    ))

    tdata['i'] = tdata['i'] + n_sitems
    y = np.zeros((n_users, n_items))
    for i in range(sdata.shape[0]):
        y[sdata['u'].iloc[i]][sdata['i'].iloc[i]] = 1.0
    for i in range(tdata.shape[0]):
        y[tdata['u'].iloc[i]][tdata['i'].iloc[i]] = 1.0
    if not os.path.exists(path+'train_cd.csv'):
        print('bulid dataset......')
        start = time()
        n_test_users = int(n_users * 0.2)
        users = sdata['u'].value_counts().index._values
        test_users = random.sample(list(users), n_test_users)
        remain_users = list(set(users) - set(test_users))
        tmp_train_data = pd.DataFrame()
        tmp_test_data = pd.DataFrame()
        # tmp_train_data_source = pd.DataFrame()
        tmp_train_data_target = pd.DataFrame()
        for u in test_users:
            tmp_train_data = tmp_train_data.append(sdata[sdata['u'] == u])
            tmp_test_data = tmp_test_data.append(tdata[tdata['u'] == u])
        for u in remain_users:
            tmp_train_data = tmp_train_data.append(sdata[sdata['u'] == u])
            tmp_train_data = tmp_train_data.append(tdata[tdata['u'] == u])
            tmp_train_data_target = tmp_train_data_target.append(tdata[tdata['u'] == u])

        tmp_train_data = tmp_train_data.sample(frac=1).reset_index(drop=True)
        tmp_train_data_target = tmp_train_data_target.append(tdata[tdata['u'] == u])
        tmp_train_data_target['i'] = tmp_train_data_target['i'] - n_sitems
        tmp_test_data = tmp_test_data.sample(frac=1).reset_index(drop=True)
        tmp_train_data.to_csv(path+'train_cross.csv',index=False,header=False)
        tmp_train_data_target.to_csv(path+'train_single.csv',index=False,header=False)
        # tmp_test_data.to_csv('../data/amazon/test_cd.csv',index=False,header=False)

        # def build_dataset_cross_domain_step2(self):
        print('user max {}, user num {}, item max {}, item num {}'.format(tmp_train_data['u'].max()+1,
                                                                          tmp_train_data['u'].value_counts().shape[0],
                                                                          tmp_train_data['i'].max()+1,
                                                                          tmp_train_data['i'].value_counts().shape[0]))
        test_single = []
        test_cross = []
        for (u,i) in zip(list(tmp_test_data['u']),list(tmp_test_data['i'])):
            test_single_tmp = [u,i-n_sitems]
            test_cross_tmp = [u,i]
            for _ in range(99):
                j = np.random.randint(n_sitems,n_items)
                while y[u][j] == 1:
                    j = np.random.randint(n_sitems,n_items)
                test_cross_tmp.append(j)
                test_single_tmp.append(j-n_sitems)
            test_cross.append(test_cross_tmp)
            test_single.append(test_single_tmp)

        with open(path+'test_sample100_cross.txt', 'w') as f:
            for i in test_cross:
                for j in i:
                    f.write(str(j))
                    f.write(' ')
                f.write('\n')
            f.close()

        with open(path+'test_sample100_single.txt', 'w') as f:
            for i in test_single:
                for j in i:
                    f.write(str(j))
                    f.write(' ')
                f.write('\n')
            f.close()

        print('bulid over! cost {:4f}s!'.format(time()-start))

def build_movie_book():
    sdata = pd.read_csv(path + 'ori_source.csv', names=['u', 'i', 'r', 't'])
    tdata = pd.read_csv(path + 'ori_target.csv', names=['u', 'i', 'r', 't'])
    n_users = sdata['u'].value_counts().shape[0]
    n_tusers = tdata['u'].value_counts().shape[0]
    n_sitems = sdata['i'].value_counts().shape[0]
    n_titems = tdata['i'].value_counts().shape[0]
    n_items = n_sitems + n_titems
    print(
        'New Dataset Choose Over!\n users: {},tusers: {}, sitems: {}, titems: {}\n s records: {}, t records: {}'.format(
            n_users, n_tusers, n_sitems, n_titems, sdata.shape[0], tdata.shape[0]
        ))
    t_n_records = tdata['u'].value_counts()
    uids = t_n_records[t_n_records>=5].index._data
    t_n_records = tdata['i'].value_counts()
    iids = t_n_records[t_n_records>=10].index._data
    #print('Number of users (records >=5 )in t domain:', len(t_test_uids))
    sdata = sdata[sdata.u.isin(uids)]
    tdata = tdata[tdata.u.isin(uids)]
    tdata = tdata[tdata.i.isin(iids)]
    sdata['u'].replace(uids,range(len(uids)),inplace=True)
    tdata['u'].replace(uids,range(len(uids)),inplace=True)
    tdata['i'].replace(iids,range(len(iids)),inplace=True)
    #sdata = sdata.replace({'u':uid_dict})
    #tdata = tdata.replace({'u':uid_dict})


    tdata_train = pd.DataFrame(columns=['u', 'i', 'r', 't'])
    tdata_test = pd.DataFrame(columns=['u', 'i', 'r', 't'])
    for uid in range(len(uids)):
        t_inds = tdata[tdata['u']==uid].index._data
        temp_n = min(10,int(1/2*len(t_inds)))
        tdata_train = tdata_train.append(tdata.loc[t_inds[:-temp_n]])
        tdata_test = tdata_test.append(tdata.loc[t_inds[-temp_n:]])

    sdata.to_csv('movie-book/source.csv', index=False)
    tdata_train.to_csv('movie-book/target.csv', index=False)
    tdata_test.to_csv('movie-book/target_test.csv', index=False)

    print(
        'New Dataset Choose Over!\n susers: {},sitems: {}, tusers: {}, titems: {},\n '
        'tusers_train: {}, titems_train: {}, t_u_test: {}, t_i_test: {},\n '
        's records: {}, t records: {}/{}'.format(
            sdata['u'].value_counts().shape[0], sdata['i'].value_counts().shape[0],
            tdata['u'].value_counts().shape[0], tdata['i'].value_counts().shape[0],
            tdata_train['u'].value_counts().shape[0], tdata_train['i'].value_counts().shape[0],
            tdata_test['u'].value_counts().shape[0], tdata_test['i'].value_counts().shape[0],
            sdata.shape[0], tdata_train.shape[0], tdata_test.shape[0]
        ))

    print('pause')

build_movie_book()
