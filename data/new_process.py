import pandas as pd
import numpy as np
import os
from time import time
import random
import random

dataset = 'amazon'
path = '%s/' % (dataset)

def build_movie_book():
    adata = pd.read_csv(path + 'ori_source.csv', names=['u', 'i', 'r', 't'])
    tdata = pd.read_csv(path + 'ori_target.csv', names=['u', 'i', 'r', 't'])

    a_n_records = adata['u'].value_counts()
    uids = a_n_records[a_n_records>=5].index._data
    adata = adata[adata.u.isin(uids)]

    t_n_records = tdata['u'].value_counts()
    uids = t_n_records[t_n_records>=5].index._data
    tdata = tdata[tdata.u.isin(uids)]

    ori_uid = sorted(list(set((adata['u'])) | set((tdata['u']))))
    new_uid = range(len(ori_uid))
    adata['u'].replace(ori_uid,new_uid,inplace=True)
    tdata['u'].replace(ori_uid,new_uid,inplace=True)


    ori_aiid = sorted(list(set((adata['i']))))
    adata['i'].replace(ori_aiid,range(len(ori_aiid)),inplace=True)
    ori_tiid = sorted(list(set((tdata['i']))))
    tdata['i'].replace(ori_tiid,range(len(ori_tiid)),inplace=True)
    tdata.reset_index(drop=True,inplace=True)
    n_users = len(ori_uid)
    n_ausers = len(list(set((adata['u']))))
    n_tusers = len(list(set((tdata['u']))))
    n_aitems = len(ori_aiid)
    n_titems = len(ori_tiid)
    print('\n users: {},ausers: {}, aitems: {}, tusers: {}, titems: {}\n a records: {}, t records: {}'.format(
            n_users, n_ausers, n_aitems, n_tusers, n_titems, adata.shape[0], tdata.shape[0]
        ))

    # 存储auxiliary域的数据，并存储总user数
    adata.to_csv('movie-book/auxiliary.csv', index=False)
    with open("movie-book/n_users.txt", mode='w') as f:
        f.write(str(n_users))

    R = np.zeros((n_users, n_aitems))
    R[adata['u'], adata['i']] = 1
    test_auxiliary = adata.sample(n=30000)
    test_negative = []
    for (u,i) in zip(test_auxiliary['u'], test_auxiliary['i']):
        line = str(u)+" "+str(i)
        for _ in range(99):
            j = np.random.randint(n_aitems)
            while (u, j) in R:
                j = np.random.randint(n_aitems)
            line += " "+str(j)
        line += '\n'
        test_negative.append(line)
    with open('movie-book/auxiliary_test.txt','w') as file:
        file.writelines(test_negative)

    #  按照user历史记录的数量进行排序，分别存储（a域饱和-t域饱和、a域稀疏-t域饱和、a域饱和-t域稀疏、a域稀疏-t域稀疏）四组数据作为test
    # 5%+5%+5%+5%
    n_test = int(tdata.shape[0]*0.05)
    R = np.zeros((n_users, n_titems))
    R[tdata['u'], tdata['i']] = 1
    debris_index = []
    a_n_records = adata['u'].value_counts()
    t_n_records = tdata['u'].value_counts()

    # a域饱和-t域饱和，顺便存储overlap 30%的用户id
    uids = set((a_n_records.index._data[:int(0.3*n_ausers)])) & set((t_n_records.index._data[:int(0.3*n_tusers)]))
    with open('movie-book/overlap_users.txt','w') as file:
        file.writelines(' '.join([str(i) for i in list(uids)]))
    tmp_index = random.sample(list(tdata[tdata.u.isin(uids)].index),n_test)
    tmp_test_data= tdata.loc[tmp_index]
    test_negative = []
    for (u,i) in zip(tmp_test_data['u'], tmp_test_data['i']):
        line = str(u)+" "+str(i)
        for _ in range(99):
            j = np.random.randint(n_titems)
            while (u, j) in R:
                j = np.random.randint(n_titems)
            line += " "+str(j)
        line += '\n'
        test_negative.append(line)
    with open('movie-book/test_a_t.txt','w') as file:
        file.writelines(test_negative)
    debris_index.extend(tmp_index)


    # a域饱和-t域稀疏
    uids = set((a_n_records.index._data[:int(0.5*n_ausers)])) & set((t_n_records.index._data[-int(0.3*n_tusers):]))
    line = list(tdata[tdata.u.isin(uids)].index)
    tmp_index = random.sample(line,min(len(line),n_test))
    tmp_test_data = tdata.loc[tmp_index]
    test_negative = []
    for (u, i) in zip(tmp_test_data['u'], tmp_test_data['i']):
        line = str(u) + " " + str(i)
        for _ in range(99):
            j = np.random.randint(n_titems)
            while (u, j) in R:
                j = np.random.randint(n_titems)
            line += " " + str(j)
        line += '\n'
        test_negative.append(line)
    with open('movie-book/test_a_ts.txt', 'w') as file:
        file.writelines(test_negative)
    debris_index.extend(tmp_index)

    # a域稀疏-t域饱和
    uids = set((a_n_records.index._data[-int(0.3*n_ausers):])) & set((t_n_records.index._data[:int(0.3*n_tusers)]))
    tmp_index = random.sample(list(tdata[tdata.u.isin(uids)].index),n_test)
    tmp_test_data = tdata.loc[tmp_index]
    test_negative = []
    for (u, i) in zip(tmp_test_data['u'], tmp_test_data['i']):
        line = str(u) + " " + str(i)
        for _ in range(99):
            j = np.random.randint(n_titems)
            while (u, j) in R:
                j = np.random.randint(n_titems)
            line += " " + str(j)
        line += '\n'
        test_negative.append(line)
    with open('movie-book/test_as_t.txt', 'w') as file:
        file.writelines(test_negative)
    debris_index.extend(tmp_index)

    # a域稀疏-t域稀疏
    uids = set((a_n_records.index._data[-int(0.3*n_ausers):])) & set((t_n_records.index._data[-int(0.3*n_tusers):]))
    line = list(tdata[tdata.u.isin(uids)].index)
    tmp_index = random.sample(line,min(len(line),n_test))
    tmp_test_data = tdata.loc[tmp_index]
    test_negative = []
    for (u, i) in zip(tmp_test_data['u'], tmp_test_data['i']):
        line = str(u) + " " + str(i)
        for _ in range(99):
            j = np.random.randint(n_titems)
            while (u, j) in R:
                j = np.random.randint(n_titems)
            line += " " + str(j)
        line += '\n'
        test_negative.append(line)
    with open('movie-book/test_as_ts.txt', 'w') as file:
        file.writelines(test_negative)
    debris_index.extend(tmp_index)

    tdata.drop(index=debris_index,inplace=True)
    tdata.to_csv('movie-book/target_train.csv', index=False)
    print('pause')



build_movie_book()



# 将数据集分成auxiliary域和target域（原始数据集就是这么分的）
# 存储user总数N

# 对于auxiliary域的数据来说
#   使用模型训练得到user的embedding，其中，部分user无历史记录，这部分embedding为全0（也可尝试服从某种分布的随机值）
#   存储供target域使用

# 对于target域的数据来说
#   按照user历史记录的数量进行排序，分别存储（a域饱和-t域饱和、a域稀疏-t域饱和、a域饱和-t域稀疏、a域稀疏-t域稀疏）四组数据作为test

