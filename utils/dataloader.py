import pandas as pd
import random
import heapq
import numpy as np
import torch
import os
import math
from time import time

class Loader():
    def __init__(self,config):
        print(f'loading [{config.dataset}]')
        self.config = config
        self.overlapratio = config.LS_overlapratio
        self.sdata = pd.read_csv('../data/%s/s_final.csv'%(config.dataset),names=['u','i','r','t'])
        self.tdata = pd.read_csv('../data/%s/t_final.csv'%(config.dataset),names=['u','i','r','t'])
        self.n_users = self.sdata['u'].value_counts().shape[0]
        self.n_tusers = self.tdata['u'].value_counts().shape[0]
        self.n_sitems = self.sdata['i'].value_counts().shape[0]
        self.n_titems = self.tdata['i'].value_counts().shape[0]
        self.n_items = self.n_sitems + self.n_titems
        print('New Dataset Choose Over!\n users: {},tusers: {}, sitems: {}, titems: {}\n s records: {}, t records: {}'.format(
            self.n_users,self.n_tusers,self.n_sitems,self.n_titems,self.sdata.shape[0],self.tdata.shape[0]
        ))
        self.generate_dataset()

    def build_dataset_one_domain(self):
        self.dataset = self.sdata.append(self.tdata)
        self.train_data = self.dataset.sample(frac=0.8)
        self.test_data = self.dataset[~self.dataset.index.isin(self.train_data.index)]

        lines = []
        for (u,i) in zip(list(self.test_data['u']),list(self.test_data['i'])):
            line = [u,i]
            for _ in range(99):
                j = np.random.randint(self.n_items)
                while self.y[u][j] == 1:
                    j = np.random.randint(self.n_items)
                line.append(j)
            lines.append(line)

        self.train_data.to_csv('../data/amazon/test/train.csv', index=False, header=False)
        self.test_data.to_csv('../data/amazon/test/test.csv', index=False, header=False)
        with open('../data/amazon/test/test_negetive.txt', 'w') as f:
            for i in lines:
                for j in i:
                    f.write(str(j))
                    f.write(' ')
                f.write('\n')
            f.close()

    def build_dataset_cross_domain_step1(self):
        n_test_users = int(self.n_users * 0.2)
        users = self.sdata['u'].value_counts().index._values
        test_users = random.sample(list(users), n_test_users)
        remain_users = list(set(users) - set(test_users))
        tmp_train_data = pd.DataFrame()
        tmp_test_data = pd.DataFrame()
        for u in test_users:
            tmp_train_data = tmp_train_data.append(self.sdata[self.sdata['u'] == u])
            tmp_test_data = tmp_test_data.append(self.tdata[self.tdata['u'] == u])
        for u in remain_users:
            tmp_train_data = tmp_train_data.append(self.sdata[self.sdata['u'] == u])
            tmp_train_data = tmp_train_data.append(self.tdata[self.tdata['u'] == u])
        tmp_train_data = tmp_train_data.sample(frac=1).reset_index(drop=True)
        tmp_test_data = tmp_test_data.sample(frac=1).reset_index(drop=True)
        tmp_train_data.to_csv('../data/amazon/train_cd.csv',index=False,header=False)
        tmp_test_data.to_csv('../data/amazon/test_cd.csv',index=False,header=False)

        # def build_dataset_cross_domain_step2(self):
        print('user max {}, user num {}, item max {}, item num {}'.format(self.train_data['u'].max()+1,
                                                                          self.train_data['u'].value_counts().shape[0],
                                                                          self.train_data['i'].max()+1,
                                                                          self.train_data['i'].value_counts().shape[0]))
        lines = []
        for (u,i) in zip(list(self.test_data['u']),list(self.test_data['i'])):
            line = [u,i]
            for _ in range(99):
                j = np.random.randint(self.n_sitems,self.n_items)
                while self.y[u][j] == 1:
                    j = np.random.randint(self.n_sitems,self.n_items)
                line.append(j)
            lines.append(line)

        with open('../data/amazon/test/test_negetive_cd.txt', 'w') as f:
            for i in lines:
                for j in i:
                    f.write(str(j))
                    f.write(' ')
                f.write('\n')
            f.close()

    def build_STDataset(self):
        result = pd.DataFrame(columns=['u', 'si', 'ti'])
        user_list = self.sdata['u'].value_counts().index._values
        for u in user_list:
            temp_a = self.sdata[self.sdata['u'] == u]
            temp_b = self.tdata[self.tdata['u'] == u]
            n_a = temp_a.shape[0]
            n_b = temp_b.shape[0]
            temp = pd.DataFrame(columns=['u', 'si', 'ti'])
            if n_a > n_b:
                temp['u'] = temp_a['u'].reset_index(drop=True)
                temp['si'] = temp_a['i'].reset_index(drop=True)
                temp['ti'] = temp_b['i'].sample(n=n_a, replace=True).reset_index(drop=True)
            elif n_a < n_b:
                temp['u'] = temp_b['u'].reset_index(drop=True)
                temp['si'] = temp_a['i'].sample(n=n_b, replace=True).reset_index(drop=True)
                temp['ti'] = temp_b['i'].reset_index(drop=True)
            else:
                temp['u'] = temp_b['u'].reset_index(drop=True)
                temp['si'] = temp_a['i'].reset_index(drop=True)
                temp['ti'] = temp_b['i'].reset_index(drop=True)
            result = result.append(temp)

    def generate_dataset(self):
        self.tdata['i'] = self.tdata['i'] + self.n_sitems
        self.y = np.zeros((self.n_users, self.n_items))
        for i in range(self.sdata.shape[0]):
            self.y[self.sdata['u'].iloc[i]][self.sdata['i'].iloc[i]] = 1.0
        for i in range(self.tdata.shape[0]):
            self.y[self.tdata['u'].iloc[i]][self.tdata['i'].iloc[i]] = 1.0
        if not os.path.exists('../data/amazon/train_cd.csv'):
            print('bulid dataset......')
            start = time()
            self.build_dataset_cross_domain_step1()
            print('bulid over! cost {:4f}s!'.format(time()-start))
        self.train_data = pd.read_csv('../data/amazon/train_cd.csv', ',', names=['u', 'i', 'r', 't'])
        self.test_data = pd.read_csv('../data/amazon/test_cd.csv', ',', names=['u', 'i', 'r', 't'])
        self.train_sdata = self.train_data[self.train_data['i']<self.n_sitems]
        self.train_tdata = self.train_data[self.train_data['i']>=self.n_sitems]
        self.train_tdata['i'] = self.train_tdata['i'] - self.n_sitems

        print('s:\nuser max {}, user num {}, item max {}, item num {}'.format(self.train_sdata['u'].max() + 1,
                                                                          self.train_sdata['u'].value_counts().shape[0],
                                                                          self.train_sdata['i'].max() + 1,
                                                                          self.train_sdata['i'].value_counts().shape[0]))
        print('t:\nuser max {}, user num {}, item max {}, item num {}'.format(self.train_tdata['u'].max() + 1,
                                                                          self.train_tdata['u'].value_counts().shape[0],
                                                                          self.train_tdata['i'].max() + 1,
                                                                          self.train_tdata['i'].value_counts().shape[0]))
        # self.build_STDataset()
        # print(self.train_tdata['i'].min())
        self.sy = np.zeros((self.n_users, self.n_sitems))
        self.ty = np.zeros((self.n_users, self.n_sitems))
        for i in range(self.train_sdata.shape[0]):
            self.sy[self.train_sdata['u'].iloc[i]][self.train_sdata['i'].iloc[i]] = 1.0
        for i in range(self.train_tdata.shape[0]):
            self.ty[self.train_tdata['u'].iloc[i]][self.train_tdata['i'].iloc[i]-self.n_sitems] = 1.0
        self.train_users = self.train_data['u'].value_counts().index._values
        self.test_users = self.test_data['u'].value_counts().index._values

        self.n_overlaps = int(self.overlapratio * self.n_users)
        self.val_users = list(set(self.train_users) - set(self.test_users))
        self.overlap_users = heapq.nsmallest(self.n_overlaps,self.val_users)
        temp_overlap_user = max(self.overlap_users)
        self.train_overlapdata = self.train_tdata[self.train_tdata['u']<=temp_overlap_user]

        self.train_sdata_batchsize = math.ceil(self.train_sdata.shape[0]/self.config.n_batch)
        self.train_tdata_batchsize = math.ceil(self.train_tdata.shape[0]/self.config.n_batch)
        self.train_overdata_batchsize = math.ceil(self.train_overlapdata.shape[0]/self.config.n_batch)



        self.test_negetive = []
        with open('../data/amazon/test/test_negetive_cd.txt', 'r') as fd:
            line = fd.readline()
            while line != None and line != '':
                arr = line.split(' ')
                if arr[-1] == '\n':
                    arr = arr[:-1]
                u = int(arr[0])
                for i in arr[1:]:
                    self.test_negetive.append([u, int(i)-self.n_sitems])
                line = fd.readline()
        print('train_data:{}, test_data:{}'.format(self.train_data.shape[0],self.test_data.shape[0]))

class PairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, type, num_ng, is_training = None):
        super(PairwiseDataset,self).__init__()
        if type == 'source':
            self.data = pd.read_csv('../data/amazon/ori_source.csv', ',', names=['u', 'i', 'r', 't'])
            self.n_items = self.data['i'].value_counts().shape[0]
            self.n_users = self.data['u'].value_counts().shape[0]
            self.R = np.zeros((self.n_users, self.n_items))
        else:
            self.data = pd.read_csv('../data/amazon/train_single.csv', ',', names=['u', 'i', 'r', 't'])
            self.n_items = self.data['i'].value_counts().shape[0]
            self.n_users = self.data['u'].value_counts().shape[0]
            self.R = np.zeros((self.n_users, self.n_items))

        # self.is_training = is_training
        self.num_ng = num_ng

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        items_ng = []
        for (u,i) in zip(self.data['u'], self.data['i']):
            j = np.random.randint(self.n_items)
            while (u, j) in self.R:
                j = np.random.randint(self.n_items)
            items_ng.append(j)
        self.final_data = (self.data['u'], self.data['i'], items_ng)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        user = self.final_data[0][idx]
        item = self.final_data[1][idx]
        item_ng = self.final_data[2][idx]
        return user, item, item_ng

class PointwiseDataset(torch.utils.data.Dataset):
    def __init__(self,loader,data,num_ng,is_training = None):
        super(PointwiseDataset,self).__init__()
        self.data = data
        self.n_items = loader.n_titems+loader.n_sitems
        self.y = loader.y

        self.is_training = is_training
        self.num_ng = num_ng
        if is_training:
            self.users = list(self.data['u'])
            self.items = list(self.data['i'])
            self.labels = [1 for _ in range(len(self.users))]
        else:
            self.labels = [1 for _ in range(len(self.data))]

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        users = []
        items_ng = []
        data_labels_ng = []
        for (u,i) in zip(self.users,self.items):
            for _ in range(self.num_ng):
                users.append(u)
                data_labels_ng.append(0)
                j = np.random.randint(self.n_items)
                while self.y[u][j] == 1:
                    j = np.random.randint(self.n_items)
                items_ng.append(j)

        users.extend(self.users)
        items_ng.extend(self.items)
        data_labels_ng.extend(self.labels)
        self.data_ = (users,items_ng,data_labels_ng)

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        if self.is_training:
            user = self.data_[0][idx]
            item = self.data_[1][idx]
            label = self.data_[2][idx]
        else:
            user = self.data[idx][0]
            item = self.data[idx][1]
            label = self.labels[idx]
        return user, item ,label

class CDPointwiseDataset(torch.utils.data.Dataset):
    def __init__(self,loader,data,mode,num_ng):
        super(CDPointwiseDataset,self).__init__()
        self.data = data
        if mode == 's':
            self.n_items = loader.n_sitems
            self.y = loader.sy
        else:
            self.n_items = loader.n_titems
            self.y = loader.ty
        self.mode = mode

        self.num_ng = num_ng
        self.users = list(self.data['u'])
        self.items = list(self.data['i'])
        self.labels = [1 for _ in range(len(self.users))]

    def ng_sample(self):
        users = []
        items_ng = []
        data_labels_ng = []
        for (u,i) in zip(self.users,self.items):
            for _ in range(self.num_ng):
                users.append(u)
                data_labels_ng.append(0)
                j = np.random.randint(self.n_items)
                while self.y[u][j] == 1:
                    j = np.random.randint(self.n_items)
                items_ng.append(j)

        users.extend(self.users)
        items_ng.extend(self.items)
        data_labels_ng.extend(self.labels)
        self.data_ = (users,items_ng,data_labels_ng)

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        user = self.data_[0][idx]
        item = self.data_[1][idx]
        label = self.data_[2][idx]
        return user, item ,label

class OLPointwiseDataset(torch.utils.data.Dataset):
    def __init__(self,loader,sdata,tdata):
        self.sdata = sdata
        self.tdata = tdata
        self.susers = list(self.sdata['u'])
        self.items = list(self.sdata['i'])
        self.labels = [1 for _ in range(len(self.users))]

    def ng_sample(self):
        users = []
        items_ng = []
        data_labels_ng = []
        for (u,i) in zip(self.users,self.items):
            for _ in range(self.num_ng):
                users.append(u)
                data_labels_ng.append(0)
                j = np.random.randint(self.n_items)
                while self.y[u][j] == 1:
                    j = np.random.randint(self.n_items)
                items_ng.append(j)

        users.extend(self.users)
        items_ng.extend(self.items)
        data_labels_ng.extend(self.labels)
        self.data_ = (users,items_ng,data_labels_ng)

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        user = self.data_[0][idx]
        item = self.data_[1][idx]
        label = self.data_[2][idx]
        return user, item ,label


class CDModePointwiseDataset(torch.utils.data.Dataset):
    def __init__(self,loader,data,num_ng):
        super(CDModePointwiseDataset,self).__init__()
        self.n_sitems = loader.n_sitems
        self.n_titems = loader.n_titems
        self.sy = loader.sy
        self.ty = loader.ty
        self.n_items = loader.n_titems
        self.data = data


        self.num_ng = num_ng
        self.users = list(self.data['u'])
        self.items = list(self.data['i'])
        self.mode = list(self.data['mode'])
        self.labels = [1 for _ in range(len(self.users))]

    def ng_sample(self):
        users = []
        modes = []
        items_ng = []
        data_labels_ng = []
        for (u,i,mod) in zip(self.users,self.items,self.mode):
            for _ in range(self.num_ng):
                users.append(u)
                modes.append(mod)
                data_labels_ng.append(0)
                if mod == 's':
                    temp_n = self.n_sitems
                    y = self.sy
                else:
                    temp_n = self.n_titems
                    y = self.ty
                j = np.random.randint(self.temp_n)
                while y[u][j] == 1:
                    j = np.random.randint(self.temp_n)
                items_ng.append(j)

        users.extend(self.users)
        modes.extend(self.mode)
        items_ng.extend(self.items)
        data_labels_ng.extend(self.labels)
        self.data_ = (users,items_ng,modes,data_labels_ng)

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        user = self.data_[0][idx]
        item = self.data_[1][idx]
        mode = self.data_[2][idx]
        label = self.data_[3][idx]
        return user, item ,label
