import pandas as pd
import random
import heapq
import numpy as np
import torch
import os
import math
from time import time
import scipy.sparse as sp

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class PairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, filename, num_ng, is_training = None):
        super(PairwiseDataset,self).__init__()
        self.data = pd.read_csv(filename, header=0)
        with open('data/movie-book/n_users.txt') as file:
            self.n_users = int(file.read().rstrip())
        self.n_items = max(self.data['i'])+1
        self.R = np.zeros((self.n_users, self.n_items))
        self.R[self.data['u'],self.data['i']] = 1
        n_record = np.sum(self.R,axis=1)
        self.ratio = n_record
        for i in range(len(self.ratio)):
            if self.ratio[i]>20: #修改
                self.ratio[i] = 1
            else:
                self.ratio[i] /= 20
        self.num_ng = num_ng
        # self.sp_Graph = self.getSparseGraph()

    def ng_sample(self):
        users = []
        items_ps = []
        items_ng = []
        for (u,i) in zip(self.data['u'], self.data['i']):
            for _ in range(self.num_ng):
                j = np.random.randint(self.n_items)
                while (u, j) in self.R:
                    j = np.random.randint(self.n_items)
                users.append(u)
                items_ps.append(i)
                items_ng.append(j)
        self.final_data = (users, items_ps, items_ng)

    def __len__(self):
        return self.data.shape[0]*(self.num_ng)

    def __getitem__(self, idx):
        user = self.final_data[0][idx]
        item = self.final_data[1][idx]
        item_ng = self.final_data[2][idx]
        return user, item, item_ng

    def getSparseGraph(self):
        print("loading adjacency matrix")
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = sp.csr_matrix(self.R)
        R = R.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.Graph = self.Graph.coalesce().to(device)
        print("don't split the matrix")
        return self.Graph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

class PointwiseDataset(torch.utils.data.Dataset):
    def __init__(self,filename, num_ng, is_training = None):
        super(PointwiseDataset,self).__init__()
        self.data = pd.read_csv(filename, header=0)
        self.n_users = max(self.data['u'])+1
        self.n_items = max(self.data['i'])+1
        self.num_ng = num_ng

    def build_graph(self):
        self.R = np.zeros((self.n_users, self.n_items))
        self.R[self.data['u'], self.data['i']] = 1
        self.sp_Graph = self.getSparseGraph()

    def ng_sample(self):
        users = []
        items = []
        labels = []
        for (u,i) in zip(self.data['u'], self.data['i']):
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(self.num_ng):
                j = np.random.randint(self.n_items)
                while (u, j) in self.R:
                    j = np.random.randint(self.n_items)
                users.append(u)
                items.append(j)
                labels.append(0)
        self.final_data = (users, items, labels)

    def __len__(self):
        return self.data.shape[0]*(self.num_ng+1)

    def __getitem__(self, idx):
        user = self.final_data[0][idx]
        item = self.final_data[1][idx]
        label = self.final_data[2][idx]
        return user, item, label


    def getSparseGraph(self):
        print("loading adjacency matrix")
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = sp.csr_matrix(self.R)
        R = R.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.Graph = self.Graph.coalesce().to(device)
        print("don't split the matrix")
        return self.Graph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, filename='/test_a_t.txt'):
        super(TestDataset, self).__init__()
        # dataset = 'movie-book'
        self.test_data = self.read_data(filename)

    def read_data(self, filename):
        tmp_data = []
        with open(filename, 'r') as fd:
            line = fd.readline()
            while line != None and line != '':
                arr = line.split(' ')
                if arr[-1] == '\n':
                    arr = arr[:-1]
                u = int(arr[0])
                for i in arr[1:]:
                    tmp_data.append([u, int(i)])
                line = fd.readline()
        return tmp_data

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        return self.test_data[idx][0], self.test_data[idx][1]

class testDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(testDataset,self).__init__()
        dataset = 'movie-book'
        self.users = []
        self.items = []
        self.max_uid = -1
        self.max_iid = -1
        dataset = pd.read_csv('data/'+dataset+'/target_test.csv',header=0)
        self.users = []
        self.items = []
        for i in dataset.groupby('u'):
            self.users.append(i[0])
            self.items.append(i[1]['i'].values)
        # self.data = pd.DataFrame({'users': self.users,'items': self.items})

    # def __len__(self):
    #     return len(self.users)
    #
    # def __getitem__(self, idx):
    #     return self.users[idx], self.items[idx]

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, num_ng, is_training = None):
        super(TrainDataset,self).__init__()
        dataset = 'movie-book'
        self.train_dataset = pd.read_csv('data/'+dataset+'/train.csv',header=0)
        test_dataset = pd.read_csv('data/'+dataset+'/test.csv',header=0)
        self.n_items = max(max(self.train_dataset['i']),max(test_dataset['i']))+1
        self.n_users = max(max(self.train_dataset['u']),max(test_dataset['u']))+1
        self.R = np.zeros((self.n_users, self.n_items))
        self.R[self.train_dataset['u'], self.train_dataset['i']] = 1
        self.t_graph = self.getSparseGraph(self.R)
        # self.is_training = is_training
        self.num_ng = num_ng

        self.train_dict = self.build_dict(self.train_dataset)
        self.test_dict = self.build_dict(test_dataset)

    def build_dict(self,dataset):
        users = []
        items = []
        for i in dataset.groupby('u'):
            items.append(i[1]['i'].values)
            users.append(i[0])
        return dict(zip(users,items))

    def getSparseGraph(self,R):
        print("loading adjacency matrix")
        adj_mat = sp.dok_matrix((len(R) + len(R[0]), len(R) + len(R[0])), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        # R = sp.lil_matrix(R)
        adj_mat[:len(R), len(R):] = R
        adj_mat[len(R):, :len(R)] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        return self.Graph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def ng_sample(self):
        # assert self.is_training, 'no need to sampling when testing'
        t_u,t_i,t_y = [],[],[]
        for (u,i) in zip(self.train_dataset['u'], self.train_dataset['i']):
            t_u[u].append(u)
            t_i[u].append(i)
            t_y[u].append(1)
            for _ in range(self.num_ng):
                j = np.random.randint(self.n_items)
                while (u, j) in self.R:
                    j = np.random.randint(self.n_items)
                t_u[u].append(u)
                t_i[u].append(j)
                t_y[u].append(0)

        self.data = (t_u, t_i, t_y)

    def __len__(self):
        return self.train_dataset.shape[0]*(1+self.num_ng)

    def __getitem__(self, item):
        return self.data[0][item],self.data[1][item],self.data[2][item]

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.data_dict[user])
        return posItems
