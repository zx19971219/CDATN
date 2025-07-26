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
import pandas as pd
from sklearn.decomposition import PCA
from time import time
from sklearn.manifold import TSNE
# 在文件顶部添加以下导入（与其他import语句放在一起）
from sklearn.preprocessing import StandardScaler
from scipy import stats  # 用于Z-score计算

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class CDATN(CustomModel):
    def __init__(self, dataset, experiment_id=None, experiment_dir=None,
                 embed_dim=32, K_g=2, D=3, ratio=0,
                 **kwargs):
        super().__init__(experiment_id, experiment_dir)
        self.n_users = dataset.n_users
        self.n_titems = dataset.n_items
        self.Graph = dataset.getSparseGraph()
        self.embed_dim = embed_dim
        self.n_layers = K_g
        self.ratio = torch.Tensor(dataset.ratio).to(device)
        self.weight_decay = 0.0001
        self.D = D
        self.ratio_state = ratio
        self.current_epoch = 0
        self.embedding_user = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.embed_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.n_titems, embedding_dim=self.embed_dim)
        self.aspect_names = ['Popularity', 'Genre', 'Price'][:3]  # Example aspect names
        #os.makedirs('aspect_visualization', exist_ok=True)
        self.transfer_layer = []#mlp
        # for i in range(D):
        #     self.transfer_layer.append(nn.Identity().to(device)) 
        for i in range(D):#原mlp
            self.transfer_layer.append(nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),nn.ReLU()).to(device))
        
        self.mse_loss = nn.MSELoss()
        
    def _plot_tsne(self, users, items, au_embed, epoch):
     with torch.no_grad():
        # 1. 数据准备和t-SNE降维（保持不变）
        aspects = [self.transfer_layer[i](au_embed).cpu() for i in range(self.D)]
        all_embeddings = torch.cat([au_embed.cpu()] + aspects).numpy()
        scaler = StandardScaler()
        scaled_emb = scaler.fit_transform(all_embeddings)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        points = tsne.fit_transform(scaled_emb)
        z_scores = stats.zscore(points, axis=0)  

# 创建掩码：保留Z-score绝对值小于2.5的点（即95%的正常数据）
        valid_mask = (np.abs(z_scores) < 2.5).all(axis=1)
        # 2. 创建白色背景的画布
        plt.figure(figsize=(12, 10), dpi=150, facecolor='white')  # 宽度从14->12
        ax = plt.gca()
        ax.set_facecolor('white')

        plt.rcParams.update({
            'xtick.labelsize': 16,  # X轴刻度字号增大
            'ytick.labelsize': 16,  # Y轴刻度字号增大
            'axes.labelsize': 18,    # 坐标轴标签字号增大（原为12）
            'axes.titlesize': 20,    # 标题再增大
            'legend.title_fontsize': 18,  # 图例标题
            'legend.fontsize': 16,        # 图例条目
        })
        ax.tick_params(axis='both', which='major', 
                      labelsize=16,  # 主刻度字号
                      length=6,      # 刻度线长度
                      width=1.5)     # 刻度线粗细
        
        ax.set_xlabel('t-SNE Dimension 1', 
                     fontsize=18,    # 比刻度大2pt
                     fontweight='bold',
                     labelpad=10)     # 标签与轴的距离
        
        ax.set_ylabel('t-SNE Dimension 2',
                     fontsize=18,
                     fontweight='bold',
                     labelpad=10)

        
        # 4. 绘制点（边缘加粗）
        markers = ['o', 's', 'D', '^', 'v', '<', '>']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # 源域（蓝色圆形）
        plt.scatter(points[:len(users), 0], points[:len(users), 1],
                   s=80, c=colors[0], marker=markers[0],
                   alpha=0.8, label='Source Domain',
                   edgecolor='k', linewidths=0.8)  # 黑色边缘
        
        # 各aspect
        for i in range(self.D):
            start = (i+1)*len(users)
            plt.scatter(points[start:start+len(users), 0], 
                       points[start:start+len(users), 1],
                       s=80, c=colors[(i+1)%len(colors)], 
                       marker=markers[(i+1)%len(markers)],
                       alpha=0.8, label=self.aspect_names[i],
                       edgecolor='k', linewidths=0.8)
        
        # 5. 增强标注
        plt.title(f't-SNE Projection of Cross-Domain Preference Aspects (Epoch {epoch})', fontsize=20,pad=20, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1', fontweight='bold')
        plt.ylabel('t-SNE Dimension 2', fontweight='bold')
        from sklearn.metrics import silhouette_score
        labels = np.repeat(range(self.D + 1), len(users))[valid_mask[:len(points)]]  # 修正索引对齐
        if len(np.unique(labels)) > 1:  # 确保有多个类别
         sil_score = silhouette_score(points, labels)
         plt.text(0.05, 0.95, 
                f'Silhouette Score: {sil_score:.3f}',
                transform=ax.transAxes,
                fontsize=14,
                bbox=dict(facecolor='white', 
                         edgecolor='#444444',
                         alpha=0.8,
                         boxstyle='round,pad=0.5'))
        # 6. 超大图例（带白色半透明背景）
        legend = plt.legend(
    loc='upper right',
    bbox_to_anchor=(0.99, 0.99),
    
    # === 边框控制 ===
    frameon=True,                # 启用边框
    edgecolor='#444444',         # 深灰边框色
    
    # === 间距控制 ===
    borderpad=0.8,               # 图例内容与边框的内间距（单位：字体大小倍数）
    borderaxespad=0.5,           # 图例与坐标轴的间距（单位：英寸）
    
    # === 内容排版 ===
    handletextpad=0.5,           # 图标与文本间距
    columnspacing=1.0,           # 多列时的列间距
    handlelength=1.5,            # 图标长度
    handleheight=1.2,            # 图标高度
    
    # === 文字样式 ===
    title='Aspects', 
    title_fontsize=20,           # 标题字体大小
    fontsize=18,                 # 条目字体大小
    prop={'weight': 'bold'},     # 加粗字体
    
    # === 背景样式 ===
    facecolor='white',           # 背景色
    framealpha=0.9               # 背景透明度
)
        legend.get_frame().set_linewidth(0.8) 
        for handle in legend.legendHandles:
           handle.set_sizes([80])
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        ax.set_xlim(x_min - 0.1*(x_max-x_min), x_max + 0.15*(x_max-x_min))  # 右侧多留15%空间
        ax.set_ylim(y_min - 0.1*(y_max-y_min), y_max + 0.1*(y_max-y_min))

        plt.tight_layout(pad=2) 
        # 7. 网格线（浅灰色）
        plt.grid(color='lightgray', linestyle='--', alpha=0.5)
        
        # 8. 保存（300dpi高清图）
        os.makedirs('tsne_whitenumber', exist_ok=True)
        plt.savefig(
            f'tsne_whitenumber/epoch_{epoch}.png',
            #bbox_inches='tight',
            dpi=300,                 # 更高分辨率
            facecolor='white'        # 保存时保持白底
        )
        plt.close() 

    def visualize_embeddings(self, model_a, users, epoch, sample_size=500, save_dir="embedding_visualization"):
        """
        可视化三种用户embedding的t-SNE分布（类方法版）
        :param model_a: 辅助域模型
        :param users: 用户ID列表
        :param epoch: 当前epoch数
        :param sample_size: 采样用户数量
        :param save_dir: 保存目录
        """
        with torch.no_grad():
            # 1. 随机采样用户
            sample_users = random.sample(users, min(sample_size, len(users)))
            users_tensor = torch.tensor(sample_users).long().to(device)
            
            # 2. 获取三种embedding
            # 辅助域原始embedding
            au = model_a.get_embed(users_tensor)
            # 迁移后的embedding (使用第一个transfer layer)
            transfer_au = self.transfer_layer[0](au)
            # 目标域embedding
            all_users, _ = self.computer(self.embedding_user.weight, self.embedding_item.weight)
            tu = all_users[users_tensor]
            
            # 3. 转换为numpy数组
            au_embs = au.cpu().numpy()
            transfer_au_embs = transfer_au.cpu().numpy()
            tu_embs = tu.cpu().numpy()
            
            # 4. 合并并创建标签
            all_embs = np.concatenate([au_embs, transfer_au_embs, tu_embs])
            labels = ["Auxiliary"]*len(au_embs) + ["Transferred"]*len(transfer_au_embs) + ["Target"]*len(tu_embs)
            
            # 5. t-SNE降维
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embs_2d = tsne.fit_transform(all_embs)
            
            # 6. 创建可视化
            plt.figure(figsize=(12, 10))
            sns.set_style("whitegrid")
            
            # 使用不同颜色和标记样式
            palette = {"Auxiliary": "#1f77b4", "Transferred": "#ff7f0e", "Target": "#2ca02c"}
            markers = {"Auxiliary": "o", "Transferred": "s", "Target": "D"}
            
            for label in ["Auxiliary", "Transferred", "Target"]:
                mask = np.array(labels) == label
                plt.scatter(embs_2d[mask, 0], embs_2d[mask, 1], 
                        c=palette[label], marker=markers[label],
                        label=label, alpha=0.7, s=60, edgecolor='k', linewidth=0.5)
            
            # 7. 美化图表
            plt.title(f"t-SNE Projection of User Embeddings (Epoch {epoch})", fontsize=14, pad=20)
            plt.xlabel("t-SNE Dimension 1", fontsize=12)
            plt.ylabel("t-SNE Dimension 2", fontsize=12)
            plt.legend(fontsize=12, framealpha=0.9)
            
            # 8. 保存结果
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"), 
                    dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()
            print(f"✅ Saved t-SNE visualization for epoch {epoch} to {save_dir}/epoch_{epoch}.png")
        
    def _plot_aspects(self, users, items, au_embed, epoch):
     with torch.no_grad():
        aspects = [self.transfer_layer[i](au_embed).cpu() for i in range(self.D)]
        all_embeddings = torch.cat([au_embed.cpu()] + aspects)
        
        # 2. 标准化和PCA
        scaler = StandardScaler()
        scaled_emb = scaler.fit_transform(all_embeddings.numpy())
        pca = PCA(n_components=2)
        points = pca.fit_transform(scaled_emb)
        
        # 3. 离群点过滤（统一处理）
        z_scores = stats.zscore(points, axis=0)
        inliers = (np.abs(z_scores) < 2.5).all(axis=1)
        points = points[inliers]
        
        # 4. 动态调整sizes数组
        n_filtered = sum(inliers[:len(users)])  # 实际保留的源域点数
        sizes = np.linspace(20, 100, n_filtered)  # 根据实际点数生成
        
        # 5. 绘制源域（修正关键部分）
        src_points = points[:n_filtered]
        plt.scatter(src_points[:, 0], src_points[:, 1], 
                   s=sizes,  # 现在长度匹配
                   c='#1f77b4', alpha=0.7, label='Source Domain')
        
        # 6. 绘制各维度
        for i in range(self.D):
            start = n_filtered * (i + 1)
            end = start + n_filtered
            dim_points = points[start:end]
            
            plt.scatter(dim_points[:, 0], dim_points[:, 1],
                       s=sizes,  # 使用同样调整后的sizes
                       marker=['o','s','D','^'][i % 4],
                       alpha=0.7,
                       label=f'{self.aspect_names[i]}')
        
        # 5. 专业图表装饰
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.title(f'Aspect Embeddings (Epoch {epoch})\n'
                 f'Total Variance Explained: {sum(pca.explained_variance_ratio_):.1%}')
        plt.grid(alpha=0.2)
        
        # 智能图例位置
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 保存高清图
        plt.savefig(f'aspect_visualization/epoch_{epoch}_enhanced.png', 
                   bbox_inches='tight', transparent=True)
        plt.close()
        #  with torch.no_grad():
        #     # 1. 获取各维度嵌入
        #     aspects = [self.transfer_layer[i](au_embed).cpu() for i in range(3)]
            
        #     # 2. PCA降维
        #     pca = PCA(n_components=2)
        #     embeds = torch.cat([au_embed.cpu()] + aspects)
        #     points = pca.fit_transform(embeds)
            
        #     # 3. 绘制
        #     plt.figure(figsize=(10, 6))
        #     plt.scatter(points[:len(users), 0], points[:len(users), 1], 
        #                label='Source Domain', alpha=0.5)
            
        #     for i in range(3):
        #         start = (i+1)*len(users)
        #         plt.scatter(points[start:start+len(users), 0], 
        #                   points[start:start+len(users), 1],
        #                   label=f'{self.aspect_names[i]}', alpha=0.5)
            
        #     plt.title(f'Aspect Embeddings (Epoch {epoch})')
        #     plt.legend()
        #     plt.savefig(f'aspect_visualization/epoch_{epoch}.png')
        #     plt.close()

        
    def computer(self,user_embed,item_embed):
        all_emb = torch.cat([user_embed, item_embed])
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_titems])
        return users, items

    def attention(self, transfer_embed, i_embed):
        if self.D==1:
            tmp_embed = transfer_embed[0].unsqueeze(1)
        elif self.D==2:
            tmp_embed = torch.stack([transfer_embed[0], transfer_embed[1]], dim=1)
        elif self.D==3:
            tmp_embed = torch.stack([transfer_embed[0], transfer_embed[1], transfer_embed[2]], dim=1)
        elif self.D==4:
            tmp_embed = torch.stack([transfer_embed[0], transfer_embed[1], transfer_embed[2], transfer_embed[3]], dim=1)
        attn = torch.nn.functional.softmax(torch.sum(tmp_embed * i_embed.unsqueeze(1), dim=2), dim=1)
        ans = torch.matmul(attn.unsqueeze(1), tmp_embed).squeeze()
        return ans

    def mapping(self, users, items, au_embed):
        all_users, all_items = self.computer(self.embedding_user.weight,self.embedding_item.weight)
        item_emb = all_items[items]
        tu = all_users[users]

        transfer_embed = []
        for i in range(self.D):
            transfer_embed.append(self.transfer_layer[i](au_embed))
        
        stacked_embed = torch.stack(transfer_embed, dim=1)  # [batch_size, D, embed_dim]
        trans_user_emb, _ = torch.max(stacked_embed, dim=1)
        #trans_user_emb = self.attention(transfer_embed, item_emb)#attentiontihuan
        #trans_user_emb = torch.mean(torch.stack(transfer_embed), dim=0)
        loss = mse_loss(trans_user_emb, tu)
        return loss

    def forward(self, users, items):
        all_users, all_items = self.computer(self.embedding_user.weight,self.embedding_item.weight)
        user_emb = all_users[users]
        item_emb = all_items[items]
        score = torch.sum(user_emb * item_emb, dim=1)
        return user_emb, item_emb, score

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

    def predict(self, users, items, au_embed):
        all_users, all_items = self.computer(self.embedding_user.weight,self.embedding_item.weight)
        item_emb = all_items[items]
        transfer_embed = []
        for i in range(self.D):
            transfer_embed.append(self.transfer_layer[i](au_embed))
        
        stacked_embed = torch.stack(transfer_embed, dim=1)  # [batch_size, D, embed_dim]
        trans_user_emb, _ = torch.max(stacked_embed, dim=1)  # maxpool along aspect dimension

        #trans_user_emb = self.attention(transfer_embed, item_emb)#attentiontihuan
        #trans_user_emb = torch.mean(torch.stack(transfer_embed), dim=0)

        all_users, all_items = self.computer(self.ori_user_embed,self.ori_item_embed)
        t_user_emb = all_users[users]

        user_emb = self.ratio[users].repeat(self.embed_dim, 1).permute(1, 0) * t_user_emb +\
                   (1-self.ratio[users].repeat(self.embed_dim, 1).permute(1, 0))*trans_user_emb

        #user_emb = 0.5 * (t_user_emb + trans_user_emb)
        #user_emb = trans_user_emb
        score = torch.sum(user_emb * item_emb, dim=1)
        return user_emb,item_emb,score

    def transfer_block(self, users, items, au_embed):
        all_users, all_items = self.computer(self.embedding_user.weight,self.embedding_item.weight)
        t_user_emb = all_users[users]
        item_emb = all_items[items]
        transfer_embed = []
        for i in range(self.D):
            transfer_embed.append(self.transfer_layer[i](au_embed))
        
        stacked_embed = torch.stack(transfer_embed, dim=1)  # [batch_size, D, embed_dim]
        trans_user_emb, _ = torch.max(stacked_embed, dim=1)  # maxpool along aspect dimension
        #trans_user_emb = self.attention(transfer_embed, item_emb)#attentiontihuan
        #trans_user_emb = torch.mean(torch.stack(transfer_embed), dim=0) 
        score = torch.sum(trans_user_emb * item_emb, dim=1)
        return trans_user_emb, item_emb, score

    def transfer_bprloss(self, users, pos, neg, au_embed):
        #if users[0] % 20 == 0:
           #self.visualize_aspects(users[:100], pos[:100], au_embed[:100], epoch=self.current_epoch)
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
        all_users, all_items = self.computer(self.embedding_user.weight,self.embedding_item.weight)
        user_emb = all_users[users]
        return user_emb


def parse_opt():
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--batch_size', type=int, default=512, help="b")
    parser.add_argument('--init_lr', type=float, default=0.001, help="b")
    parser.add_argument('--test_num_ng', type=int, default=99, help="b")
    parser.add_argument('--train_num_ng', type=int, default=4, help="b")

    parser.add_argument('--embed_dim', type=int, default=32, help="b")
    parser.add_argument('--K_g', type=int, default=3, help="b")
    parser.add_argument('--mapping', type=int, default=0, help="b")
    parser.add_argument('--ratio', type=int, default=1, help="b")
    parser.add_argument('--K_l', type=int, default=1, help="b")#combine way
    parser.add_argument('--D', type=int, default=3, help="b")#combine way
    args = parser.parse_args()
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

    for epoch in range(50):
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

    mdir = 'pretrain/movie-book/EATN/'
    if not os.path.exists(mdir):
        os.makedirs(mdir, exist_ok=True)
    torch.save(model.state_dict(), mdir+savefname)
    return model

EPOCHS=101
args = parse_opt()
print(args)
train_data = PairwiseDataset('data/movie-book/balance_target_train.csv', args.train_num_ng)
test_data_a_t = TestDataset('data/movie-book/test_a_t.txt')
test_data_a_ts = TestDataset('data/movie-book/test_a_ts.txt')
test_data_as_t = TestDataset('data/movie-book/test_as_t.txt')
test_data_as_ts = TestDataset('data/movie-book/test_as_ts.txt')
best_HR, best_NDCG = np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])
n_test = (len(test_data_a_t)+len(test_data_a_ts)+len(test_data_as_ts)+len(test_data_as_t))/100
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
    model_t.load_state_dict(torch.load('pretrain/movie-book/EATN/model_t.pth.tar'))

model_t.ori_user_embed = model_t.embedding_user.weight
model_t.ori_item_embed = model_t.embedding_item.weight

print('\n================a Domain================')
filename = 'data/movie-book/auxiliary.csv'
savefname = 'model_a.pth.tar'
if train_single:
    model_a = single_train(filename, savefname)
else:
    dataset = PairwiseDataset(filename, args.train_num_ng)
    model_a = CDATN(dataset, **vars(args))
    model_a = model_a.to(device)
    model_a.load_state_dict(torch.load('pretrain/movie-book/EATN/model_a.pth.tar'))

for name, param in model_a.named_parameters():
    param.requires_grad = False


print('\n================train================')
overlap_users = overlap_user('data/movie-book/overlap_users.txt')
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model_t.parameters(), lr=0.001)
batchSize = 512

print("Total number of param in {} is {}".format('CDATN', sum(x.numel() for x in model_t.parameters())))
batch_time = 0
for epoch in range(EPOCHS):
    
    print("=" * 20 + "Epoch ", epoch, "=" * 20)
    dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    model_t.train()
    dataloader.dataset.ng_sample()
    loss, n = 0.0, 0
    start_time = time()
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
    batch_time +=(time()-start_time)/n
    # if epoch % 20 == 0:
    #     temp_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    #     for batch in temp_loader:  # 只取第一个batch
    #         users, items, _ = batch
    #         users = users.to(device)
    #         items = items.to(device)
    #         au_embed = model_a.get_embed(users)
    #         model_t._plot_tsne(users, items, au_embed, epoch)
    #         break
    # if epoch % 20 == 0 or epoch == EPOCHS - 1:
    #     model_t.visualize_embeddings(
    #         model_a=model_a,
    #         users=overlap_users,
    #         epoch=epoch,
    #         sample_size=500,
    #         save_dir="embedding_visualization"
    #     )
    if epoch % 5 == 0:
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
                _,_,rating = model_t.predict(user, item, model_a.get_embed(user))
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
print('batch_time {}'.format(batch_time/EPOCHS))
