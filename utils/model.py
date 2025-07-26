import torch
import torch.nn as nn


class PureMF(nn.Module):
    def __init__(self,config,loader):
        super(PureMF, self).__init__()
        self.n_user = loader.n_users
        self.n_item = loader.n_items
        self.embed_dim = config.PureMF_dim
        self.U_emb = nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.embed_dim)
        self.V_emb = nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.embed_dim)
        self.f = nn.Sigmoid()

    def get_embed(self, u):
        return self.U_emb(u)

    def get_rating(self, user):
        item = self.V_emb.weight.t()
        user = self.U_emb(user)
        score = torch.matmul(user, item)
        score = self.f(score)
        return score[:,self.n_sitems:]

    def forward(self, u,i):
        user = self.U_emb(u)
        item = self.V_emb(i)
        score = torch.sum(user * item, dim=1)
        return score

    def bceloss(self, users, items, labels):
        prediction = self.forward(users, items)
        loss = torch.nn.BCEWithLogitsLoss()(prediction, labels.float())
        return loss

class MF(nn.Module):
    def __init__(self, n_user, n_item, dim):
        super(MF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.U_emb = nn.Embedding(num_embeddings=n_user, embedding_dim=dim)
        self.V_emb = nn.Embedding(num_embeddings=n_item, embedding_dim=dim)
        self.f = nn.Sigmoid()

    def get_embed(self, u):
        return self.U_emb(u)

    def get_item_embed(self, i):
        return self.V_emb(i)

    def get_rating(self, user,item):
        item = self.V_emb(item)
        score = torch.sum(user * item, dim=1)
        #score = torch.matmul(user, item)
        return score

    def forward(self, u,i):
        user = self.U_emb(u)
        item = self.V_emb(i)
        score = torch.sum(user * item, dim=1)
        return score

    def bceloss(self, users, items, labels):
        prediction = self.forward(users, items)
        loss = torch.nn.BCEWithLogitsLoss()(prediction, labels.float())
        return loss

class MLP(nn.Module):
    def __init__(self, dim, layers):
        super(MLP, self).__init__()
        self.hidden = dim // 2
        self.layers = layers - 1
        MLP_modules = [nn.Linear(dim, self.hidden),nn.ReLU()]
        for i in range(self.layers):
            MLP_modules.append(nn.Linear(self.hidden, self.hidden))
            MLP_modules.append(nn.ReLU())
        MLP_modules.append(nn.Linear(self.hidden, dim))
        self.MLP_layers = nn.Sequential(*MLP_modules)

    def forward(self, u):
        x = self.MLP_layers(u)
        return x

class Multi_MLP(nn.Module):
    def __init__(self, dim, layers):
        super(Multi_MLP, self).__init__()
        self.hidden = dim // 2
        self.layers = layers - 1
        MLP1_modules = [nn.Linear(dim, self.hidden),nn.ReLU()]
        for i in range(self.layers):
            MLP1_modules.append(nn.Linear(self.hidden, self.hidden))
            MLP1_modules.append(nn.ReLU())
        MLP1_modules.append(nn.Linear(self.hidden, dim))
        self.MLP1_layers = nn.Sequential(*MLP1_modules)

        MLP2_modules = [nn.Linear(dim, self.hidden),nn.ReLU()]
        for i in range(self.layers):
            MLP2_modules.append(nn.Linear(self.hidden, self.hidden))
            MLP2_modules.append(nn.ReLU())
        MLP2_modules.append(nn.Linear(self.hidden, dim))
        self.MLP2_layers = nn.Sequential(*MLP2_modules)

        MLP3_modules = [nn.Linear(dim, self.hidden),nn.ReLU()]
        for i in range(self.layers):
            MLP3_modules.append(nn.Linear(self.hidden, self.hidden))
            MLP3_modules.append(nn.ReLU())
        MLP3_modules.append(nn.Linear(self.hidden, dim))
        self.MLP3_layers = nn.Sequential(*MLP3_modules)

    def forward(self, u):
        x1 = self.MLP1_layers(u)
        x2 = self.MLP2_layers(u)
        x3 = self.MLP3_layers(u)
        return 1/3*(x1+x2+x3)

class EATN(nn.Module):
    def __init__(self, config,loader,mf_s_model = None,mf_t_model=None,mapping_model = None):
        super(EATN, self).__init__()
        self.preweight = config.preweight
        if self.preweight:
            self.mf_t_model = mf_t_model
            self.mf_s_model = mf_s_model
            self.MLP_model = mapping_model
        self.__parameters__(loader,config)
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

    def __parameters__(self,loader,config):
        self.n_user = loader.n_users
        self.n_sitem = loader.n_sitems
        self.n_titem = loader.n_titems
        self.embed_dim = config.MATN_embed_dim
        self.layers = config.layers - 1
        self.hidden = self.embed_dim // 2

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


class NCF(nn.Module):
    def __init__(self,config,loader):
        super(NCF, self).__init__()
        self.n_user = loader.n_users
        self.n_item = loader.n_items
        self.embed_dim = config.NCF_dim
        self.num_layers = config.NCF_layers
        self.U_emb = nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.embed_dim)
        self.V_emb = nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.embed_dim)

        self.embed_user_GMF = nn.Embedding(self.n_user, self.embed_dim)
        self.embed_item_GMF = nn.Embedding(self.n_item, self.embed_dim)
        self.embed_user_MLP = nn.Embedding(
            self.n_user, self.embed_dim * (2 ** (self.num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(
            self.n_item, self.embed_dim * (2 ** (self.num_layers - 1)))

        MLP_modules = []
        for i in range(self.num_layers):
            input_size = self.embed_dim * (2 ** (self.num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
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

class ETN(nn.Module):
    def __init__(self, config,loader,mf_s_model = None,mf_t_model=None,mapping_model = None):
        super(ETN, self).__init__()
        self.preweight = config.preweight
        if self.preweight:
            self.mf_t_model = mf_t_model
            self.mf_s_model = mf_s_model
            self.MLP_model = mapping_model
        self.__parameters__(loader,config)
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

    def __parameters__(self,loader,config):
        self.n_user = loader.n_users
        self.n_sitem = loader.n_sitems
        self.n_titem = loader.n_titems
        self.embed_dim = config.BTN_embed_dim
        self.layers = config.layers - 1
        self.hidden = self.embed_dim // 2

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

    def mlp(self,user):
        x = self.MLP1_layers(user)
        return x

    def mapping(self,user):
        su_embed,tu_embed = self.su_emb(user), self.tu_emb(user)
        fu_embed = self.mlp(su_embed)
        loss = self.mse_loss(fu_embed, tu_embed)
        return loss

    def prediction(self,user,item):
        u_embed = self.su_emb(user)
        i_embed = self.ti_emb(item)
        fu_embed = self.mlp(u_embed)
        score = torch.sum(fu_embed * i_embed, dim=1)
        return score

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

class MTN(nn.Module):
    def __init__(self, config,loader,mf_s_model = None,mf_t_model=None,mapping_model = None):
        super(MTN, self).__init__()
        self.preweight = config.preweight
        if self.preweight:
            self.mf_t_model = mf_t_model
            self.mf_s_model = mf_s_model
            self.MLP_model = mapping_model
        self.__parameters__(loader,config)
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

    def __parameters__(self,loader,config):
        self.n_user = loader.n_users
        self.n_sitem = loader.n_sitems
        self.n_titem = loader.n_titems
        self.embed_dim = config.MTN_embed_dim
        self.layers = config.layers - 1
        self.hidden = self.embed_dim // 2

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


    def mlp(self,user):
        x1 = self.MLP1_layers(user)
        x2 = self.MLP2_layers(user)
        x3 = self.MLP3_layers(user)
        x = 1/3*(x1+x2+x3)
        return x

    def mapping(self,user,item):
        su_embed,tu_embed = self.su_emb(user), self.tu_emb(user)
        fu_embed = self.mlp(su_embed)
        loss = self.mse_loss(fu_embed, tu_embed)
        return loss


    def prediction(self,user,item):
        u_embed = self.su_emb(user)
        i_embed = self.ti_emb(item)
        fu_embed = self.mlp(u_embed)
        score = torch.sum(fu_embed * i_embed, dim=1)
        return score

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


class ATN(nn.Module):
    def __init__(self, dim, layers):
        super(ATN, self).__init__()
        self.hidden = dim // 2
        self.layers = layers - 1
        MLP1_modules = [nn.Linear(dim, self.hidden),nn.ReLU()]
        for i in range(self.layers):
            MLP1_modules.append(nn.Linear(self.hidden, self.hidden))
            MLP1_modules.append(nn.ReLU())
        MLP1_modules.append(nn.Linear(self.hidden, dim))
        self.MLP1_layers = nn.Sequential(*MLP1_modules)

        MLP2_modules = [nn.Linear(dim, self.hidden),nn.ReLU()]
        for i in range(self.layers):
            MLP2_modules.append(nn.Linear(self.hidden, self.hidden))
            MLP2_modules.append(nn.ReLU())
        MLP2_modules.append(nn.Linear(self.hidden, dim))
        self.MLP2_layers = nn.Sequential(*MLP2_modules)

        MLP3_modules = [nn.Linear(dim, self.hidden),nn.ReLU()]
        for i in range(self.layers):
            MLP3_modules.append(nn.Linear(self.hidden, self.hidden))
            MLP3_modules.append(nn.ReLU())
        MLP3_modules.append(nn.Linear(self.hidden, dim))
        self.MLP3_layers = nn.Sequential(*MLP3_modules)
        self.mse_loss = torch.nn.MSELoss()

    def mapping(self, su_embed, i_embed):
        x1 = self.MLP1_layers(su_embed)
        x2 = self.MLP2_layers(su_embed)
        x3 = self.MLP3_layers(su_embed)
        fu_embed = self.attention(x1, x2, x3, i_embed)
        return fu_embed

    def attention(self,x1,x2,x3,i_embed):
        x = torch.stack([x1, x2, x3], dim=1)
        attn = torch.nn.functional.softmax(torch.sum(x * i_embed.unsqueeze(1), dim=2), dim=1)
        ans = torch.matmul(attn.unsqueeze(1), x).squeeze()
        return ans

    def forward(self, su_embed, tu_embed, i_embed):
        x1 = self.MLP1_layers(su_embed)
        x2 = self.MLP2_layers(su_embed)
        x3 = self.MLP3_layers(su_embed)
        fu_embed = self.attention(x1, x2, x3, i_embed)
        loss = self.mse_loss(fu_embed, tu_embed)
        return loss