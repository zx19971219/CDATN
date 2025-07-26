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

class TMF(nn.Module):
    def __init__(self, config,loader):
        super(TMF, self).__init__()
        self.__parameters__(loader,config)
        self.__layers__()
        self.__init_weight__()

    def __parameters__(self,loader,config):
        self.n_user = loader.n_users
        self.n_sitem = loader.n_sitems
        self.n_titem = loader.n_titems
        self.MF_dim = config.TMF_dim

        self.MLP_dim = config.TMF_MLP_dim
        self.hidden = config.TMF_MLP_dim // 2
        self.layers = config.layers - 1

    def __layers__(self):
        self.su_emb = nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.MF_dim)
        self.tu_emb = nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.MF_dim)
        self.si_emb = nn.Embedding(num_embeddings=self.n_sitem, embedding_dim=self.MF_dim)
        self.ti_emb = nn.Embedding(num_embeddings=self.n_titem, embedding_dim=self.MF_dim)
        self.f = nn.Sigmoid()
        self.net_in = nn.Sequential(
            nn.Linear(self.MLP_dim, self.hidden),
            nn.ReLU(),
        )
        self.net_hid = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )
        self.net_out = nn.Sequential(
            nn.Linear(self.hidden, self.MLP_dim)
        )
        self.mse_loss = nn.MSELoss()

    def get_embed(self, u):
        return self.su_emb(u), self.tu_emb(u)

    def get_rating(self, user):
        item = self.V_emb.weight.t()
        score = torch.matmul(user, item)
        return self.f(score)

    def forward(self, suser,sitem,tuser,titem,muser):
        suser = self.su_emb(suser)
        tuser = self.tu_emb(tuser)
        sitem = self.si_emb(sitem)
        titem = self.ti_emb(titem)
        sscore = torch.sum(suser * sitem, dim=1)
        tscore = torch.sum(tuser * titem, dim=1)

        su_embed,tuembed = self.get_embed(muser)
        x = self.net_in(su_embed)
        for _ in range(self.layers):
            x = self.net_hid(x)
        f_su_embed = self.net_out(x)
        return sscore,tscore,f_su_embed,tuembed

    def bceloss(self,  suser,sitem,slabel, tuser,titem,tlabel,muser):
        sscore,tscore,f_su_embed,tuembed = self.forward(suser,sitem,tuser,titem,muser)
        s_bceloss = torch.nn.BCEWithLogitsLoss()(sscore, slabel.float())
        t_bceloss = torch.nn.BCEWithLogitsLoss()(tscore, tlabel.float())
        mapping_mseloss = torch.nn.MSELoss(f_su_embed, tuembed)
        loss = s_bceloss + t_bceloss + mapping_mseloss
        return loss

class EMCDRETE(nn.Module):
    def __init__(self, config,loader,mf_s_model = None,mf_t_model=None,mapping_model = None):
        super(EMCDRETE, self).__init__()
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

            '''
            # mlp layers
            for (m1, m2) in zip(
                    self.net_in, self.mapping_model.net_in):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)
            for (m1, m2) in zip(
                    self.net_hid, self.mapping_model.net_hid):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)
            for (m1, m2) in zip(
                    self.net_out, self.mapping_model.net_out):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)
            '''

    def __parameters__(self,loader,config):
        self.n_user = loader.n_users
        self.n_sitem = loader.n_sitems
        self.n_titem = loader.n_titems
        self.MF_dim = config.TMF_dim

        self.MLP_dim = config.TMF_MLP_dim
        self.hidden = config.TMF_MLP_dim // 2
        self.layers = config.layers - 1

    def __layers__(self):
        self.su_emb = nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.MF_dim)
        self.tu_emb = nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.MF_dim)
        self.si_emb = nn.Embedding(num_embeddings=self.n_sitem, embedding_dim=self.MF_dim)
        self.ti_emb = nn.Embedding(num_embeddings=self.n_titem, embedding_dim=self.MF_dim)
        self.f = nn.Sigmoid()
        self.mse_loss = torch.nn.MSELoss()
        MLP1_modules = [nn.Linear(self.MLP_dim, self.hidden),nn.ReLU()]
        for i in range(self.layers):
            MLP1_modules.append(nn.Linear(self.hidden, self.hidden))
            MLP1_modules.append(nn.ReLU())
        MLP1_modules.append(nn.Linear(self.hidden, self.MLP_dim))
        self.MLP1_layers = nn.Sequential(*MLP1_modules)

        MLP2_modules = [nn.Linear(self.MLP_dim, self.hidden),nn.ReLU()]
        for i in range(self.layers):
            MLP2_modules.append(nn.Linear(self.hidden, self.hidden))
            MLP2_modules.append(nn.ReLU())
        MLP2_modules.append(nn.Linear(self.hidden, self.MLP_dim))
        self.MLP2_layers = nn.Sequential(*MLP2_modules)

        MLP3_modules = [nn.Linear(self.MLP_dim, self.hidden),nn.ReLU()]
        for i in range(self.layers):
            MLP3_modules.append(nn.Linear(self.hidden, self.hidden))
            MLP3_modules.append(nn.ReLU())
        MLP3_modules.append(nn.Linear(self.hidden, self.MLP_dim))
        self.MLP3_layers = nn.Sequential(*MLP3_modules)


    def get_embed(self, u):
        return self.su_emb(u), self.tu_emb(u)

    def mlp(self,u):
        x = self.MLP1_layers(u)
        return x

    def multi_mlp(self,u):
        x1 = self.MLP1_layers(u)
        x2 = self.MLP2_layers(u)
        x3 = self.MLP3_layers(u)
        return 1/3*(x1+x2+x3)

    def mapping(self,u):
        us,ut = self.get_embed(u)
        fu = self.multi_mlp(us)
        loss = self.mse_loss(fu, ut)
        return loss

    def prediction(self,user,item):
        user = self.su_emb(user)
        fuser = self.multi_mlp(user)
        item = self.ti_emb(item)
        score = torch.sum(fuser * item, dim=1)
        return score

    def get_rating(self, user):
        item = self.V_emb.weight.t()
        score = torch.matmul(user, item)
        return self.f(score)


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

    def forward(self):
        return

    def bceloss(self,user,item,label):
        score = self.prediction(user,item)
        loss = torch.nn.BCEWithLogitsLoss()(score, label.float())
        return loss

class MATN(nn.Module):
    def __init__(self, config,loader,mf_s_model = None,mf_t_model=None,mapping_model = None):
        super(MATN, self).__init__()
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


    def mlp(self,u):
        x = self.MLP1_layers(u)
        return x

    def multi_mlp(self,user):
        x1 = self.MLP1_layers(user)
        x2 = self.MLP2_layers(user)
        x3 = self.MLP3_layers(user)
        return x1,x2,x3

    def mapping(self,user):
        su_embed,tu_embed = self.su_emb(user), self.tu_emb(user)
        fu_embed = self.mlp(su_embed)
        loss = self.mse_loss(fu_embed, tu_embed)
        return loss

    def attention(self,x1,x2,x3,i_embed):
        x = torch.stack([x1, x2, x3], dim=1)
        attn = torch.nn.functional.softmax(torch.sum(x * i_embed.unsqueeze(1), dim=2), dim=1)
        ans = torch.matmul(attn.unsqueeze(1), x).squeeze()
        return ans

    def prediction(self,user,item):
        u_embed = self.su_emb(user)
        i_embed = self.ti_emb(user)
        x1,x2,x3 = self.multi_mlp(u_embed)
        fu_embed = self.attention(x1,x2,x3,i_embed)
        i_embed = self.ti_emb(item)
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
