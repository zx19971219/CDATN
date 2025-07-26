from config import config
import dataloader
from torch.utils.data import DataLoader
import torch
from time import time
from model import PureMF,MF,MLP,TMF,EMCDRETE,Multi_MLP,MATN
import evaluate
import numpy as np
import os

loader = dataloader.Loader(config)
loader.load_data('one')
if config.model == 'EMCDRETE':
    TTrainData = dataloader.CDPointwiseDataset(loader, loader.train_tdata, 't',config.train_num_ng)
    TTrainLoader = DataLoader(TTrainData, batch_size=config.LF_batchsize, shuffle=True)
    STrainData = dataloader.CDPointwiseDataset(loader, loader.train_sdata, 's', config.train_num_ng)
    STrainLoader = DataLoader(STrainData, batch_size=config.LF_batchsize, shuffle=True)
    OverLoader = DataLoader(loader.overlap_users, batch_size=config.LS_batchsize, shuffle=True)
    TestData = dataloader.PointwiseDataset(loader, loader.test_negetive, 0, False)
    TestLoader = DataLoader(TestData, batch_size=config.test_num_ng + 1, shuffle=False)

    if config.preweight == True:
        assert os.path.exists(config.MF_S_path), 'lack of MF_S model'
        assert os.path.exists(config.MF_T_path), 'lack of MF_T model'
        assert os.path.exists(config.Mapping_path), 'lack of MF_T model'
        mf_s_model = torch.load(config.MF_S_path)
        mf_t_model = torch.load(config.MF_T_path)
        mapping_model = torch.load(config.Mapping_path)
        model = EMCDRETE(config, loader, mf_s_model, mf_t_model, mapping_model)
    else:
        model = EMCDRETE(config, loader)

    n_users = loader.n_users
    n_titems = loader.n_titems
    n_sitems = loader.n_sitems
    model = model.to(config.device)
    opt = torch.optim.Adam(model.parameters(), lr=config.LF_lr)

    for epoch in range(config.TMF_epochs):
        start_train_time = time()
        model.train()
        start_time = time()
        #print('ng sample start!')
        TTrainLoader.dataset.ng_sample()
        #print('ng sample over!')
        loss = 0.0
        temp_n = 0
        for batch in TTrainLoader:
            opt.zero_grad()
            users, items, labels = map(lambda x: x.cuda(), batch)
            batch_loss = model.mf_t(users, items, labels)

            batch_loss.backward()
            opt.step()
            loss += batch_loss
            temp_n += 1
        loss /= temp_n
        print('MF_T: Epoch %d train==[%.5f]' % (epoch, loss))

        STrainLoader.dataset.ng_sample()
        loss = 0.0
        temp_n = 0
        for batch in STrainLoader:
            opt.zero_grad()
            users, items, labels = map(lambda x: x.cuda(), batch)
            batch_loss = model.mf_s(users, items, labels)

            batch_loss.backward()
            opt.step()
            loss += batch_loss
            temp_n += 1
        loss /= temp_n
        print('MF_S: Epoch %d train==[%.5f]' % (epoch, loss))


        loss = 0
        temp_n = 0

        for users in OverLoader:
            users = users.cuda()
            batch_loss = model.mapping(users)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            loss += batch_loss
            temp_n += 1
        loss /= temp_n
        print('Mapping: Epoch %d train==[%.5f]' % (epoch, loss))

        for batch in TTrainLoader:
            users, items, labels = map(lambda x: x.cuda(), batch)
            batch_loss = model.bceloss(users, items, labels)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            loss += batch_loss
            temp_n += 1
        loss /= temp_n

        print('Score: Epoch %d train==[%.5f]' % (epoch, loss))

        #if epoch % 10 == 0:
        print('================test===============')
        #    with torch.no_grad():
        #        HR, NDCG = evaluate.EMCDRETE_metrics(model, TestLoader, config.topk)

        #elapsed_time = time() - start_time
        HR, NDCG = evaluate.EMCDRETE_metrics(model, TestLoader, config.topk)
        #print("The time elapse of epoch {:03d}".format(epoch) + " is: " + '{}'.format(elapsed_time))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

if config.model == 'EMCDR':
    print('\n================t Domain MF================')
    TTrainData = dataloader.CDPointwiseDataset(loader, loader.train_tdata, 't',config.train_num_ng)
    TTrainLoader = DataLoader(TTrainData, batch_size=config.LF_batchsize, shuffle=True)
    n_users = loader.n_users
    n_items = loader.n_titems
    mf_t = MF(n_users, n_items, config.LF_dim)
    mf_t = mf_t.to(config.device)
    opt = torch.optim.Adam(mf_t.parameters(), lr=config.LF_lr)

    for epoch in range(config.LF_epochs):
        start_train_time = time()
        mf_t.train()
        start_time = time()
        #print('ng sample start!')
        TTrainLoader.dataset.ng_sample()
        #print('ng sample over!')
        loss = 0.0
        temp_n = 0
        for batch in TTrainLoader:
            mf_t.zero_grad()
            users, items, labels = map(lambda x: x, batch)
            batch_loss = mf_t.bceloss(users.cuda(), items.cuda(), labels.cuda())

            batch_loss.backward()
            opt.step()
            loss += batch_loss
            temp_n += 1
        loss /= temp_n

        #if (epoch + 1) % 10 != 0:
        #    continue
        #print('================================Epoch %d===================================' % (epoch))
        print('Epoch %d train==[%.5f]' % (epoch, loss))

    print('\n================s Domain MF================')
    STrainData = dataloader.CDPointwiseDataset(loader, loader.train_sdata, 's',config.train_num_ng)
    STrainLoader = DataLoader(STrainData, batch_size=config.LF_batchsize, shuffle=True)
    n_users = loader.n_users
    n_items = loader.n_sitems
    mf_s = MF(n_users, n_items, config.LF_dim)
    mf_s = mf_s.to(config.device)
    opt = torch.optim.Adam(mf_s.parameters(), lr=config.LF_lr,)

    for epoch in range(config.LF_epochs):
        start_train_time = time()
        mf_s.train()
        start_time = time()
        #print('ng sample start!')
        STrainLoader.dataset.ng_sample()
        #print('ng sample over!')
        loss = 0.0
        temp_n = 0
        for batch in STrainLoader:
            mf_s.zero_grad()
            users, items, labels = map(lambda x: x, batch)
            batch_loss = mf_s.bceloss(users.cuda(), items.cuda(), labels.cuda())

            batch_loss.backward()
            opt.step()
            loss += batch_loss
            temp_n += 1
        loss /= temp_n

        #if (epoch + 1) % 10 != 0:
        #    continue
        #print('================================Epoch %d===================================' % (epoch))
        print('Epoch %d train==[%.5f]' % (epoch, loss))


    print('\n================mapping================')
    OverLoader = DataLoader(loader.overlap_users, batch_size=config.LS_batchsize, shuffle=True)
    if config.LS_model == 'MLP':
        mapping = MLP(config.LS_dim, config.LS_layers)
    if config.LS_model == 'Multi_MLP':
        mapping = Multi_MLP(config.LS_dim, config.LS_layers)
    mapping = mapping.to(config.device)
    opt = torch.optim.Adam(mapping.parameters(), lr=config.LS_lr)
    mse_loss = torch.nn.MSELoss()

    for epoch in range(config.LS_epochs):
        loss_sum = 0
        temp_n = 0
        '''
        for users in OverLoader:
            users = users.cuda()
            u = mf_s.get_embed(users)
            y = mf_t.get_embed(users)
            out = mapping(u)
            loss = mse_loss(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss
            temp_n += 1
        loss_sum /= temp_n
        print('Epoch %d train==[%.5f]' % (epoch, loss_sum))
        '''
        for batch in TTrainLoader:
            users, items, labels = map(lambda x: x.cuda(), batch)
            users = users.cuda()
            u = mf_s.get_embed(users)
            y = mf_t.get_embed(users)
            out = mapping(u)
            prediction = torch.sum(out * mf_t.get_item_embed(items), dim=1)
            batch_loss = torch.nn.BCEWithLogitsLoss()(prediction, labels.float())
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            loss_sum += batch_loss
            temp_n += 1
        loss_sum /= temp_n
        print('Epoch %d train==[%.5f]' % (epoch, loss_sum))

    TestData = dataloader.PointwiseDataset(loader, loader.test_negetive, 0, False)
    TestLoader = DataLoader(TestData, batch_size=config.test_num_ng+1, shuffle=False)
    with torch.no_grad():
        HR, NDCG = evaluate.EMCDR_metrics(mf_s, mf_t, mapping, TestLoader, config.topk)
        if config.out:
            if not os.path.exists(config.model_path):
                os.mkdir(config.model_path)
            torch.save(mf_s,'{}{}.pth'.format(config.model_path,'mf_s'))
            torch.save(mf_t,'{}{}.pth'.format(config.model_path,'mf_t'))
            torch.save(mapping,'{}{}.pth'.format(config.model_path,'mapping'))

        elapsed_time = time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + '{}'.format(elapsed_time))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

if config.model == 'TMF':

    print('\n================t Domain MF================')
    TrainData = dataloader.CDModePointwiseDataset(loader, loader.train_data_mode, config.train_num_ng)
    TrainLoader = DataLoader(TrainData, batch_size=config.TMF_batchsize, shuffle=True)
    n_users = loader.n_users
    n_titems = loader.n_titems
    n_sitems = loader.n_sitems
    mf = TMF(config,loader)
    mf = mf.to(config.device)
    opt = torch.optim.Adam(mf.parameters(), lr=config.LF_lr)

    for epoch in range(config.TMF_epochs):
        start_train_time = time()
        mf.train()
        start_time = time()
        # print('ng sample start!')
        TrainLoader.dataset.ng_sample()
        # print('ng sample over!')
        loss = 0.0
        temp_n = 0
        for batch in TrainLoader:
            users, items, modes,labels = map(lambda x: x, batch)
            batch_loss = mf.bceloss(users.cuda(), items.cuda(), labels.cuda())

            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            loss += batch_loss
            temp_n += 1
        loss /= temp_n

        # if (epoch + 1) % 10 != 0:
        #    continue
        # print('================================Epoch %d===================================' % (epoch))
        print('Epoch %d train==[%.5f]' % (epoch, loss))

    print('\n================s Domain MF================')
    STrainData = dataloader.CDPointwiseDataset(loader, loader.train_sdata, 's', config.train_num_ng)
    STrainLoader = DataLoader(STrainData, batch_size=config.LF_batchsize, shuffle=True)
    n_users = loader.n_users
    n_items = loader.n_sitems
    mf_s = MF(n_users, n_items, config.LF_dim)
    mf_s = mf_s.to(config.device)
    opt = torch.optim.Adam(mf_s.parameters(), lr=config.LF_lr, )

    for epoch in range(config.LF_epochs):
        start_train_time = time()
        mf_s.train()
        start_time = time()
        # print('ng sample start!')
        STrainLoader.dataset.ng_sample()
        # print('ng sample over!')
        loss = 0.0
        temp_n = 0
        for batch in STrainLoader:
            mf_s.zero_grad()
            users, items, labels = map(lambda x: x, batch)
            batch_loss = mf_s.bceloss(users.cuda(), items.cuda(), labels.cuda())

            batch_loss.backward()
            opt.step()
            loss += batch_loss
            temp_n += 1
        loss /= temp_n

        # if (epoch + 1) % 10 != 0:
        #    continue
        # print('================================Epoch %d===================================' % (epoch))
        print('Epoch %d train==[%.5f]' % (epoch, loss))

    print('\n================mapping================')
    OverLoader = DataLoader(loader.overlap_users, batch_size=config.LS_batchsize, shuffle=True)
    if config.LS_model == 'MLP':
        mapping = MLP(config.LS_dim, config.LS_layers)
    mapping = mapping.to(config.device)
    opt = torch.optim.Adam(mapping.parameters(), lr=config.LS_lr)
    mse_loss = torch.nn.MSELoss()

    for epoch in range(config.LS_epochs):
        loss_sum = 0
        temp_n = 0
        for users in OverLoader:
            users = users.cuda()
            u = mf_s.get_embed(users)
            y = mf_t.get_embed(users)
            out = mapping(u)
            loss = mse_loss(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss
            temp_n += 1
        loss_sum /= temp_n
        print('Epoch %d train==[%.5f]' % (epoch, loss_sum))

    TestData = dataloader.PointwiseDataset(loader, loader.test_negetive, 0, False)
    TestLoader = DataLoader(TestData, batch_size=config.test_num_ng + 1, shuffle=False)
    with torch.no_grad():
        HR, NDCG = evaluate.EMCDR_metrics(mf_s, mf_t, mapping, TestLoader, config.topk)

        elapsed_time = time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + '{}'.format(elapsed_time))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

if config.model == 'MF':
    print('Choose Model: PureMF!')
    TrainData = dataloader.PointwiseDataset(loader,loader.train_data,config.train_num_ng,True)
    TrainLoader = DataLoader(TrainData, batch_size=config.PureMF_batch_size, shuffle=True)

    TestData = dataloader.PointwiseDataset(loader, loader.test_negetive, 0, False)
    TestLoader = DataLoader(TestData, batch_size=config.test_num_ng+1, shuffle=False)
    model = PureMF(config,loader).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=config.PureMF_lr)
    start = time()
    count, best_hr = 0, 0
    for epoch in range(config.EPOCHS):
        start_train_time = time()
        model.train()
        start_time = time()
        #print('ng sample start!')
        TrainLoader.dataset.ng_sample()
        #print('ng sample over!')
        loss = 0.0
        temp_n = 0
        for batch in TrainLoader:
            model.zero_grad()
            users, items, labels = map(lambda x: x, batch)
            batch_loss = model.bceloss(users.cuda(), items.cuda(), labels.cuda())

            batch_loss.backward()
            optim.step()
            loss += batch_loss
            temp_n += 1
        loss /= temp_n

        #if (epoch + 1) % 10 != 0:
        #    continue
        #print('================================Epoch %d===================================' % (epoch))
        print('Epoch %d train==[%.5f]' % (epoch, loss))

        model.eval()
        HR, NDCG = evaluate.metrics(model, TestLoader, config.topk)

        elapsed_time = time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +'{}'.format(elapsed_time))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch

if config.model == 'EMCDRETE-batch':
    TTrainData = dataloader.CDPointwiseDataset(loader, loader.train_tdata, 't',config.train_num_ng)
    TTrainLoader = DataLoader(TTrainData, batch_size=loader.train_sdata_batchsize, shuffle=True)
    STrainData = dataloader.CDPointwiseDataset(loader, loader.train_sdata, 's', config.train_num_ng)
    STrainLoader = DataLoader(STrainData, batch_size=loader.train_tdata_batchsize, shuffle=True)
    OverTrainData = dataloader.CDPointwiseDataset(loader, loader.train_overlapdata, 't',config.train_num_ng)
    OverTrainLoader = DataLoader(OverTrainData, batch_size=loader.train_overdata_batchsize, shuffle=True)
    # OverLoader = DataLoader(loader.overlap_users, batch_size=config.LS_batchsize, shuffle=True)
    TestData = dataloader.PointwiseDataset(loader, loader.test_negetive, 0, False)
    TestLoader = DataLoader(TestData, batch_size=config.test_num_ng + 1, shuffle=False)

    if config.preweight == True:
        assert os.path.exists(config.MF_S_path), 'lack of MF_S model'
        assert os.path.exists(config.MF_T_path), 'lack of MF_T model'
        assert os.path.exists(config.Mapping_path), 'lack of MF_T model'
        mf_s_model = torch.load(config.MF_S_path)
        mf_t_model = torch.load(config.MF_T_path)
        mapping_model = torch.load(config.Mapping_path)
        model = EMCDRETE(config, loader, mf_s_model, mf_t_model, mapping_model)
    else:
        model = EMCDRETE(config, loader)

    n_users = loader.n_users
    n_titems = loader.n_titems
    n_sitems = loader.n_sitems
    model = model.to(config.device)
    opt = torch.optim.Adam(model.parameters(), lr=config.LF_lr)

    for epoch in range(config.TMF_epochs):
        start_train_time = time()
        model.train()
        start_time = time()
        #print('ng sample start!')
        TTrainLoader.dataset.ng_sample()
        STrainLoader.dataset.ng_sample()
        OverTrainLoader.dataset.ng_sample()
        lossS,lossT,lossMapping,lossScore = 0.0, 0.0, 0.0, 0.0
        temp_n = 0
        #print('ng sample over!')
        for i,(batchT, batchS, batchOver) in enumerate(zip(TTrainLoader,STrainLoader,OverTrainLoader)):
            users, items, labels = map(lambda x: x.cuda(), batchT)
            batch_loss = model.mf_t(users, items, labels)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            lossS += batch_loss
            temp_n += 1

            users, items, labels = map(lambda x: x.cuda(), batchS)
            batch_loss = model.mf_s(users, items, labels)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            lossT += batch_loss

            users, items, labels = map(lambda x: x.cuda(), batchOver)
            users = users.cuda()

            batch_loss = model.mapping(users)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            lossMapping += batch_loss

            batch_loss = model.bceloss(users, items, labels)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            lossScore += batch_loss

        lossS /= temp_n
        lossT /= temp_n
        lossMapping /= temp_n
        lossScore += temp_n

        print('MF_S: Epoch %d train==[%.5f]' % (epoch, lossS))
        print('MF_T: Epoch %d train==[%.5f]' % (epoch, lossT))
        print('Mapping: Epoch %d train==[%.5f]' % (epoch, lossMapping))
        print('Score: Epoch %d train==[%.5f]' % (epoch, lossScore))

        if epoch % 10 == 0:
            print('================test===============')
            with torch.no_grad():
                HR, NDCG = evaluate.EMCDRETE_metrics(model, TestLoader, config.topk)

        #elapsed_time = time() - start_time
        HR, NDCG = evaluate.EMCDRETE_metrics(model, TestLoader, config.topk)
        #print("The time elapse of epoch {:03d}".format(epoch) + " is: " + '{}'.format(elapsed_time))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
if config.model == 'MATN':
    print('Choose MATN!')
    TTrainData = dataloader.CDPointwiseDataset(loader, loader.train_tdata, 't',config.train_num_ng)
    TTrainLoader = DataLoader(TTrainData, batch_size=loader.train_sdata_batchsize, shuffle=True)
    STrainData = dataloader.CDPointwiseDataset(loader, loader.train_sdata, 's', config.train_num_ng)
    STrainLoader = DataLoader(STrainData, batch_size=loader.train_tdata_batchsize, shuffle=True)
    OverTrainData = dataloader.CDPointwiseDataset(loader, loader.train_overlapdata, 't',config.train_num_ng)
    OverTrainLoader = DataLoader(OverTrainData, batch_size=loader.train_overdata_batchsize, shuffle=True)
    # OverLoader = DataLoader(loader.overlap_users, batch_size=config.LS_batchsize, shuffle=True)
    TestData = dataloader.PointwiseDataset(loader, loader.test_negetive, 0, False)
    TestLoader = DataLoader(TestData, batch_size=config.test_num_ng + 1, shuffle=False)

    if config.preweight == True:
        assert os.path.exists(config.MF_S_path), 'lack of MF_S model'
        assert os.path.exists(config.MF_T_path), 'lack of MF_T model'
        assert os.path.exists(config.Mapping_path), 'lack of MF_T model'
        mf_s_model = torch.load(config.MF_S_path)
        mf_t_model = torch.load(config.MF_T_path)
        mapping_model = torch.load(config.Mapping_path)
        model = MATN(config, loader, mf_s_model, mf_t_model, mapping_model)
    else:
        model = MATN(config, loader)

    n_users = loader.n_users
    n_titems = loader.n_titems
    n_sitems = loader.n_sitems
    model = model.to(config.device)
    opt = torch.optim.Adam(model.parameters(), lr=config.MATN_lr)

    for epoch in range(config.MATN_epochs):
        start_train_time = time()
        model.train()
        start_time = time()
        #print('ng sample start!')
        TTrainLoader.dataset.ng_sample()
        STrainLoader.dataset.ng_sample()
        OverTrainLoader.dataset.ng_sample()
        Loss,lossS,lossT,lossScore,lossMapping = 0.0, 0.0, 0.0,0.0,0.0
        temp_n = 0
        #print('ng sample over!')
        for i,(batchT, batchS, batchOver) in enumerate(zip(TTrainLoader,STrainLoader,OverTrainLoader)):
            users, items, labels = map(lambda x: x.cuda(), batchT)
            batch_lossT = model.mf_t(users, items, labels)
            users, items, labels = map(lambda x: x.cuda(), batchS)
            batch_lossS = model.mf_s(users, items, labels)
            users, items, labels = map(lambda x: x.cuda(), batchOver)
            batch_lossScore = model.bceloss(users, items, labels)
            #batch_lossMapping = model.mapping(users)
            loss = (batch_lossT + batch_lossS) +config.MATN_loss_weight* (batch_lossScore)
            opt.zero_grad()
            loss.backward()
            opt.step()
            Loss += loss.cpu().detach()
            temp_n += 1
            lossS += batch_lossS.cpu().detach()
            lossT += batch_lossT.cpu().detach()
            #lossMapping += batch_lossMapping.cpu().detach()
            lossScore += batch_lossScore.cpu().detach()

        lossS /= temp_n
        lossT /= temp_n
        lossScore /= temp_n
        #lossMapping /= temp_n
        Loss /= temp_n

        print('MF_S: Epoch %d train==[%.5f]' % (epoch, lossS))
        print('MF_T: Epoch %d train==[%.5f]' % (epoch, lossT))
        print('MappingScore: Epoch %d train==[%.5f]' % (epoch, lossScore))
        #print('Mapping: Epoch %d train==[%.5f]' % (epoch, lossMapping))
        print('Loss: Epoch %d train==[%.5f]' % (epoch, Loss))


        if epoch % 1 == 0:
            print('================test===============')
            with torch.no_grad():
                HR, NDCG = evaluate.EMCDRETE_metrics(model, TestLoader, config.topk)
                print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
