from config import config
import dataloader
from torch.utils.data import DataLoader
import torch
from time import time
from model import PureMF,MF,MLP,Multi_MLP,EATN,ETN,ATN
import evaluate
import numpy as np
import os

loader = dataloader.Loader(config)
loader.load_data('one')


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

if config.model == 'NCF':
    print('Choose Model: PureMF!')
    TrainData = dataloader.PointwiseDataset(loader,loader.train_data,config.train_num_ng,True)
    TrainLoader = DataLoader(TrainData, batch_size=config.NCF_batch_size, shuffle=True)
    TestData = dataloader.PointwiseDataset(loader, loader.test_negetive, 0, False)
    TestLoader = DataLoader(TestData, batch_size=config.test_num_ng+1, shuffle=False)

    model = PureMF(config,loader).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=config.NCF_lr)
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

if config.model == 'EATN':
    print('Choose EATN!')
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
        model = EATN(config, loader, mf_s_model, mf_t_model, mapping_model)
    else:
        model = EATN(config, loader)

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
            #batch_lossScore = model.bceloss(users, items, labels)
            batch_lossMapping = model.mapping(users,items)
            loss = (batch_lossT + batch_lossS) + config.MATN_loss_weight* (batch_lossMapping)
            opt.zero_grad()
            loss.backward()
            opt.step()
            Loss += loss.cpu().detach()
            temp_n += 1
            lossS += batch_lossS.cpu().detach()
            lossT += batch_lossT.cpu().detach()
            lossMapping += batch_lossMapping.cpu().detach()
            #lossScore += batch_lossScore.cpu().detach()

        lossS /= temp_n
        lossT /= temp_n
        #lossScore /= temp_n
        lossMapping /= temp_n
        Loss /= temp_n

        print('MF_S: Epoch %d train==[%.5f]' % (epoch, lossS))
        print('MF_T: Epoch %d train==[%.5f]' % (epoch, lossT))
        #print('MappingScore: Epoch %d train==[%.5f]' % (epoch, lossScore))
        print('Mapping: Epoch %d train==[%.5f]' % (epoch, lossMapping))
        print('Loss: Epoch %d train==[%.5f]' % (epoch, Loss))


        if epoch % 1 == 0:
            print('================test===============')
            with torch.no_grad():
                HR, NDCG = evaluate.EMCDRETE_metrics(model, TestLoader, config.topk)
                print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

if config.model == 'ETN':
    print('Choose ETN!')
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
        model = ETN(config, loader, mf_s_model, mf_t_model, mapping_model)
    else:
        model = ETN(config, loader)

    n_users = loader.n_users
    n_titems = loader.n_titems
    n_sitems = loader.n_sitems
    model = model.to(config.device)
    opt = torch.optim.Adam(model.parameters(), lr=config.BTN_lr)

    for epoch in range(config.BTN_epochs):
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
            #batch_lossMapping = model.mapping(users,items)
            loss = (batch_lossT + batch_lossS) +config.BTN_loss_weight* (batch_lossScore)
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

if config.model == 'ATN':
    print('\n================t Domain ATN================')
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

    print('\n================s Domain ATN================')
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
    mapping = ATN(config.LS_dim, config.LS_layers)
    mapping = mapping.to(config.device)
    opt = torch.optim.Adam(mapping.parameters(), lr=config.LS_lr)
    for epoch in range(config.LS_epochs):
        loss_sum = 0
        temp_n = 0

        for batch in TTrainLoader:
            users, items, labels = map(lambda x: x.cuda(), batch)
            users = users.cuda()
            u_s = mf_s.get_embed(users)
            u_t = mf_t.get_embed(users)
            i_t = mf_t.get_item_embed(items)
            batch_loss = mapping(u_s,u_t,i_t)

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
        HR, NDCG = evaluate.ATN_metrics(mf_s, mf_t, mapping.mapping, TestLoader, config.topk)
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

'''
if config.model == 'ATN':
    print('\n================t Domain MF================')
    TTrainData = dataloader.CDPointwiseDataset(loader, loader.train_tdata, 't',config.train_num_ng)
    TTrainLoader = DataLoader(TTrainData, batch_size=config.LF_batchsize, shuffle=True)
    #n_users = loader.n_users
    #n_items = loader.n_titems
    if config.preweight == True:
        assert os.path.exists(config.MF_S_path), 'lack of MF_S model'
        assert os.path.exists(config.MF_T_path), 'lack of MF_T model'
        assert os.path.exists(config.Mapping_path), 'lack of MF_T model'
        mf_s_model = torch.load(config.MF_S_path)
        mf_t_model = torch.load(config.MF_T_path)
        mapping_model = torch.load(config.Mapping_path)
        model = ATN(config, loader, mf_s_model, mf_t_model, mapping_model)
    else:
        model = ATN(config, loader)

    model = model.to(config.device)
    opt = torch.optim.Adam(model.parameters(), lr=config.MATN_lr)

    for epoch in range(config.LF_epochs):
        start_train_time = time()
        model.train()
        start_time = time()
        TTrainLoader.dataset.ng_sample()
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
        print('Epoch %d train==[%.5f]' % (epoch, loss))

    print('\n================s Domain MF================')
    STrainData = dataloader.CDPointwiseDataset(loader, loader.train_sdata, 's',config.train_num_ng)
    STrainLoader = DataLoader(STrainData, batch_size=config.LF_batchsize, shuffle=True)
    #n_users = loader.n_users
    #n_items = loader.n_sitems

    for epoch in range(config.LF_epochs):
        start_train_time = time()
        model.train()
        start_time = time()
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
        print('Epoch %d train==[%.5f]' % (epoch, loss))


    print('\n================mapping================')
    OverLoader = DataLoader(loader.overlap_users, batch_size=config.LS_batchsize, shuffle=True)

    for epoch in range(config.LS_epochs):
        loss_sum = 0
        temp_n = 0
        for batch in TTrainLoader:
            users, items, labels = map(lambda x: x.cuda(), batch)
            batch_loss = model.mapping(users,items)
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
            loss_sum += batch_loss
            temp_n += 1
        loss_sum /= temp_n
        print('Epoch %d train==[%.5f]' % (epoch, loss_sum))

    TestData = dataloader.PointwiseDataset(loader, loader.test_negetive, 0, False)
    TestLoader = DataLoader(TestData, batch_size=config.test_num_ng+1, shuffle=False)
    print('================test===============')
    with torch.no_grad():
        HR, NDCG = evaluate.EMCDRETE_metrics(model, TestLoader, config.topk)
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
'''

if config.model == 'EATN-s':
    print('Choose EATN-s!')
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
        model = EATN(config, loader, mf_s_model, mf_t_model, mapping_model)
    else:
        model = EATN(config, loader)

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
            #batch_lossScore = model.bceloss(users, items, labels)
            batch_lossMapping = model.mapping(users,items)
            loss = batch_lossMapping
            opt.zero_grad()
            loss.backward()
            opt.step()
            Loss += loss.cpu().detach()
            temp_n += 1
            lossS += batch_lossS.cpu().detach()
            lossT += batch_lossT.cpu().detach()
            lossMapping += batch_lossMapping.cpu().detach()
            #lossScore += batch_lossScore.cpu().detach()

        lossS /= temp_n
        lossT /= temp_n
        #lossScore /= temp_n
        lossMapping /= temp_n
        Loss /= temp_n

        print('MF_S: Epoch %d train==[%.5f]' % (epoch, lossS))
        print('MF_T: Epoch %d train==[%.5f]' % (epoch, lossT))
        #print('MappingScore: Epoch %d train==[%.5f]' % (epoch, lossScore))
        print('Mapping: Epoch %d train==[%.5f]' % (epoch, lossMapping))
        print('Loss: Epoch %d train==[%.5f]' % (epoch, Loss))


        if epoch % 1 == 0:
            print('================test===============')
            with torch.no_grad():
                HR, NDCG = evaluate.EMCDRETE_metrics(model, TestLoader, config.topk)
                print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

if config.model == 'MTN':
    print('Choose MTN!')
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
        model = MTN(config, loader, mf_s_model, mf_t_model, mapping_model)
    else:
        model = MTN(config, loader)

    n_users = loader.n_users
    n_titems = loader.n_titems
    n_sitems = loader.n_sitems
    model = model.to(config.device)
    opt = torch.optim.Adam(model.parameters(), lr=config.MTN_lr)

    for epoch in range(config.MTN_epochs):
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
            batch_lossMapping = model.mapping(users,items)
            loss = (batch_lossT + batch_lossS) +config.MTN_loss_weight* (batch_lossScore)
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
