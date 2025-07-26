import numpy as np
import torch

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
device = torch.device('cpu')

def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    # recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    # here, recall_n identical to 1, because the form of test dataset is [0(pos), 1(neg), ..., 99(neg)]
    recall_n = 1
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    # assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    test_matrix[:, :k] = 1
    # for i, items in enumerate(test_data):
    #     length = k if k <= len(items) else len(items)
    #     test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def hit(gt_item, pred_items):
    if not isinstance(pred_items, list):
        pred_items = [pred_items]
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if not isinstance(pred_items, list):
        pred_items = [pred_items]
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0

def getLabel(groundTrue, pred_data):
    r = []
    for i in range(len(pred_data)):
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def EMCDR_metrics(mf_s, mf_t, mapping, test_loader, top_k):
    HR, NDCG = [], []
    print('start test')
    for user, item in test_loader:
        user, item = user.to(device), item.to(device)
        user_embed = mapping(mf_s.get_embed(user))
        rating = mf_t.get_rating(user_embed, item)
        _, indices = torch.topk(rating, k=top_k)
        # recommends = torch.take(
        #         item, indices).cpu().numpy().tolist()
        indices = indices.cpu().numpy().tolist()
        # gt_item = item[0].item()
        for i in range(len(indices)):
            HR.append(hit(0, indices[i]))
            NDCG.append(ndcg(0, indices[i]))
    return np.mean(HR), np.mean(NDCG)
