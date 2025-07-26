import argparse
import torch


parser = argparse.ArgumentParser(description="Test")
parser.add_argument('--dataset', type=str, default='amazon', help="Musical_Patio")
parser.add_argument('--model', type=str, default='ATN', help="MF,EMCDR,NCF,MATN,CMF")
parser.add_argument('--layers', type=int, default=1, help="l")
parser.add_argument('--batchsize', type=int, default=2, help="b")
parser.add_argument('--topk', type=int, default=10, help="tk")
parser.add_argument('--out', type=bool, default=True, help="tk")
parser.add_argument('--preweight', type=bool, default=True, help="tk")
parser.add_argument('--n_batch', type=int, default=200, help="b")



parser.add_argument('--LF_model', type=str, default='MF', help="MF")
parser.add_argument('--LF_dim', type=int, default=16, help="embedding")
parser.add_argument('--LF_lr', type=float, default=0.001, help="lr")
parser.add_argument('--LF_reg', type=float, default=0, help="r")
parser.add_argument('--LF_epochs', type=int, default=10, help='e')
parser.add_argument('--LF_batchsize', type=int, default=1024, help="b")

parser.add_argument('--LS_model', type=str, default='MLP', help="MLP,Multi_MLP")
parser.add_argument('--LS_dim', type=int, default=16, help="embedding")
parser.add_argument('--LS_layers', type=int, default=1, help="l")
parser.add_argument('--LS_lr', type=float, default=0.001, help="lr")
parser.add_argument('--LS_reg', type=float, default=0, help="r")
parser.add_argument('--LS_epochs', type=int, default=10, help='e')
parser.add_argument('--LS_batchsize', type=int, default=32, help="b")
parser.add_argument('--LS_overlapratio', type=float, default=0.5, help="b")

# PureMF
parser.add_argument('--test_num_ng', type=int, default=99, help="b")
parser.add_argument('--train_num_ng', type=int, default=4, help="b")
parser.add_argument('--PureMF_batch_size', type=int, default=256, help="b")
parser.add_argument('--EPOCHS', type=int, default=100, help="b")
parser.add_argument('--PureMF_dim', type=int, default=32, help="b")
parser.add_argument('--PureMF_lr', type=float, default=0.001, help="b")

#TMF

parser.add_argument('--NCF_dim', type=int, default=16, help="b")
parser.add_argument('--NCF_layers', type=int, default=3, help="b")
parser.add_argument('--NCF_batch_size', type=int, default=256, help="b")
parser.add_argument('--NCF_lr', type=float, default=0.001, help="b")

parser.add_argument('--MATN_loss_weight', type=float, default=0.001, help="b")
parser.add_argument('--MATN_lr', type=float, default=0.001, help="b")
parser.add_argument('--MATN_embed_dim', type=int, default=16, help="b")
parser.add_argument('--MATN_epochs', type=int, default=100, help="b")

parser.add_argument('--BTN_loss_weight', type=float, default=0.001, help="b")
parser.add_argument('--BTN_lr', type=float, default=0.001, help="b")
parser.add_argument('--BTN_embed_dim', type=int, default=16, help="b")
parser.add_argument('--BTN_epochs', type=int, default=100, help="b")

parser.add_argument('--MTN_loss_weight', type=float, default=0.001, help="b")
parser.add_argument('--MTN_lr', type=float, default=0.001, help="b")
parser.add_argument('--MTN_embed_dim', type=int, default=16, help="b")
parser.add_argument('--MTN_epochs', type=int, default=100, help="b")

config = parser.parse_args()
config.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
config.model_path = '../models/'
config.read_model_path = '../models/Multi_MLP/'
config.MF_S_path = config.read_model_path + 'mf_s.pth'
config.MF_T_path = config.read_model_path + 'mf_t.pth'
config.Mapping_path = config.read_model_path + 'mapping.pth'