import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from models.early_stopping import EarlyStopping

def setup_model(args,dataset):
    if args.model_name == 'CDATN':
        from .CDATN import CDATN
        model = CDATN(dataset,**vars(args))
        model.load_state_dict(torch.load(r'data/movie-book/auxiliary_model.pth'), strict=False)
    if args.model_name == 'LightGCN':
        from .LightGCN import LightGCN
        model = LightGCN(dataset,**vars(args))
    else:
        raise Exception("%s model doesn't exist." % args.model_name)
    return model

class CustomModel(nn.Module):
    def __init__(self, experiment_id, experiment_dir):
        super().__init__()
        # Experiment information
        self.experiment_id = experiment_id
        if experiment_dir is not None:
            self.checkpoint_path = os.path.join(experiment_dir, 'model_parameter.ckpt')
            self.writer = SummaryWriter(experiment_dir)
        self.n_epoch = 0
        self.n_train = 0
        self.n_valid = 0

        # Training strategy
        self.early_stopping = EarlyStopping('recall')

    def save(self):
        torch.save(self.state_dict(), self.checkpoint_path)

    def load(self, checkpoint_path=None):
        if checkpoint_path:
            self.checkpoint_path = checkpoint_path
        self.load_state_dict(torch.load(self.checkpoint_path, map_location=lambda storage, loc: storage))

    def update_model_counter(self):
        if self.training:
            self.n_train += 1
        else:
            self.n_valid += 1

    @staticmethod
    def detect_gradient_anomaly(x):
        if torch.isnan(x).any():
            raise ValueError("NaN Found.")
        if torch.isinf(x).any():
            raise ValueError("Inf Found.")

