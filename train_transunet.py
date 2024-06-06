import torch
from torch.optim import SGD, Adam

# Additional Scripts
from utils.transunet import TransUNet
from utils.utils import dice_loss
from config import cfg
from utils.metrics import *

class TransUNetSeg:
    def __init__(self, device):
        self.device = device
        self.model = TransUNet(img_dim=cfg.transunet.img_dim,
                               in_channels=cfg.transunet.in_channels,
                               out_channels=cfg.transunet.out_channels,
                               head_num=cfg.transunet.head_num,
                               mlp_dim=cfg.transunet.mlp_dim,
                               block_num=cfg.transunet.block_num,
                               patch_dim=cfg.transunet.patch_dim,
                               class_num=cfg.transunet.class_num).to(self.device)

        self.criterion = dice_loss
        # self.optimizer = SGD(self.model.parameters(), lr=cfg.learning_rate,
        #                      momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        self.optimizer = Adam(self.model.parameters(), lr= cfg.learning_rate)

    def load_model(self, path):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        self.model.eval()

    def train_step(self, **params):
        self.model.train()

        self.optimizer.zero_grad()
        pred_mask = self.model(params['img'], params['img_sail'])
        loss = self.criterion(pred_mask, params['mask'])
        IOU = intersection_over_union(pred_mask, params['mask'])
        acc = accuracy(pred_mask, params['mask'])
        F1 = f1_score(pred_mask, params['mask'])

        loss.backward()

        self.optimizer.step()

        metrics = [IOU , F1 , acc]

        return loss.item(), pred_mask , metrics

    def test_step(self, **params):
        self.model.eval()

        pred_mask = self.model(params['img'], params['img_sail'])
        loss = self.criterion(pred_mask, params['mask'])
        
        a = 0

        return loss.item(), pred_mask , a
