from utils import *
import torch.nn as nn
import torch.nn.functional as F


class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss



class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.iou = SoftIoULoss()

    def forward(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for pred in preds:
                loss_total += (self.iou(pred, gt_masks) + self.bce(pred, gt_masks))
            return loss_total / len(preds)
        else:
            return self.iou(preds, gt_masks) + self.bce(preds, gt_masks)




class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for pred in preds:
                loss_total += self.calculate_loss(pred, gt_masks)
            return loss_total / len(preds)
        else:
            return self.calculate_loss(preds, gt_masks)

    def calculate_loss(self, pred, target):
        bce_loss = self.bce(pred, target)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

