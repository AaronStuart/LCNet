import torch

from loss.cluster_loss import ClusterLoss
from loss.focal_loss import FocalLoss


class LossFactory(object):
    def __init__(self, num_classes, use_cluster_loss=False, cluster_loss_weight=0):
        """"""
        self.num_classes = num_classes

        self.use_cluster_loss = use_cluster_loss
        self.cluster_loss_weight = cluster_loss_weight

        # base loss
        self.focal_loss = FocalLoss(num_classes)

        if use_cluster_loss:
            self.cluster_loss = ClusterLoss(num_classes)

    def compute_loss(self, logits, label):
        """

        :param logits: logits of shape [N, C, H, W], before softmax
        :param label: label of shape [N, 1, H, W]
        :return:
        """
        loss = {'total_loss': torch.tensor(0.0, dtype = torch.float, device = logits.device)}

        # focal loss
        focal_loss = self.focal_loss.compute_loss(logits, label)
        loss['focal_loss'] = focal_loss
        loss['total_loss'] += focal_loss

        # extra loss
        if self.use_cluster_loss:
            cluster_loss = self.cluster_loss.compute_loss(logits, label)
            loss['cluster_loss'] = cluster_loss
            loss['total_loss'] += self.cluster_loss_weight * cluster_loss

        return loss
