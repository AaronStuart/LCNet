import torch

from loss.focal_loss import FocalLoss
from loss.metric_loss import RankedListLoss


class LossFactory(object):
    def __init__(self, num_classes, use_metric_loss=False, metric_loss_weight=1):
        """"""
        self.num_classes = num_classes
        self.use_metric_loss = use_metric_loss
        self.metric_loss_weight = metric_loss_weight

        # base loss
        self.focal_loss = FocalLoss(num_classes)

        # extra loss
        if use_metric_loss:
            self.metric_loss = RankedListLoss()

    def compute_loss(self, logits, target, iter, warm_up_iters):
        """

        :param logits: logits of shape [N, C, H, W], before softmax
        :param target: label of shape [N, 1, H, W]
        :return:
        """
        prob = torch.nn.functional.softmax(logits, dim=1)

        # focal loss
        focal_loss = self.focal_loss.compute_loss(prob, target)

        # metric loss
        metric_loss = torch.tensor(0.0)
        if self.use_metric_loss and iter > warm_up_iters:
            metric_loss = self.metric_loss.compute_loss(logits, target)

        # weighted loss
        weighted_loss = focal_loss + self.metric_loss_weight * metric_loss

        return {
            'focal_loss': focal_loss,
            'metric_loss': metric_loss,
            'total_loss': weighted_loss
        }
