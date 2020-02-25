import torch

from loss.boundary import BoundaryMask
from loss.focal_loss import FocalLoss
from loss.metric_loss import MetricLoss


class LossFactory:
    def __init__(self, cur_iter, warm_up_iters, num_classes,
                 use_boundary_loss = False, boundary_loss_weight = 1,
                 use_metric_loss = False, metric_loss_weight = 0.001
                 ):
        self.cur_iter = cur_iter
        self.warm_up_iters = warm_up_iters
        self.num_classes = num_classes
        self.use_boundary_loss = use_boundary_loss
        self.boundary_loss_weight = boundary_loss_weight
        self.use_metric_loss = use_metric_loss
        self.metric_loss_weight = metric_loss_weight

        # base loss
        self.focal_loss = FocalLoss(num_classes)

        # extra boundary loss option
        if use_boundary_loss and cur_iter >= warm_up_iters:
            # only use focal loss when warm up
            self.boundary_mask = BoundaryMask()

        # extra metric loss option
        if use_metric_loss and cur_iter >= warm_up_iters:
            # only use focal loss when warm up
            self.metric_loss = MetricLoss()

    def compute_loss(self, logits, target):
        """

        :param logits: logits of shape [N, C, H, W], before softmax
        :param target: label of shape [N, 1, H, W]
        :return:
        """
        prob = torch.nn.functional.softmax(logits, dim = 1)

        # focal loss
        focal_loss_return = self.focal_loss.compute_focal_loss(prob, target)

        # boundary loss
        boundary_loss = torch.tensor(0.0)
        if self.use_boundary_loss and self.cur_iter > self.warm_up_iters:
            boundary_loss = self.boundary_loss_weight * self.boundary_mask.get_boundary_mask(target) * focal_loss_return['focal_loss']
            # average on boundary
            boundary_loss = torch.sum(boundary_loss) / torch.sum(boundary_loss != 0)

        # metric loss
        metric_loss = torch.tensor(0.0)
        if self.use_metric_loss and self.cur_iter > self.warm_up_iters:
            metric_loss = self.metric_loss.compute_metric_loss(logits, target)

        # weighted loss
        weighted_loss = focal_loss_return['loss_mean'] + self.boundary_loss_weight * boundary_loss + self.metric_loss_weight * metric_loss

        return {
            'focal_loss': focal_loss_return['loss_mean'],
            'boundary_loss': boundary_loss,
            'metric_loss': metric_loss,
            'weighted_loss': weighted_loss
        }