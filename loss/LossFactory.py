from loss.boundary_loss import BoundaryLoss
from loss.focal_loss import FocalLoss

class LossFactory:
    def __init__(self, cur_iter, num_classes, use_boundary_loss = False, boundary_weight = 0):
        self.num_classes = num_classes
        self.use_boundary_loss = use_boundary_loss

        # base loss
        self.focal_loss = FocalLoss(num_classes)

        # extra loss option
        if use_boundary_loss:
            if cur_iter < 1000:
                # only use focal loss when warm up
                self.boundary_weight = 0
                self.boundary_loss = BoundaryLoss(weight = 0)
            else:
                # add boundary loss
                self.boundary_weight = boundary_weight
                self.boundary_loss = BoundaryLoss(weight = boundary_weight)

    def compute_loss(self, input, target):
        """

        :param input: prob of shape [N, C, H, W], after softmax
        :param target: label of shape [N, 1, H, W]
        :return:
        """
        focal_loss_result = self.focal_loss.compute_focal_loss(input, target)

        if self.use_boundary_loss:
            boundary_loss_result = self.boundary_loss.compute_boundary_loss(focal_loss_result['focal_loss'], target)
            return {
                'focal_loss': focal_loss_result['loss_mean'],
                'boundary_weight': self.boundary_weight,
                'boundary_loss': boundary_loss_result['loss_mean'],
                'total_loss': focal_loss_result['loss_mean'] + boundary_loss_result['loss_mean']
            }
        else:
            return {
                'focal_loss': focal_loss_result['loss_mean'],
                'total_loss': focal_loss_result['loss_mean']
            }