import numpy as np
import torch
import cv2 as cv

from scripts.apollo_label import color2trainId


class BoundaryMask:
    def __init__(self):
        pass

    def get_boundary_mask(self, target):
        # create blank mask
        boundary_mask = np.zeros(target.shape, dtype=np.uint8)

        # fill blank mask by boundary
        for i in range(target.shape[0]):
            label = target[i, 0].numpy().astype(np.uint8)
            contours, _ = cv.findContours(label, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(boundary_mask[i, 0], contours, -1, 1, 2)

        return torch.tensor(boundary_mask, dtype = torch.float)

    def compute_boundary_loss(self, input_loss, target):
        """Compute boundary weight mask by boundary

        :param input_loss: tensor of shape [N, 1, H, W]
        :param target: tensor of shape [N, 1, H, W]
        :return: loss of shape [N, 1, H, W]
        """
        # create weight mask
        boundary_weight = np.zeros(target.shape, dtype = np.uint8)

        # fill weight mask by boundary
        for i in range(target.shape[0]):
            label = target[i, 0].numpy().astype(np.uint8)
            contours, _ = cv.findContours(label, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(boundary_weight[i, 0], contours, -1, self.weight, 2)
        # compute boundary loss
        boundary_loss = input_loss * torch.tensor(boundary_weight, dtype = torch.float)

        # compute mean on boundary
        non_zero = torch.sum(boundary_loss != 0)
        if non_zero == 0:
            mean_loss = torch.zeros(1)
        else:
            mean_loss = torch.sum(boundary_loss) / non_zero

        return {
            'boundary_loss': boundary_loss,
            'loss_mean': mean_loss
        }


if __name__ == '__main__':
    label_bgr = cv.imread("/media/stuart/data/dataset/Apollo/Lane_Detection/Labels_road04/Label/Record001/Camera 5/171206_053953494_Camera_5_bin.png", cv.IMREAD_UNCHANGED)
    # create a black train_id_label
    canvas = np.zeros(label_bgr.shape[:2], dtype=np.uint8)
    for color, trainId in color2trainId.items():
        # map color to trainId
        mask = (label_bgr == color[::-1]).all(axis=2)
        canvas[mask] = trainId
    canvas = np.expand_dims(canvas, axis=0)

    label_trainId = torch.tensor(canvas)
    input_loss = torch.ones_like(label_trainId)
    BoundaryLoss().compute_boundary_loss(input_loss, label_trainId)