import torch
import cv2 as cv
import numpy as np

from scripts.apollo_label import color2trainId


class MetricLoss:
    def __init__(self, alpha = 1.2, margin = 0.4):
        super(MetricLoss, self).__init__()
        self.alpha = alpha
        self.margin = margin

    def compute_metric_loss(self, logits, label):
        # L2 Normalize
        normed_logits = logits / torch.norm(logits, p = 2, dim = 1, keepdim = True)

        # reshape to [N, C, H*W] format
        N, C, H, W = label.shape
        flat_label = label.reshape(shape = (N, -1, H * W))
        flat_logits = normed_logits.reshape(shape = (N, -1, H * W))

        batch_positive_loss, batch_negative_loss = 0, 0
        for i in range(N):
            unique_classes = torch.unique(label[i])
            positive_loss, negative_loss = 0, 0
            for j in unique_classes:
                # collect positive set and negative set
                pos_indexes = torch.nonzero(flat_label[i, 0] == j).numpy().squeeze()
                neg_indexes = torch.nonzero(flat_label[i, 0] != j).numpy().squeeze()
                positives = flat_logits[i, :, pos_indexes].transpose(0, 1)
                negatives = flat_logits[i, :, neg_indexes].transpose(0, 1)

                # random select an anchor
                random_index = np.random.choice(pos_indexes)
                anchor = torch.unsqueeze(flat_logits[i, :, random_index], dim = 0)

                # calculate max inner-class distance
                max_inner_distance = torch.dist(anchor, positives).max()
                positive_loss += max_inner_distance if max_inner_distance > (self.alpha - self.margin) else 0

                # calculate min between-class distance
                min_between_distance = torch.dist(anchor, negatives).min()
                negative_loss += min_between_distance if min_between_distance < self.alpha else 0

            batch_positive_loss += positive_loss / len(unique_classes)
            batch_negative_loss += negative_loss / len(unique_classes)

        batch_positive_loss = batch_positive_loss / N
        batch_negative_loss = batch_negative_loss / N

        batch_loss = batch_positive_loss + batch_negative_loss

        return batch_loss

if __name__ == '__main__':
    label_bgr = cv.imread(
        "/media/stuart/data/dataset/Apollo/Lane_Detection/Labels_road04/Label/Record001/Camera 5/171206_053953494_Camera_5_bin.png",
        cv.IMREAD_UNCHANGED)
    # create a black train_id_label
    canvas = np.zeros(label_bgr.shape[:2], dtype=np.uint8)
    for color, trainId in color2trainId.items():
        # map color to trainId
        mask = (label_bgr == color[::-1]).all(axis=2)
        canvas[mask] = trainId
    canvas = np.expand_dims(canvas, axis=0)
    canvas = np.expand_dims(canvas, axis=0)
    label = torch.tensor(canvas)
    logits = torch.rand(1, 38, label.shape[2], label.shape[3])

    print(MetricLoss().compute_metric_loss(logits, label)['loss_mean'])
