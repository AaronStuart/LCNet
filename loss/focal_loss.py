import numpy as np
import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=1, gamma=2):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def compute_class_weights(self, frequency):
        classWeights = torch.ones(len(frequency), dtype=torch.float32)
        frequency = frequency / torch.sum(frequency)
        for i in range(len(frequency)):
            classWeights[i] = 1 / (torch.log(1.1 + frequency[i]))

        return classWeights

    def get_one_hot_label(self, label):
        N, C, H, W = label.shape

        # flatten
        flatten_label = label.view(-1)

        # generate one hot label
        one_hot_label = torch.tensor(
            np.eye(self.num_classes, dtype=np.uint8)[flatten_label.cpu().numpy()],
            device=label.device
        )

        # reshape one_hot_label
        one_hot_label = one_hot_label.view(N, H, W, self.num_classes).permute(0, 3, 1, 2)

        return one_hot_label

    def compute_loss(self, logits, label):
        '''
        :param logits: shape [N, C, H, W]
        :param label: shape [N, 1, H, W]
        :return: shape of [N, 1, H, W]
        '''
        prob = torch.nn.functional.softmax(logits, dim=1)

        one_hot_label = self.get_one_hot_label(label)

        p_t = torch.sum(prob * one_hot_label, dim=1)
        p_t = torch.clamp(p_t, min=1e-4, max=1.0)

        focal_loss = -1 * torch.pow((1 - p_t), self.gamma) * torch.log(p_t)

        return focal_loss.mean()
