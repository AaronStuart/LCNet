import torch
import torch.nn.functional as F
import numpy as np

class ClusterLoss(object):
    def __init__(self, gamma=2, normalize_type='L2', distance_type='L2'):
        self.gamma = gamma

        assert normalize_type in ['L2', 'softmax'], "Error normalize_type"
        self.normalize_type = normalize_type

        assert distance_type in ['L2'], "Error distance_type"
        self.distance_type = distance_type

    def compute_distance(self, logits, center):
        """

        :param logits: torch tensor of shape [N, classes]
        :param center: torch tensor of shape [N, classes]
        :return: torch tensor of shape [N]
        """
        if self.distance_type == 'L2':
            distance = torch.pow(logits - center, 2).sum(dim = 1).sqrt()

        return distance

    def compute_loss(self, logits, label):
        """

        :param logits: torch tensor, shape is [N, num_classes, H, W]
        :param label: torch tesnor, shape is [N, 1, H, W]
        :return:
        """
        # scale feature vectors
        if self.normalize_type == 'L2':
            logits = logits / torch.norm(logits, p=2, dim=1, keepdim=True)
        elif self.normalize_type == 'softmax':
            logits = F.softmax(logits, dim = 1)

        N, C, H, W = logits.shape
        flatten_logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        flatten_label = label.view(-1)

        foreground_logits = flatten_logits[flatten_label != 0]

        foreground_center = torch.tensor(
            np.eye(C, dtype = np.float)[(flatten_label[flatten_label != 0]).cpu().numpy()],
            device = logits.device
        )

        foreground_distance = self.compute_distance(foreground_logits, foreground_center)

        cluster_loss = torch.pow(torch.tanh(foreground_distance), self.gamma) * torch.log(1 + foreground_distance)

        return cluster_loss.mean()