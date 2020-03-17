import random

import torch
from memory_profiler import profile
from line_profiler import LineProfiler
from tqdm import tqdm

from scripts.apollo_label import valid_trainIds


class RankedListLoss(object):
    def __init__(self, alpha=1.2, margin=0.4, T=10, lamb=1):
        super(RankedListLoss, self).__init__()
        self.alpha = alpha
        self.m = margin
        self.T = T
        self.lamb = lamb


    def euclidean_distance(self, feature, logits):
        """Compute euclidean distance between pixels

        :param feature: torch tensor, shape is [num_classes, 1]
        :param logits: torch tensor, shape is [num_classes, H * W]
        :return: euclidean vector, shape is [H * W]
        """
        # compute euclidean distance
        dist = torch.pow(feature - logits, 2).sum(dim = 0).clamp(min = 1e-12).sqrt().view(-1)

        return dist


    def compute_single_image_loss(self, logits, label):
        """

        :param logits: torch tensor, shape is [num_classes, H, W]
        :param label: torch tesnor, shape is [1, H, W]
        :return:
        """
        total_loss = torch.tensor(0.0, device=label.device)

        # flatten logits and label
        C, H, W = logits.shape
        logits = logits.view(C, -1)
        label = label.view(-1)

        unique_classes = label.unique()
        for label_value in unique_classes:
            # get all index of one class
            index = torch.nonzero(label == label_value)

            # random select one pixel for each class to query
            random_index = random.choice(index)

            # Compute positve pairs' weight
            distances = self.euclidean_distance(logits[:, random_index], logits)
            positive_weight = torch.where(
                (label == label_value) * (distances > self.alpha - self.m),
                torch.exp(self.T * (distances - (self.alpha - self.m))),
                torch.tensor(0.0, device=label.device)
            )

            # Compute positive pair loss
            weight_sum = torch.sum(positive_weight)
            loss_positive = torch.sum(
                (positive_weight / weight_sum) * (distances - (self.alpha - self.m))
            ) if weight_sum != 0 else 0

            # Compute negative pairs' weight
            negative_weight = torch.where(
                (label != label_value) * (distances < self.alpha),
                torch.exp(self.T * (self.alpha - distances)),
                torch.tensor(0.0, device=label.device)
            )

            # Compute negative pair loss
            weight_sum = torch.sum(negative_weight)
            loss_negative = torch.sum(
                (negative_weight / weight_sum) * (self.alpha - distances)
            ) if weight_sum != 0 else 0

            loss_rll = loss_positive + self.lamb * loss_negative

            # update state
            total_loss += loss_rll

        return total_loss / len(unique_classes)


    def compute_loss(self, logits, label):
        """

        :param logits: torch tensor, shape is [N, num_classes, H, W]
        :param label: torch tesnor, shape is [N, 1, H, W]
        :return:
        """

        # Normalize feature vectors
        logits = logits / torch.norm(logits, p=2, dim=1, keepdim=True)

        loss = torch.tensor(0.0, device=label.device)
        for i in range(label.shape[0]):
            loss = loss + self.compute_single_image_loss(logits[0], label[0])

        return loss / label.shape[0]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = torch.rand([1, 38, 800, 800], requires_grad=True)
    label = torch.randint(low=0, high=37, size=[1, 1, 800, 800])
    print(RankedListLoss().compute_loss(logits.to(device), label.to(device)))
