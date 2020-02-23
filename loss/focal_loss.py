import numpy as np
import torch
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=1, gamma=2):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    # def compute_class_weights(self, frequency):
    #     classWeights = torch.ones(len(frequency), dtype=torch.float32)
    #     frequency = frequency / torch.sum(frequency)
    #     for i in range(len(frequency)):
    #         classWeights[i] = 1 / (torch.log(1.1 + frequency[i]))
    #     return classWeights

    def compute_focal_loss(self, input, target):
        '''
            :param input: shape [N, C, H, W] 经过Softmax后的输出
            :param target: shape [N, 1, H, W]
            :return: shape of [N, 1, H, W]
            '''
        # # calculate class weights
        # frequency = torch.Tensor([torch.sum(target == i) for i in range(self.num_classes)])
        # classWeights = self.compute_class_weights(frequency)
        #
        # # generate classWeights mask
        # weightsMask = torch.zeros_like(target, dtype = torch.float)
        # for i in range(self.num_classes):
        #     mask = target == i
        #     weightsMask[mask] = classWeights[i]

        one_hot_label = torch.zeros_like(input).scatter_(dim = 1, index = target.long(), src = torch.tensor(1))
        p_t = torch.sum(input * one_hot_label, dim = 1)
        # Consider numerical stability
        p_t = torch.clamp(p_t, min = 1e-8, max = 1.0)
        focal_loss = -1 * torch.pow((1 - p_t), self.gamma) * torch.log(p_t)
        # focal_loss = -1 * weightsMask * torch.pow((1 - p_t), self.gamma) * torch.log(p_t)

        return {
            'focal_loss' : focal_loss,
            'loss_mean' : focal_loss.mean()
        }

if __name__ == '__main__':
    predict = np.array([
        [0.0, 100.0],
        [0.0, 10.0],
        [0.0, 1.0]
    ])
    label = np.array([0, 0, 0])
    print(FocalLoss()(torch.Tensor(predict), torch.Tensor(label).to(torch.int64)))