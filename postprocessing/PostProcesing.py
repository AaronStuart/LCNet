import torch
import torch.nn as nn
import torch.nn.functional as F

class PostProcessing(nn.Module):
    def __init__(self):
        super(PostProcessing, self).__init__()
        self.softmax = torch.nn.Softmax(dim = 1)
        self.resize = F.interpolate

    def forward(self, input_logits, label):
        # resize to label's size
        result = self.resize(input_logits, label.shape[2 : ])
        # softmax
        result = self.softmax(result)

        return result