import torch


class PostProcessing:
    def __init__(self):
        pass

    def forward(self, input_logits, label):
        result = self.softmax(input_logits)
        result = self.resize_to_label_size(result, label)

        return result

    def softmax(self, input_logits, dim = 1):
        return torch.nn.functional.softmax(input_logits, dim = dim)

    def resize_to_label_size(self, input, label):
        N, C, H, W = label.shape
        result = torch.nn.functional.interpolate(input, size = (H, W), mode = 'bilinear')

        return result