import torchvision.utils as vutils
import torch
from tensorboardX import SummaryWriter

from scripts.apollo_label import trainId2color

import PIL
class TrainVisualize:
    def __init__(self, log_dir, model, use_boundary_loss, use_metric_loss):
        self.model = model
        self.summary = SummaryWriter(log_dir)
        self.use_boundary_loss = use_boundary_loss
        self.use_metric_loss = use_metric_loss

    def update(self, iteration, input, label, logits, loss):
        label = torch.squeeze(label)
        predict = torch.argmax(logits, axis=0)
        H, W = label.shape
        # map trainId to color
        predict_color = torch.zeros(size=[3, H, W], dtype=torch.uint8)
        for trainId, rgb in trainId2color.items():
            mask = (predict == trainId)
            predict_color[0][mask] = rgb[0]
            predict_color[1][mask] = rgb[1]
            predict_color[2][mask] = rgb[2]

        # loss visualize
        self.summary.add_scalar('loss/total_loss', loss['total_loss'], iteration)
        self.summary.add_scalar('loss/focal_loss', loss['focal_loss'], iteration)
        self.summary.add_scalar('loss/boundary_loss', loss['boundary_loss'], iteration)
        self.summary.add_scalar('loss/metric_loss', loss['metric_loss'], iteration)

        # input output visualize
        self.summary.add_image('image/input', vutils.make_grid(input.to(torch.uint8)), iteration)
        self.summary.add_image('image/label', vutils.make_grid(label), iteration)
        self.summary.add_image('image/predict', vutils.make_grid(predict_color), iteration)

        # weight visualize
        for name, param in self.model.named_parameters():
            self.summary.add_histogram(name.replace('.', '/'), param.clone().cpu().data.numpy(), iteration)

        # feature map visualize
        logits = logits.unsqueeze(dim = 1)
        self.summary.add_image('origin_logits', vutils.make_grid(logits, normalize = True), iteration)