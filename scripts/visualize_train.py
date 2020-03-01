import torchvision.utils as vutils
import torch
from tensorboardX import SummaryWriter

from scripts.apollo_label import trainId2color


class TrainVisualize:
    def __init__(self, log_dir, model, use_boundary_loss, use_metric_loss):
        self.model = model
        self.summary = SummaryWriter(log_dir)
        self.use_boundary_loss = use_boundary_loss
        self.use_metric_loss = use_metric_loss

    def update(self, iteration, input, label, logits, loss):
        # change to NCHW, and change form BGR to RGB
        label = label.permute(dims = [0, 3, 1, 2])
        predict_gray = torch.argmax(logits, axis=1)
        # map trainId to color
        predict_color = torch.zeros_like(label)
        for trainId, rgb in trainId2color.items():
            mask = (predict_gray == trainId)
            predict_color[:, 0, :, :][mask] = rgb[0]
            predict_color[:, 1, :, :][mask] = rgb[1]
            predict_color[:, 2, :, :][mask] = rgb[2]

        # loss visualize
        self.summary.add_scalar('loss/weighted_loss', loss['weighted_loss'], iteration)
        self.summary.add_scalar('loss/focal_loss', loss['focal_loss'], iteration)
        self.summary.add_scalar('loss/boundary_loss', loss['boundary_loss'], iteration)
        self.summary.add_scalar('loss/metric_loss', loss['metric_loss'], iteration)

        # input output visualize
        self.summary.add_image('image/input', vutils.make_grid(input), iteration)
        self.summary.add_image('image/label', vutils.make_grid(label), iteration)
        self.summary.add_image('image/predict', vutils.make_grid(predict_color), iteration)

        # weight visualize
        for name, param in self.model.named_parameters():
            self.summary.add_histogram(name.replace('.', '/'), param.clone().cpu().data.numpy(), iteration)

        # feature map visualize
        visualize_logits = logits[0].detach().cpu().unsqueeze(dim = 1)
        self.summary.add_image('origin_logits', vutils.make_grid(visualize_logits, normalize = True), iteration)