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

    def update(self, iteration, origin_image, origin_label, origin_logits, resized_logits, loss):
        predict_gray = torch.argmax(resized_logits, axis=1)
        # map trainId to color
        predict_color = torch.zeros_like(origin_label)
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
        self.summary.add_image('image/origin_image', vutils.make_grid(origin_image), iteration)
        self.summary.add_image('image/origin_label', vutils.make_grid(origin_label), iteration)
        self.summary.add_image('image/predict_image', vutils.make_grid(predict_color), iteration)

        # weight visualize
        for name, param in self.model.named_parameters():
            self.summary.add_histogram(name.replace('.', '/'), param.clone().cpu().data.numpy(), iteration)

        # feature map visualize
        visualize_logits = origin_logits[0].detach().cpu().unsqueeze(dim = 1)
        self.summary.add_image('origin_logits', vutils.make_grid(visualize_logits, normalize = True), iteration)

        # visualize_logits = resized_logits[0].detach().cpu().unsqueeze(dim = 1)
        # self.summary.add_image('resized_logits', vutils.make_grid(visualize_logits, normalize = True), iteration)