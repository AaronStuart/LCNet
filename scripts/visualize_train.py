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
        """
        add info to summary
        :param iteration:
        :param input: tensor, shape is [N, 3, H, W]
        :param label: tensor, shape is [N, 3, H, W]
        :param logits: tensor, shape is [N, C, H, W]
        :param loss:
        :return:
        """
        # change to NCHW, and change form BGR to RGB
        predict_gray = torch.argmax(logits, axis=1)
        # map trainId to color
        predict_color = torch.zeros_like(label)
        for trainId, rgb in trainId2color.items():
            mask = (predict_gray == trainId)
            predict_color[:, 0, :, :][mask] = rgb[0]
            predict_color[:, 1, :, :][mask] = rgb[1]
            predict_color[:, 2, :, :][mask] = rgb[2]

        # loss visualize
        self.summary.add_scalar('loss/focal_loss', loss, iteration)

        # input output visualize
        self.summary.add_image('image/input', vutils.make_grid(input[0]), iteration)
        self.summary.add_image('image/label', vutils.make_grid(label[0]), iteration)
        self.summary.add_image('image/predict', vutils.make_grid(predict_color[0]), iteration)

        # weight visualize
        for name, param in self.model.named_parameters():
            self.summary.add_histogram(name.replace('.', '/'), param.clone().cpu().data.numpy(), iteration)

        # feature map visualize
        visualize_logits = logits[0].detach().cpu().unsqueeze(dim = 1)
        self.summary.add_image('logits', vutils.make_grid(visualize_logits, normalize = True), iteration)