import os
from datetime import datetime

import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from scripts.apollo_label import trainId2color, trainId2name


class TrainVisualize:
    def __init__(self, log_dir, model):
        self.model = model
        self.summary = SummaryWriter(
            logdir=os.path.join(log_dir, "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now()))
        )

    def update(self, iteration, learning_rate, input, label, logits, loss):
        label = torch.squeeze(label)
        predict = torch.argmax(logits, axis=0)
        H, W = label.shape

        # map predict to color
        predict_color = torch.zeros(size=[3, H, W], dtype=torch.uint8)
        for trainId, rgb in trainId2color.items():
            mask = (predict == trainId)
            predict_color[0][mask] = rgb[0]
            predict_color[1][mask] = rgb[1]
            predict_color[2][mask] = rgb[2]

        # map label to color
        label_color = torch.zeros(size=[3, H, W], dtype=torch.uint8)
        for trainId, rgb in trainId2color.items():
            mask = (label == trainId)
            label_color[0][mask] = rgb[0]
            label_color[1][mask] = rgb[1]
            label_color[2][mask] = rgb[2]

        # learning rate visualize
        self.summary.add_scalar('learning_rate', learning_rate, iteration)

        # loss visualize
        self.summary.add_scalar('loss/total_loss', loss['total_loss'], iteration)
        self.summary.add_scalar('loss/focal_loss', loss['focal_loss'], iteration)
        self.summary.add_scalar('loss/metric_loss', loss['metric_loss'], iteration)
        self.summary.add_scalar('loss/cluster_loss', loss['cluster_loss'], iteration)

        # input visualize
        self.summary.add_image('image/input', vutils.make_grid(input.to(torch.uint8)), iteration)
        self.summary.add_image('image/label', vutils.make_grid(label_color), iteration)
        self.summary.add_image('image/predict', vutils.make_grid(predict_color), iteration)

        # weight visualize
        for name, param in self.model.named_parameters():
            self.summary.add_histogram(name.replace('.', '/'), param.clone().cpu().data.numpy(), iteration)

        # feature map visualize
        for trainId in range(logits.shape[0]):
            label_name = trainId2name[trainId]
            label_channel = logits[trainId].unsqueeze(dim=0).unsqueeze(dim=0)
            self.summary.add_image('origin_logits/%s' % label_name, vutils.make_grid(label_channel, normalize=True),
                                   iteration)

        # visualize foreground's embedding
        C, H, W = logits.shape
        flatten_label = label.view(-1).numpy()
        self.summary.add_embedding(
            mat = logits.permute(1, 2, 0).view(-1, C)[flatten_label != 0],
            metadata = list(map(lambda x : trainId2name[x], flatten_label[[flatten_label != 0]])),
            global_step = iteration
        )