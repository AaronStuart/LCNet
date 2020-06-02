import os
from datetime import datetime

import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from scripts.apollo_label import trainId2color, trainId2name


class Visualize:
    def __init__(self, log_dir, model):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.model = model
        self.summary = SummaryWriter(
            logdir=os.path.join(log_dir, "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now()))
        )

    def train_update(self, iteration, learning_rate, input, label, logits, loss):
        predict = torch.argmax(logits, axis=1)
        N, _, H, W = label.shape

        # map predict to color
        predict_color = torch.zeros(size=[N, 3, H, W], dtype=torch.uint8)
        for trainId, rgb in trainId2color.items():
            mask = (predict == trainId)
            predict_color[:, 0, :, :][mask] = rgb[0]
            predict_color[:, 1, :, :][mask] = rgb[1]
            predict_color[:, 2, :, :][mask] = rgb[2]

        # map label to color
        label_color = torch.zeros(size=[N, 3, H, W], dtype=torch.uint8)
        for trainId, rgb in trainId2color.items():
            mask = (label.squeeze() == trainId)
            label_color[:, 0, :, :][mask] = rgb[0]
            label_color[:, 1, :, :][mask] = rgb[1]
            label_color[:, 2, :, :][mask] = rgb[2]

        # learning rate visualize
        if learning_rate:
            self.summary.add_scalar('learning_rate', learning_rate, iteration)

        # loss visualize
        for key, value in loss.items():
            self.summary.add_scalar('loss/%s' % key, loss[key], iteration)

        # input visualize
        self.summary.add_image('image/input', vutils.make_grid(input.to(torch.uint8)), iteration)
        self.summary.add_image('image/label', vutils.make_grid(label_color), iteration)
        self.summary.add_image('image/predict', vutils.make_grid(predict_color), iteration)

        # visualize foreground's embedding
        N, C, H, W = logits.shape
        flatten_logits = logits.permute(0, 2, 3, 1).reshape(-1, C)
        flatten_labels = label.view(-1).numpy()

        foreground_mask = (flatten_labels != 0) & (flatten_labels != 37)
        foreground_logits = flatten_logits[foreground_mask]
        foreground_labels = flatten_labels[foreground_mask]

        self.summary.add_embedding(
            mat=foreground_logits,
            metadata=list(map(lambda x: trainId2name[x], foreground_labels)),
            global_step=iteration
        )

        # # weight visualize
        # for name, param in self.model.named_parameters():
        #     self.summary.add_histogram(name.replace('.', '/'), param.clone().cpu().data.numpy(), iteration)
        #
        # # feature map visualize
        # for trainId in range(logits.shape[0]):
        #     label_name = trainId2name[trainId]
        #     label_channel = logits[trainId].unsqueeze(dim=0).unsqueeze(dim=0)
        #     self.summary.add_image('origin_logits/%s' % label_name, vutils.make_grid(label_channel, normalize=True),
        #                            iteration)
        #
        # # add PR curve
        # for trainId in range(logits.shape[0]):
        #     self.summary.add_pr_curve(
        #         tag = trainId2name[trainId],
        #         labels = torch.where(label == trainId, torch.tensor(1), torch.tensor(0)).view(-1),
        #         predictions= torch.softmax(logits, dim = 0)[trainId].view(-1),
        #         global_step = iteration
        #     )

    def eval_update(self, iteration, input, label, predict):
        N, _, H, W = label.shape

        # map predict to color
        predict_color = torch.zeros(size=[N, 3, H, W], dtype=torch.uint8)
        for trainId, rgb in trainId2color.items():
            mask = (predict == trainId)
            predict_color[:, 0, :, :][mask] = rgb[0]
            predict_color[:, 1, :, :][mask] = rgb[1]
            predict_color[:, 2, :, :][mask] = rgb[2]

        # map label to color
        label_color = torch.zeros(size=[N, 3, H, W], dtype=torch.uint8)
        for trainId, rgb in trainId2color.items():
            mask = (label.squeeze() == trainId)
            label_color[:, 0, :, :][mask] = rgb[0]
            label_color[:, 1, :, :][mask] = rgb[1]
            label_color[:, 2, :, :][mask] = rgb[2]

        # input visualize
        self.summary.add_image('image/input', vutils.make_grid(input.to(torch.uint8)), iteration)
        self.summary.add_image('image/label', vutils.make_grid(label_color), iteration)
        self.summary.add_image('image/predict', vutils.make_grid(predict_color), iteration)