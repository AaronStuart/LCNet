import numpy as np
import torch
import visdom

from scripts.apollo_label import trainId2color


class TrainVisualize:
    def __init__(self, model_name, batch_size, image_height, image_width, use_boundary_loss, use_metric_loss):
        self.model_name = model_name
        self.use_boundary_loss = use_boundary_loss
        self.use_metric_loss = use_metric_loss
        self.viz = visdom.Visdom(env = model_name)
        self.windows = self.init_train_visualize(batch_size, image_height, image_width)

    def init_train_visualize(self, batch_size, image_height, image_width):
        windows = {}

        total_loss_win = self.viz.line(
            Y=np.array([0]),
            X=np.array([0]),
        )
        windows['total_loss'] = total_loss_win

        focal_loss_win = self.viz.line(
            Y=np.array([0]),
            X=np.array([0]),
        )
        windows['focal_loss'] = focal_loss_win

        if self.use_boundary_loss:
            boundary_loss_win = self.viz.line(
                Y=np.array([0]),
                X=np.array([0]),
            )
            windows['boundary_loss'] = boundary_loss_win

        if self.use_metric_loss:
            metric_loss_win = self.viz.line(
                Y=np.array([0]),
                X=np.array([0]),
            )
            windows['metric_loss'] = metric_loss_win
        # image format should be RGB
        input_win = self.viz.images(
            np.random.randn(batch_size, 3, image_height, image_width),
            opts=dict(caption='input')
        )
        windows['input'] = input_win

        label_win = self.viz.images(
            np.random.randn(batch_size, 3, image_height, image_width),
            opts=dict(caption='label')
        )
        windows['label'] = label_win

        predict_win = self.viz.images(
            np.random.randn(batch_size, 3, image_height, image_width),
            opts=dict(caption='predict')
        )
        windows['predict'] = predict_win

        return windows

    def update(self, iteration, input, predict, label, loss):
        self.viz.line(
            Y=np.array([loss['total_loss'].detach().cpu()]),
            X=np.array([iteration]),
            win=self.windows['total_loss'],
            name='total_loss',
            update='append'
        )

        self.viz.line(
            Y=np.array([loss['focal_loss'].detach().cpu()]),
            X=np.array([iteration]),
            win=self.windows['focal_loss'],
            name='focal_loss',
            update='append'
        )

        if self.use_boundary_loss:
            self.viz.line(
                Y=np.array([loss['boundary_loss'].detach().cpu()]),
                X=np.array([iteration]),
                win=self.windows['boundary_loss'],
                name='boundary_loss',
                update='append'
            )

        if self.use_metric_loss:
            self.viz.line(
                Y=np.array([loss['metric_loss'].detach().cpu()]),
                X=np.array([iteration]),
                win=self.windows['metric_loss'],
                name='metric_loss',
                update='append'
            )

        self.viz.images(
            input,
            win=self.windows['input']
        )

        self.viz.images(
            label,
            win=self.windows['label']
        )

        output_grayscale = torch.argmax(predict, axis=1)
        # map trainId to color
        predict_image = torch.zeros_like(label)
        for trainId, rgb in trainId2color.items():
            mask = (output_grayscale == trainId)
            predict_image[:, 0, :, :][mask] = rgb[0]
            predict_image[:, 1, :, :][mask] = rgb[1]
            predict_image[:, 2, :, :][mask] = rgb[2]
        self.viz.images(
            predict_image,
            win=self.windows['predict']
        )

    def save(self):
        self.viz.save(envs = [self.model_name])