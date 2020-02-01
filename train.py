import argparse
import os

import numpy as np
import torch
import torchvision
import visdom
from torch import optim
from torch.utils.data import DataLoader

from dataset.apollo import ApolloLaneDataset
from loss.focal_loss import FocalLoss
from model.UNet import UNet
from scripts.apollo_label import trainId2color, trainIdsOfLanes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=38)
    parser.add_argument("--epochs", type = int, default = 1)
    parser.add_argument("--batch_size", type = int, default = 1)
    parser.add_argument("--learning_rate", type = float, default = 0.001)
    parser.add_argument("--num_threads", type = int, default = 8)
    parser.add_argument("--pretrained_weights", type=str)
    parser.add_argument("--checkpoint_interval", type = int, default = 1000, help = "How many iterations are saved once?")
    parser.add_argument("--visualize_interval", type=int, default=100, help = "How many iterations are visualized once?")
    parser.add_argument("--train_file", type = str, default = './dataset/train_apollo.txt')
    args = parser.parse_args()
    print(args)

    # Visualize
    viz = visdom.Visdom()
    train_loss_win = viz.line(
        Y=np.array([0]),
        X=np.array([0]),
    )
    train_input_win = viz.images(
        np.random.randn(args.batch_size, 3, 512, 1024),
        opts = dict(caption = 'train_input')
    )
    train_label_win = viz.images(
        np.random.randn(args.batch_size, 3, 512, 1024),
        opts = dict(caption = 'train_label')
    )
    train_predict_win = viz.images(
        np.random.randn(args.batch_size, 1, 512, 1024),
        nrow = args.batch_size,
        opts = dict(caption = 'train_predict')
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial model
    # model = UNet(in_channels = 3, num_classes = args.num_classes, bilinear = True, init_weights=True).to(device)
    model = torchvision.models.segmentation.fcn_resnet50(num_classes=args.num_classes).to(device)

    # Define loss and optimizer
    focal_loss = FocalLoss(num_classes=args.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Start from checkpoints if specified
    begin_epoch, begin_iter = 0, 0
    if args.pretrained_weights:
        model.load_state_dict(torch.load(args.pretrained_weights))
        print("load", args.pretrained_weights, "successfully.")
        begin_epoch = int(args.pretrained_weights.split('_')[1])
        begin_iter = int(args.pretrained_weights.split('_')[-1].split('.')[0]) + 1

    # Define dataloader
    trainset = ApolloLaneDataset(
        root_dir = "/media/stuart/data/dataset/Apollo/Lane_Detection",
        path_file = args.train_file,
        is_train=True
    )
    trainloader = DataLoader(
        trainset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_threads,
        pin_memory = True,
        drop_last = True
    )

    model.train()
    for epoch in range(begin_epoch, args.epochs):
        for iter, data in enumerate(trainloader, begin_iter):
            # get data
            input, label_trainId, label_bgr = data['input'], data['label_trainId'], data['label_bgr']

            # train model
            optimizer.zero_grad()
            output = torch.nn.functional.softmax(model(input.to(device))['out'], dim=1).cpu()
            loss = focal_loss(output, label_trainId)

            # print log
            log_str = "Epoch %d/%d, iter %d/%d, loss = %f" % (epoch, args.epochs, iter, len(trainloader), loss)
            print(log_str)

            # update weights
            loss.backward()
            optimizer.step()

            # visualize train process
            if iter % args.visualize_interval == 0:
                output_grayscale = torch.argmax(output, axis=1)

                # map trainId to color
                output_bgr = torch.zeros_like(label_bgr)
                for trainId, rgb in trainId2color.items():
                    bgr = rgb[::-1]
                    mask = output_grayscale == trainId
                    output_bgr[:, 0, :, :][mask] = bgr[0]
                    output_bgr[:, 1, :, :][mask] = bgr[1]
                    output_bgr[:, 2, :, :][mask] = bgr[2]

                viz.line(
                    Y = np.array([loss.detach().cpu()]),
                    X = np.array([epoch * len(trainloader) + iter]),
                    win = train_loss_win,
                    name = 'train_loss',
                    update = 'append'
                )
                viz.images(
                    input,
                    win = train_input_win
                )
                viz.images(
                    label_bgr,
                    win=train_label_win
                )
                viz.images(
                    output_bgr,
                    win = train_predict_win
                )

            # Save model
            if (epoch * len(trainloader) + iter) != 0 and (epoch * len(trainloader) + iter) % args.checkpoint_interval == 0:
                os.makedirs("weights/%s" % (model.__class__.__name__), exist_ok=True)
                save_path = 'weights/%s/epoch_%d_iter_%d.pth' % (model.__class__.__name__, epoch, iter)
                torch.save(model.state_dict(), save_path)
                print('Save to', save_path, "successfully.")
    viz.save()







