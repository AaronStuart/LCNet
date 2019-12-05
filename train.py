import argparse
import os

import numpy as np
import torch
import torchvision
import visdom
from torch import optim
from torch.utils.data import DataLoader

from dataset.bdd100k import BDD100K
from loss.focal_loss import FocalLoss
from model.EDANet import EDANet
from model.EDA_DDB import EDA_DDB

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs", type = int, default = 100)
    parser.add_argument("--batch_size", type = int, default = 5)
    parser.add_argument("--learning_rate", type = float, default = 0.001)
    parser.add_argument("--num_threads", type = int, default = 1)
    parser.add_argument("--foreground_threshold", type=float, default=0.6,
                        help = "If the predicted probability exceeds this threshold, it will be judged as the foreground.")
    parser.add_argument("--checkpoint_interval", type = int, default = 1, help = "How many epochs are saved once?")
    parser.add_argument("--evaluation_interval", type = int, default = 1, help = "How many epochs are evaluated once?")
    parser.add_argument("--visualize_interval", type=int, default=100, help = "How many iterations are visualized once?")
    parser.add_argument("--pretrained_weights", type=str,
                        default='/home/stuart/PycharmProjects/EDANet/weights/EDANet/EDANet_epoch_4_iter_4000.pth')
    parser.add_argument("--train_file", type = str,
                        default = './dataset/train_bdd100k.txt')
    parser.add_argument("--val_file", type = str,
                        default = './dataset/val_bdd100k.txt')
    args = parser.parse_args()
    print(args)

    # Visualize
    viz = visdom.Visdom()
    train_loss_win = viz.line(
        Y=np.array([0]),
        X=np.array([0]),
    )
    train_IoU_win = viz.line(
        Y=np.array([0]),
        X=np.array([0]),
    )
    train_input_win = viz.images(
        np.random.randn(args.batch_size, 3, 512, 1024),
        nrow = args.batch_size,
        opts = dict(caption = 'train_input')
    )
    train_label_win = viz.images(
        np.random.randn(args.batch_size, 1, 512, 1024),
        nrow = args.batch_size,
        opts = dict(caption = 'train_label')
    )
    train_predict_win = viz.images(
        np.random.randn(args.batch_size, 1, 512, 1024),
        nrow = args.batch_size,
        opts = dict(caption = 'train_predict')
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial model
    model = EDANet(num_classes = args.num_classes, init_weights = True, device = device).to(device)
    # model = EDA_DDB(num_classes=2, init_weights=True).to(device)

    # Start from checkpoints if specified
    if args.pretrained_weights:
        model.load_state_dict(torch.load(args.pretrained_weights))

    # Get dataloader
    # trainset = torchvision.datasets.Cityscapes('/media/stuart/data/dataset/cityscapes')
    trainset = BDD100K(args.train_file)
    trainloader = DataLoader(
        trainset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_threads,
        pin_memory = True,
        drop_last = True
    )

    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)

    model.train()
    for epoch in range(args.epochs):
        for iter, data in enumerate(trainloader):
            ##############################
            #######  GET DATA  ###########
            ##############################
            input, label = data['image'].to(device), data['label'].to(device)

            ##############################
            #######  TRAIN MODEL  ########
            ##############################
            optimizer.zero_grad()
            # forward
            output = model(input, label)
            # compute loss
            loss = FocalLoss(args.num_classes, device=device)(output, label)
            # backward
            loss.backward()
            optimizer.step()

            log_str = "Epoch %d/%d, iter %d/%d, loss = %f" % (epoch, args.epochs, iter, len(trainloader), loss)
            print(log_str)

            ##############################
            #####  POST PROGRESS   #######
            ##############################
            output = torch.where(
                output[:, 1, :, :] > args.foreground_threshold,
                torch.ones_like(output[:, 1, :, :]),
                torch.zeros_like(output[:, 1, :, :])
            ).float().unsqueeze(1)

            ##############################
            #####  CALCULATE mIoU   ######
            ##############################
            i = (label * output).sum()
            u = (label + output - label * output).sum()
            IoU = i / u if u != 0 else u

            ##############################
            #######  VISUALIZE   #########
            ##############################
            if iter != 0 and iter % args.visualize_interval == 0:
                viz.line(
                    Y = np.array([loss.detach().cpu()]),
                    X = np.array([epoch * len(trainloader) + iter]),
                    win = train_loss_win,
                    name = 'train_loss',
                    update = 'append'
                )
                viz.line(
                    Y=np.array([IoU.detach().cpu()]),
                    X=np.array([epoch * len(trainloader) + iter]),
                    win=train_IoU_win,
                    name='IoU',
                    update='append'
                )
                viz.images(
                    input,
                    win = train_input_win,
                    nrow = args.batch_size
                )
                viz.images(
                    label,
                    win=train_label_win,
                    nrow = args.batch_size
                )
                viz.images(
                    output,
                    win = train_predict_win,
                    nrow = args.batch_size
                )
        # Save model
        if epoch != 0 and epoch % args.checkpoint_interval == 0:
            os.makedirs("weights/%s" % (model.__class__.__name__), exist_ok=True)
            save_path = 'weights/%s/epoch_%d_iter_%d.pth' % (model.__class__.__name__, epoch, iter)
            torch.save(model.state_dict(), save_path)







