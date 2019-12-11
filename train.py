import argparse
import os

import numpy as np
import torch
import torchvision
import visdom
from torch import optim
from torch.utils.data import DataLoader

from dataset.apollo import ApolloLaneDataset
from dataset.bdd100k import BDD100K
from evaluate.evaluation import mIoU
from loss.focal_loss import FocalLoss
from model.EDANet import EDANet
from model.EDA_DDB import EDA_DDB
from model.UNet import UNet
from scripts.apollo_label import trainId2color, trainIdsOfLanes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=38)
    parser.add_argument("--epochs", type = int, default = 100)
    parser.add_argument("--batch_size", type = int, default = 1)
    parser.add_argument("--learning_rate", type = float, default = 0.001)
    parser.add_argument("--num_threads", type = int, default = 8)
    parser.add_argument("--foreground_threshold", type=float, default=0.6,
                        help = "If the predicted probability exceeds this threshold, it will be judged as the foreground.")
    parser.add_argument("--checkpoint_interval", type = int, default = 1000, help = "How many iterations are saved once?")
    parser.add_argument("--evaluation_interval", type = int, default = 1, help = "How many epochs are evaluated once?")
    parser.add_argument("--visualize_interval", type=int, default=100, help = "How many iterations are visualized once?")
    parser.add_argument("--pretrained_weights", type=str)
    parser.add_argument("--train_file", type = str,
                        default = './dataset/train_apollo.txt')
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
    # model = EDANet(num_classes = args.num_classes, init_weights = True).to(device)
    model = UNet(in_channels = 3, num_classes = args.num_classes, bilinear = True, init_weights=True).to(device)

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
    trainset = ApolloLaneDataset(args.train_file)
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
            ##############################
            #######  GET DATA  ###########
            ##############################
            input, label_trainId, label_bgr = data['input'], data['label_trainId'], data['label_bgr']

            ##############################
            #######  TRAIN MODEL  ########
            ##############################
            optimizer.zero_grad()
            # forward
            output = model(input.to(device)).cpu()
            # compute loss
            loss = focal_loss(output, label_trainId)
            # backward
            loss.backward()
            optimizer.step()


            ##############################
            #####  CALCULATE mIoU   ######
            ##############################
            output_numpy = torch.argmax(output, axis=1, keepdim=True).numpy()
            label_numpy = label_trainId.numpy()

            result = mIoU(trainIdsOfLanes).evaluate(output_numpy, label_numpy)

            log_str = "Epoch %d/%d, iter %d/%d, loss = %f, mIoU = %f" % (epoch, args.epochs, iter, len(trainloader), loss, result['mIoU_of_batch'])
            print(log_str)

            ##############################
            #######  VISUALIZE   #########
            ##############################
            if iter != 0 and iter % args.visualize_interval == 0:
                # postprocess for visualize
                output_grayscale = torch.argmax(output, axis=1, keepdim=True)
                # map trainId to color
                output_bgr = torch.zeros_like(label_bgr)
                for batch in range(output_grayscale.shape[0]):
                    for row in range(output_grayscale.shape[2]):
                        for col in range(output_grayscale.shape[3]):
                            trainId = output_grayscale[batch, 0, row, col].item()
                            output_bgr[batch, :, row, col] = torch.tensor(trainId2color[trainId][::-1])

                viz.line(
                    Y = np.array([loss.detach().cpu()]),
                    X = np.array([epoch * len(trainloader) + iter]),
                    win = train_loss_win,
                    name = 'train_loss',
                    update = 'append'
                )
                viz.line(
                    Y=np.array([result['mIoU_of_batch']]),
                    X=np.array([epoch * len(trainloader) + iter]),
                    win=train_IoU_win,
                    name='IoU',
                    update='append'
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







