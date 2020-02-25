import argparse
import os

import torch
from torch import optim
from torch.utils.data import DataLoader

from dataset.apollo import ApolloLaneDataset
from loss.LossFactory import LossFactory
from model.UNet import UNet
from postprocessing.PostProcesing import PostProcessing
from scripts.visualize_train import TrainVisualize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_boundary_loss", type = bool, default = False)
    parser.add_argument("--boundary_loss_weight", type = float, default = 1)
    parser.add_argument("--use_metric_loss", type = bool, default = False)
    parser.add_argument("--metric_loss_weight", type = float, default = 0.001)
    parser.add_argument("--num_classes", type = int, default = 38)
    parser.add_argument("--epochs", type = int, default = 1)
    parser.add_argument("--warm_up_iters", type = int, default = 0)
    parser.add_argument("--batch_size", type = int, default = 1)
    parser.add_argument("--learning_rate", type = float, default = 0.001)
    parser.add_argument("--num_threads", type = int, default = 4)
    parser.add_argument("--pretrained_weights", type=str)
    parser.add_argument("--checkpoint_interval", type = int, default = 1000, help = "How many iterations are saved once?")
    parser.add_argument("--visualize_interval", type = int, default = 2, help = "How many iterations are visualized once?")
    parser.add_argument("--dataset_root_dir", type = str, default = "/media/stuart/data/dataset/Apollo/Lane_Detection")
    parser.add_argument("--train_file", type = str, default = './dataset/train_apollo.txt')
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial model
    model = UNet(in_channels = 3, num_classes = args.num_classes, bilinear = True, init_weights=True).to(device)
    # model = torchvision.models.segmentation.fcn_resnet50(num_classes=args.num_classes).to(device)

    # add train visualize
    train_visualize = TrainVisualize(
        model.__class__.__name__,
        args.batch_size, 2710, 3384,
        args.use_boundary_loss,
        args.use_metric_loss
    )

    # Define loss and optimizer

    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)

    # Define post process
    post_process = PostProcessing()

    # Start from checkpoints if specified
    begin_epoch, begin_iter = 0, 0
    if args.pretrained_weights:
        model.load_state_dict(torch.load(args.pretrained_weights))
        print("load", args.pretrained_weights, "successfully.")
        begin_epoch = int(args.pretrained_weights.split('_')[1])
        begin_iter = int(args.pretrained_weights.split('_')[-1].split('.')[0]) + 1

    # Define dataloader
    trainset = ApolloLaneDataset(
        root_dir = args.dataset_root_dir,
        path_file = args.train_file
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
            # train model
            optimizer.zero_grad()

            # get model output
            logits = model(data['input'].to(device))['out'].cpu()

            # post process
            logits = post_process.forward(logits, data['label_trainId'])

            # Loss function depend on iter
            loss = LossFactory(
                cur_iter = iter, warm_up_iters = args.warm_up_iters,
                num_classes = args.num_classes,
                use_boundary_loss = args.use_boundary_loss, boundary_loss_weight = args.boundary_loss_weight,
                use_metric_loss = args.use_metric_loss, metric_loss_weight = args.metric_loss_weight
            ).compute_loss(logits, data['label_trainId'])

            # update weights
            loss['weighted_loss'].backward()
            optimizer.step()
            print(loss)

            # visualize train process
            if iter % args.visualize_interval == 0:
                train_visualize.update(
                    epoch * len(trainloader) + iter,
                    data['origin_image'],
                    torch.nn.functional.softmax(logits, dim = 1),
                    data['origin_label'],
                    loss
                )

            if (epoch * len(trainloader) + iter) != 0 and (epoch * len(trainloader) + iter) % args.checkpoint_interval == 0:
                # save checkpoint
                os.makedirs("weights/%s" % (model.__class__.__name__), exist_ok=True)
                save_path = 'weights/%s/epoch_%d_iter_%d.pth' % (model.__class__.__name__, epoch, iter)
                torch.save(model.state_dict(), save_path)
                print('Save to', save_path, "successfully.")

                # save visualize
                os.makedirs("output/%s" % (model.__class__.__name__), exist_ok=True)
                train_visualize.save()
                print("Save visdom environment successfully.")








