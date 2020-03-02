import argparse
import os

import torch
import torchvision
from torch import optim
from torch.utils.data import DataLoader

from dataset.apollo import ApolloLaneDataset
from loss.focal_loss import FocalLoss
from scripts.visualize_train import TrainVisualize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=38)
    parser.add_argument("--epochs", type = int, default = 100)
    parser.add_argument("--batch_size", type = int, default = 4)
    parser.add_argument("--learning_rate", type = float, default = 0.001)
    # parser.add_argument("--num_threads", type = int, default = 8)
    parser.add_argument("--pretrained_weights", type=str)
    parser.add_argument("--checkpoint_interval", type = int, default = 1000, help = "How many iterations are saved once?")
    parser.add_argument("--visualize_interval", type=int, default=100, help = "How many iterations are visualized once?")
    parser.add_argument("--train_file", type = str, default = './dataset/train_apollo.txt')
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial model
    # model = EDANet(num_classes = args.num_classes, init_weights = True).to(device)
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained = False, num_classes = args.num_classes).to(device)
    # model = UNet(in_channels = 3, num_classes = args.num_classes, bilinear = True, init_weights=True).to(device)

    # Visualize
    train_visualizer = TrainVisualize(
        log_dir=os.path.join('/media/stuart/data/events', model.__class__.__name__),
        model=model,
        use_boundary_loss=False,
        use_metric_loss=True
    )

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
        pin_memory = True,
        drop_last = True
    )

    model.train()
    for epoch in range(begin_epoch, args.epochs):
        for iter, data in enumerate(trainloader, begin_iter):
            # get data
            input, label_trainId, labe_rgb = data['input'], data['label_trainId'], data['labe_rgb']

            # train model
            optimizer.zero_grad()
            logits = model(input.to(device))['out'].cpu()
            loss = focal_loss(torch.softmax(logits, dim = 1), label_trainId)

            # print log
            log_str = "Epoch %d/%d, iter %d/%d, loss = %f" % (epoch, args.epochs, iter, len(trainloader), loss)
            print(log_str)

            # update weights
            loss.backward()
            optimizer.step()

            # visualize train process
            if iter % args.visualize_interval == 0:
                train_visualizer.update(
                    iteration=iter,
                    input=data['input'],
                    label=data['labe_rgb'],
                    logits=logits,
                    loss=loss
                )

            # Save model
            if (epoch * len(trainloader) + iter) != 0 and (epoch * len(trainloader) + iter) % args.checkpoint_interval == 0:
                os.makedirs("weights/%s" % (model.__class__.__name__), exist_ok=True)
                save_path = 'weights/%s/epoch_%d_iter_%d.pth' % (model.__class__.__name__, epoch, iter)
                torch.save(model.state_dict(), save_path)
                print('Save to', save_path, "successfully.")







