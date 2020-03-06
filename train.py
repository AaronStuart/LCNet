import argparse
import os

import torch
import torchvision
from line_profiler import LineProfiler
from torch import optim
from torch.utils.data import DataLoader

from dataset.apollo import ApolloDataset
from loss.LossFactory import LossFactory
from loss.focal_loss import FocalLoss
from model.UNet import UNet
from postprocessing.PostProcesing import PostProcessing
from scripts.visualize_train import TrainVisualize

parser = argparse.ArgumentParser()
#############  Model  #############
parser.add_argument("--num_classes", type=int, default=38)

#############  Data  #############
parser.add_argument("--dataset_root_dir", type=str, default="/media/stuart/data/dataset/Apollo/Lane_Detection")
parser.add_argument("--train_file", type=str, default='./dataset/train_apollo_gray.txt')
parser.add_argument("--num_threads", type=int, default=1)

#############  Loss  #############
parser.add_argument("--use_boundary_loss", type=bool, default=False)
parser.add_argument("--boundary_loss_weight", type=float, default=1)

parser.add_argument("--use_metric_loss", type=bool, default=False)
parser.add_argument("--metric_loss_weight", type=float, default=0.001)

############# Train  #############
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--warm_up_iters", type=int, default=0)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--pretrained_weights", type=str)
parser.add_argument("--save_interval", type=int, default=10, help="How many iterations are saved once?")
parser.add_argument("--visualize_interval", type=int, default=100, help="How many iterations are visualized once?")

args = parser.parse_args()
print(args)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial model
    # model = UNet(in_channels=3, num_classes=args.num_classes, bilinear=True, init_weights=True).to(device)
    # model = torchvision.models.segmentation.fcn_resnet50(num_classes=args.num_classes).to(device)
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        pretrained = False,
        num_classes = args.num_classes
    ).to(device).train()

    train_visualizer = TrainVisualize(
        log_dir=os.path.join('/media/stuart/data/events', model.__class__.__name__),
        model=model,
        use_boundary_loss=False,
        use_metric_loss=False
    )

    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    # Start from checkpoints if specified
    restart_epoch, restart_iter = 0, 0
    if args.pretrained_weights:
        # load checkpoint file
        checkpoint = torch.load(args.pretrained_weights)

        # restore
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        restart_epoch = checkpoint['epoch']
        restart_iter = checkpoint['iteration']

        print("Load %s successfully." % args.pretrained_weights)

    # Define dataloader
    trainloader = ApolloDataset(
        root_dir = args.dataset_root_dir,
        file_path = args.train_file,
        batch_size = args.batch_size,
        num_threads = args.num_threads
    ).getIterator()

    for epoch in range(restart_epoch, args.epochs):
        for iter, data in enumerate(trainloader, restart_iter):
            # train model
            optimizer.zero_grad()

            # get model output
            logits = model(data[0]['data'].to(device))['out'].cpu()

            # Loss function depend on iter
            loss = LossFactory(
                cur_iter=iter, warm_up_iters=args.warm_up_iters,
                num_classes=args.num_classes,
                use_boundary_loss=args.use_boundary_loss, boundary_loss_weight=args.boundary_loss_weight,
                use_metric_loss=args.use_metric_loss, metric_loss_weight=args.metric_loss_weight
            ).compute_loss(logits.cpu(), data[0]['label'].cpu())

            # update weights
            loss['total_loss'].backward()
            optimizer.step()
            print(loss)

            # visualize train process
            if iter != 0 and iter % args.visualize_interval == 0:
                train_visualizer.update(
                    iteration=iter,
                    input=data[0]['data'][0].cpu(),
                    label=data[0]['label'][0].cpu(),
                    logits=logits[0].detach().cpu(),
                    loss=loss
                )

            # save checkpoint
            if iter != 0 and iter % args.save_interval == 0:
                save_path = "weights/%s" % model.__class__.__name__
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'iteration': iter
                    },
                    os.path.join(save_path, '%s_epoch_%d_iter_%d.pth' % (model.__class__.__name__, epoch, iter))
                )
                print('Finish save checkpoint.')

if __name__ == '__main__':
    lp = LineProfiler()
    lp.add_function(LossFactory.compute_loss)
    lp.add_function(FocalLoss.compute_focal_loss)
    lp.add_function(TrainVisualize.update)
    lp_wrapper = lp(main)
    lp_wrapper()
    lp.print_stats()
