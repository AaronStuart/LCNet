import argparse

import torch
from torch.utils.data import DataLoader

from dataset.apollo import ApolloLaneDataset
from model.UNet import UNet
from scripts.apollo_label import trainId2color

parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, default=38)
parser.add_argument("--epochs", type = int, default = 100)
parser.add_argument("--learning_rate", type = float, default = 0.001)
parser.add_argument("--num_threads", type = int, default = 8)
parser.add_argument("--foreground_threshold", type=float, default=0.6,
                    help = "If the predicted probability exceeds this threshold, it will be judged as the foreground.")
parser.add_argument("--pretrained_weights", type=str)
parser.add_argument("--val_file", type = str,
                    default = './dataset/val_bdd100k.txt')
parser.add_argument("--output_dir", type=str, default="./predicts")
args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial model
    model = UNet(in_channels=3, num_classes=args.num_classes, bilinear=True, init_weights=True).to(device)
    model.eval()

    # Load checkpoint file
    model.load_state_dict(torch.load(args.pretrained_weights))
    print("load", args.pretrained_weights, "successfully.")

    # Define dataloader
    valset = ApolloLaneDataset(args.val_file)
    valloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_threads,
        pin_memory=True,
        drop_last=False
    )


    for iter, data in enumerate(valloader):
        ##############################
        #######  GET DATA  ###########
        ##############################
        with open(args.val_file, 'r') as val:
            for line in val.readlines():
                input_path, label_path = line.strip().split()
                predict_path = label_path.split('')


        ##############################
        #######  EVAL MODEL  ########
        ##############################
        # forward
        output = model(input.to(device)).cpu()

        # change to trainId
        output_trainId = torch.argmax(output, axis=1)

        # map trainId to color
        output_bgr = torch.zeros_like(label_bgr)
        for trainId, rgb in trainId2color.items():
            bgr = rgb[::-1]
            mask = output_trainId == trainId
            output_bgr[:, 0, :, :][mask] = bgr[0]
            output_bgr[:, 1, :, :][mask] = bgr[1]
            output_bgr[:, 2, :, :][mask] = bgr[2]

        # save to file
