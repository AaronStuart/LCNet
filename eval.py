import argparse
import json

import torch
import torchvision
from torch.utils.data import DataLoader

from dataset.apollo import ApolloLaneDataset
from evaluate.evaluation import EvaluationOnDataset
from model.UNet import UNet

parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, default=38)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_threads", type=int, default=8)
parser.add_argument("--pretrained_weights", type=str, default='/home/stuart/PycharmProjects/LCNet/weights/DeepLabV3/epoch_0_iter_27000.pth')
parser.add_argument("--val_file", type=str, default='./dataset/val_apollo.txt')
args = parser.parse_args()




if __name__ == '__main__':
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get model
    # model = UNet(in_channels=3, num_classes=args.num_classes, bilinear=True, init_weights=True)
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=args.num_classes).to(device)

    # Load checkpoint
    model.load_state_dict(torch.load(args.pretrained_weights))
    print("load", args.pretrained_weights, "successfully.")

    # Get dataloader
    val_dataset = ApolloLaneDataset(
        root_dir = "/media/stuart/data/dataset/Apollo/Lane_Detection",
        path_file = args.val_file,
        is_train = False
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )

    # Evaluate on dataset
    eval = EvaluationOnDataset(model=model, device=device, dataloader=val_dataloader)
    result = eval.evaluate()


    # output result to json file
    with open('%s.json' % model.__class__.__name__, 'w') as json_file:
        json.dump(result, json_file, indent=4)
