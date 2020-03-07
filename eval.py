import argparse
import json

import torch

from dataset.apollo import ApolloDaliDataset
from evaluate.evaluation import Evaluation

parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, default=38)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_threads", type=int, default=1)
parser.add_argument("--model_path", type=str)
parser.add_argument("--val_file", type=str, default='./dataset/val_apollo_gray.txt')
args = parser.parse_args()

if __name__ == '__main__':
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = torch.load(args.model_path).to(device)

    # Get validation iterator
    val_iterator = ApolloDaliDataset(
        root_dir=args.dataset_root_dir,
        file_path=args.val_file,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        is_train=False
    ).getIterator()

    # Evaluate on val set
    eval_reuslt = Evaluation(
        model=model,
        device=device,
        dataloader=val_iterator
    ).eval()

    # Save eval result to disk
    model_name = model.__class__.__name__
    with open('experiments/%s/%s.json' % (model_name, model_name), 'w') as result_file:
        json.dump(eval_reuslt, result_file, indent=4)
