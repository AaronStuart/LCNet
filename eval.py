import argparse
import json
import pickle

import cupy as cp
import torch
from tqdm import tqdm

from dataset.apollo import ApolloDaliDataset
from scripts.apollo_label import labels, trainId2name

parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, default=38)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_threads", type=int, default=1)
parser.add_argument("--model_path", type=str, default='/home/stuart/PycharmProjects/LCNet/weights/DeepLabV3/model_cluster_iter_100000_pretrained.pth')
parser.add_argument("--dataset_root_dir", type=str, default='/media/stuart/data/dataset/Apollo/Lane_Detection')
parser.add_argument("--val_file", type=str, default='./dataset/val_apollo_gray.txt')
args = parser.parse_args()

class Evaluation(object):
    def __init__(self, model, device, dataloader):
        super(Evaluation, self).__init__()
        self.device = device
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.eval_trainIds = self.getEvalTrainIds()
        self.final_result = {}

    def getEvalTrainIds(self):
        result = []
        for label in labels:
            if label.ignoreInEval == True:
                continue
            result.append(label.trainId)
        return result

    def accumulateOnBatch(self, predict, label):
        predict = cp.asarray(predict, dtype = cp.uint8)
        label = cp.asarray(label, dtype = cp.uint8)

        for train_id in self.eval_trainIds:
            # get mask
            predict_mask = (predict == train_id)
            label_mask = (label == train_id)

            # calculate TP FP FN
            TP = cp.sum(label_mask * predict_mask).item()
            FP = cp.sum((~label_mask) * predict_mask).item()
            FN = cp.sum(label_mask * (~predict_mask)).item()

            label_name = trainId2name[train_id]
            # init class, if does not exist
            if label_name not in self.final_result.keys():
                self.final_result[label_name] = {}
                self.final_result[label_name]['TP'] = 0.0
                self.final_result[label_name]['FP'] = 0.0
                self.final_result[label_name]['FN'] = 0.0

            # accumulate result
            self.final_result[label_name]['TP'] += TP
            self.final_result[label_name]['FP'] += FP
            self.final_result[label_name]['FN'] += FN

    def eval(self):
        with torch.no_grad():
            for data in tqdm(self.dataloader):
                # Note: DALI's RGB image have changed to BGR sequence, after go through Pytorch, maybe a bug
                input, label = data[0]['input'][:, [2, 1, 0], :, :].to(self.device), data[0]['label'].to(self.device)

                # forward
                logits = self.model(input)['out']

                # get predict
                predict = torch.argmax(
                    torch.nn.functional.softmax(logits, dim=1),
                    axis=1,
                    keepdim=True
                ).to(torch.float)

                # resize predict to label's size
                predict = torch.nn.functional.interpolate(
                    predict,
                    label.shape[-2:],
                    mode='nearest'
                ).to(torch.uint8)

                # accumulate on batch
                self.accumulateOnBatch(predict.cpu().numpy(), label.cpu().numpy())

        # calculate IoU for each class
        for class_name, class_dict in self.final_result.items():
            class_tp = class_dict['TP']
            class_fp = class_dict['FP']
            class_fn = class_dict['FN']
            class_IoU = class_tp / (class_tp + class_fp + class_fn)
            self.final_result[class_name]['IoU'] = class_IoU

        return self.final_result


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

    print(eval_reuslt)

    # Save eval result to disk
    model_name = model.__class__.__name__
    with open('experiments/%s/cluster_100000_iter_pretrained.json' % model_name, 'w') as result_file:
        json.dump(eval_reuslt, result_file, indent=4)
