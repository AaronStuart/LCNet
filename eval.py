import argparse
import json
import os

import cupy as cp
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.apollo import ApolloDataset
from scripts.apollo_label import labels, trainId2name
from scripts.visualize import Visualize

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='DeepLabV3')
parser.add_argument("--num_classes", type=int, default=38)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_threads", type=int, default=1)
parser.add_argument("--ckpt_path", type=str,
                    default='/media/stuart/data/weights/DeepLabV3/cluster_loss_iter_100000.pth')
parser.add_argument("--dataset_root_dir", type=str, default='/media/stuart/data/dataset/Apollo/Lane_Detection')
parser.add_argument("--val_file", type=str, default='dataset/apollo_val_gray.txt')
parser.add_argument("--output_dir", type=str, default='/media/stuart/data')
parser.add_argument("--use_dali", type=bool, default=False)
args = parser.parse_args()


class Evaluation(object):
    def __init__(self, model_name, num_classes, ckpt_path):
        super(Evaluation, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.get_model(model_name, num_classes).to(self.device).eval()
        self.load_checkpoint(ckpt_path)

        self.val_dataloader = self.get_dataloader(
            dataset_dir=args.dataset_root_dir,
            file_path=args.val_file,
            batch_size=1,
            num_threads = args.num_threads,
            is_train=False
        )

        self.eval_trainIds = self.getEvalTrainIds()
        self.final_result = {}

        self.visualizer = Visualize(
            log_dir=os.path.join(args.output_dir, 'events', self.model.__class__.__name__),
            model=self.model
        )

    def get_model(self, model_name, num_classes):
        """"""
        assert model_name in ['DeepLabV3', 'FCN'], "model name doesn't exist"

        if model_name == 'DeepLabV3':
            model = torchvision.models.segmentation.deeplabv3_resnet50(
                pretrained=False,
                num_classes=num_classes
            )
        if model_name == 'FCN':
            model = torchvision.models.segmentation.fcn_resnet50(
                pretrained=False,
                num_classes=num_classes
            )

        return model

    def get_dataloader(self, dataset_dir, file_path, batch_size=1, num_threads=1, is_train=False):
        """"""
        if not os.path.exists(dataset_dir):
            print("%s doesn't exist." % dataset_dir)
            return
        if not os.path.exists(file_path):
            print("%s doesn't exist." % file_path)
            return

        dataset = ApolloDataset(
            root_dir=dataset_dir,
            path_file=file_path,
            is_train=is_train
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_threads,
            pin_memory=True,
            drop_last=False
        )

        return dataloader

    def load_checkpoint(self, ckpt_path):
        """"""
        if not os.path.exists(ckpt_path):
            print("%s not exist." % ckpt_path)
            exit()

        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print("Load from %s successfully." % ckpt_path)

    def getEvalTrainIds(self):
        result = []
        for label in labels:
            if label.ignoreInEval == True:
                continue
            result.append(label.trainId)
        return result

    def accumulateOnBatch(self, predict, label):
        predict = cp.asarray(predict, dtype=cp.uint8)
        label = cp.asarray(label, dtype=cp.uint8)

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
            iter = 0
            for data in tqdm(self.val_dataloader):
                input, label = data['image'].to(self.device), data['label'].to(self.device)

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

                # self.visualizer.eval_update(
                #     iteration=iter,
                #     input=input.cpu(),
                #     label=label.cpu(),
                #     predict=predict.cpu()
                # )

                # accumulate on batch
                self.accumulateOnBatch(predict.cpu().numpy(), label.cpu().numpy())

                iter += 1

        # calculate IoU for each class
        for class_name, class_dict in self.final_result.items():
            class_tp = class_dict['TP']
            class_fp = class_dict['FP']
            class_fn = class_dict['FN']
            if class_tp + class_fp + class_fn != 0:
                class_IoU = class_tp / (class_tp + class_fp + class_fn)
            else:
                class_IoU = 0.0
            self.final_result[class_name]['IoU'] = class_IoU

        return self.final_result


if __name__ == '__main__':
    # Evaluate on val set
    eval_reuslt = Evaluation(
        model_name=args.model_name,
        num_classes=args.num_classes,
        ckpt_path=args.ckpt_path
    ).eval()

    print(eval_reuslt)

    # Save eval result to disk
    model_name = args.model_name
    with open('experiments/%s/cluster_loss_iter_100000.json' % model_name,
              'w') as result_file:
        json.dump(eval_reuslt, result_file, indent=4)
