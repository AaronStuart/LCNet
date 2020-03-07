import cupy as cp
import torch
from tqdm import tqdm

from scripts.apollo_label import labels, trainId2name


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
            TP = cp.sum(label_mask * predict_mask)
            FP = cp.sum((~label_mask) * predict_mask)
            FN = cp.sum(label_mask * (~predict_mask))

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
        print('Totally %d batchs' % len(self.dataloader))

        with torch.no_grad():
            for iter, data in tqdm(enumerate(self.dataloader)):
                # TODO: DALI permute have bugs, use pytorch change format to "NCHW"
                # TODO: DALI's RGB image have changed to BGR sequence, after go through Pytorch, maybe a bug
                input = data[0]['input'].permute(0, 3, 1, 2)[:, [2, 1, 0], :, :].to(self.device)
                label = data[0]['label'].permute(0, 3, 1, 2)

                # forward
                logits = self.model(input)['out']

                # resize logits to label's size
                resized_logits = torch.nn.functional.interpolate(
                    logits,
                    label.shape,
                    mode='bilinear'
                )

                # get predict
                predict = torch.argmax(
                    torch.nn.functional.softmax(resized_logits, dim = 1),
                    axis = 1,
                    keepdim = True
                )

                # accumulate on batch
                self.accumulateOnBatch(predict.cpu().numpy(), label.cpu().numpy())

                print('Batch %d processed successful.' % iter * input.shape[0])

        # calculate IoU for each class
        for class_name, class_dict in self.final_result.items():
            class_tp = class_dict['TP']
            class_fp = class_dict['FP']
            class_fn = class_dict['FN']
            class_IoU = class_tp / (class_tp + class_fp + class_fn)
            self.final_result[class_name]['IoU'] = class_IoU

        return self.final_result
