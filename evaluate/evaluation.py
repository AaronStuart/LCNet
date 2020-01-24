from abc import abstractmethod

import numpy as np
import torch

from scripts.apollo_label import labels, trainId2name


class Evaluation:
    def __init__(self):
        super(Evaluation, self).__init__()

    @abstractmethod
    def evaluate(self, predict, label):
        pass


class EvaluationOnDataset(Evaluation):
    def __init__(self, model, device, dataloader):
        super(EvaluationOnDataset, self).__init__()
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

    def accumulateOnImage(self, predict, label):
        """

        :param predict: ndarray of shape [H, W]
        :param label: ndarray of shape [H, W]
        :return: a dict contain IoU of each class
        """
        for train_id in self.eval_trainIds:
            label_mask = label == train_id
            if not label_mask.any():
                continue

            predict_mask = predict == train_id
            TP = np.sum(label_mask * predict_mask)
            FP = np.sum((1 - label_mask) * predict_mask)
            FN = np.sum(label_mask * (1 - predict_mask))

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

    def accumulateOnBatch(self, predict, label):
        """

        :param predict: ndarray of shape [N, 1, H, W]
        :param label: ndarray of shape [N, 1, H, W]
        :return: a dict contain IoU of each class
        """
        predict = np.squeeze(predict, axis=1)
        label = np.squeeze(label, axis=1)

        batch_size = predict.shape[0]
        for batch in range(batch_size):
            self.accumulateOnImage(predict[batch], label[batch])

    def evaluate(self):
        print('Totally %d images' % len(self.dataloader))
        with torch.no_grad():
            for iter, data in enumerate(self.dataloader):
                # get data
                input, label_trainId = data['input'], data['label_trainId']
                label_shape = [label_trainId.shape[-2], label_trainId.shape[-1]]

                # forward
                output = self.model(input.to(self.device)).cpu()

                # resize to origin size
                output = torch.nn.functional.interpolate(output, label_shape, mode='bilinear')

                # accumulate on batch
                output = torch.argmax(output, axis=1, keepdim=True).numpy()
                label = label_trainId.numpy()
                self.accumulateOnBatch(output, label)
                print('image %d processed successful.' % iter * input.shape[0])

        # calculate IoU for each class
        for class_name, class_dict in self.final_result.items():
            class_tp = class_dict['TP']
            class_fp = class_dict['FP']
            class_fn = class_dict['FN']
            class_IoU = class_tp / (class_tp + class_fp + class_fn)
            self.final_result[class_name]['IoU'] = class_IoU

        return self.final_result


class mIoU(Evaluation):
    """
    use for semantic segmentation
    """

    def __init__(self, trainIds):
        """

        :param trainIds: trainIds to be calculated, usually without background
        """
        super(mIoU, self).__init__()
        self.trainIds = trainIds
        self.result = {}

    def perImage(self, predict, label):
        """

        :param predict: ndarray of shape [H, W]
        :param label: ndarray of shape [H, W]
        :return: a dict contain IoU of each class
        """
        result, IoUs = {}, []
        for train_id in self.trainIds:
            label_mask = label == train_id
            if not label_mask.any():
                continue

            predict_mask = predict == train_id
            TP = np.sum(label_mask * predict_mask)
            FP = np.sum((1 - label_mask) * predict_mask)
            FN = np.sum(label_mask * (1 - predict_mask))
            IoU = TP / (TP + FP + FN)
            result[train_id] = IoU
            IoUs.append(IoU)

        result['mIoU_of_image'] = np.mean(IoUs)
        return result

    def perBatch(self, predict, label):
        """

        :param predict: ndarray of shape [N, H, W]
        :param label: ndarray of shape [N, H, W]
        :return: a list of each image
        """
        batch_size = predict.shape[0]
        result = {}
        for batch in range(batch_size):
            IoU_per_image = self.perImage(predict[batch], label[batch])
            result[batch] = IoU_per_image

        return result

    def evaluateOnBatch(self, predict, label):
        """

        :param predict: ndarray of shape [N, 1, H, W]
        :param label: ndarray of shape [N, 1, H, W]
        :return: a dict contain IoU of each class
        """
        predict = np.squeeze(predict, axis=1)
        label = np.squeeze(label, axis=1)

        batch_result = self.perBatch(predict, label)
        mIoUs_of_images = [image_result['mIoU_of_image'] for image_result in batch_result.values()]
        batch_result['mIoU_of_batch'] = np.mean(mIoUs_of_images)

        return batch_result
