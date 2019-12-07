from abc import abstractmethod

import numpy as np


class Evaluation:
    def __init__(self):
        super(Evaluation, self).__init__()

    @abstractmethod
    def evaluate(self, predict, label):
        pass

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

    def evaluate(self, predict, label):
        """

        :param predict: ndarray of shape [N, 1, H, W]
        :param label: ndarray of shape [N, 1, H, W]
        :return: a dict contain IoU of each class
        """
        predict = np.squeeze(predict, axis = 1)
        label = np.squeeze(label, axis = 1)

        batch_result = self.perBatch(predict, label)
        mIoUs_of_images = [image_result['mIoU_of_image'] for image_result in batch_result.values()]
        batch_result['mIoU_of_batch'] = np.mean(mIoUs_of_images)

        return batch_result






