import os

import cupy as cp
import cv2 as cv
import numpy as np

from scripts.apollo_label import color2trainId


class ColorToGray(object):
    def __init__(self, root_dir):
        """

        :param root_dir: dataset dir
        """
        self.label_paths = self.collect_label_path(root_dir)

    def collect_label_path(self, root_dir):
        result = []
        for root, dirs, files in os.walk(root_dir):
            for image_name in files:
                if image_name.endswith('jpg'):
                    rgb_label_path = os.path.join(root, image_name.replace('jpg', 'png'))
                    gray_label_path = rgb_label_path.replace('.png', '_gray.png')
                    if not os.path.exists(gray_label_path):
                        result.append(rgb_label_path)
        return result

    def generate_trainid_label(self):
        for label_path in self.label_paths:
            label = cv.imread(label_path)
            trainid_label = self.map_color_to_gray(label)
            cv.imwrite(label_path.replace('.png', '_gray.png'), trainid_label)

    def map_opencv_to_gray(self, opencv_label):
        """

        :param bgr_label: label opened by opencv
        :return: trainid_label
        """
        label = cp.asarray(opencv_label)
        trainId_label = cp.zeros(label.shape[:2], dtype=np.uint8)

        for rgb, trainId in color2trainId.items():
            bgr = cp.array(rgb[::-1])
            mask = (label == bgr).all(axis=2)
            trainId_label[mask] = trainId

        return cp.asnumpy(trainId_label)


if __name__ == '__main__':
    ColorToGray(
        root_dir='/media/stuart/data/dataset/Apollo/Lane_Detection'
    ).generate_trainid_label()
