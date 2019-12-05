import argparse
import os

import cv2
import numpy as np

from scripts.apollo_label import labels

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', required = False,
                    default = '/Users/zhangjiazhao/projects/LCNet/scripts/apollo_label.txt',
                    help = 'Each line in the file corresponds to the absolute path of a apollo label')
args = parser.parse_args()

# create a map form BGR color to trainId
color2trainId = {label.color[::-1] : label.trainId for label in labels}

def changeToTrainId(origin_label_path):
    origin_label = cv2.imread(origin_label_path)

    # create a black train_id_label
    trainId_label = np.zeros(origin_label.shape[:2], dtype=np.uint8)

    for bgr_color, trainId in color2trainId.items():
        # map color to trainId
        mask = (origin_label == bgr_color).all(axis=2)
        trainId_label[mask] = trainId

    return trainId_label