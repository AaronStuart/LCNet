import os

import numpy as np
import json
import cv2 as cv

from scripts.apollo_label import trainId2name


class DataBalance:
    def __init__(self, root_dir, txt_path, out_path):
        self.root_dir = root_dir
        self.txt_path = txt_path

        txt_name = txt_path.split('/')[-1].split('.')[0]
        self.out_json_path = os.path.join(out_path, '%s_statistics.json' % txt_name)

    def generate_statistics(self):
        txt = open(self.txt_path, 'r')

        label_statistics = {}
        for line in txt.readlines():
            image_path, label_path = line.strip().split(',')

            label_path = os.path.join(self.root_dir, label_path)
            label = cv.imread(label_path, cv.IMREAD_UNCHANGED)

            labels, counts = np.unique(label, return_counts = True)
            for train_id, pixel_num in zip(labels, counts):
                class_name = trainId2name[train_id]

                if train_id not in label_statistics.keys():
                    label_statistics[class_name] = float(pixel_num)
                    continue

                label_statistics[class_name] += float(pixel_num)

        json.dump(label_statistics, open(self.out_json_path, 'w'), indent = 4)

if __name__ == '__main__':
    data_balance = DataBalance(
        root_dir = '/media/stuart/data/dataset/Apollo/Lane_Detection',
        txt_path = '/home/stuart/PycharmProjects/LCNet/dataset/train_apollo_gray.txt',
        out_path = '/home/stuart/PycharmProjects/LCNet/materials'
    )

    data_balance.generate_statistics()