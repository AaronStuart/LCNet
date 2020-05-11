import json
import os

import cv2 as cv
import numpy as np

from tqdm import tqdm

from scripts.apollo_label import trainId2name


class ClassSplit:
    def __init__(self, dataset_root, file_path, out_path):
        self.dataset_path = dataset_root
        self.file_path = file_path
        self.out_path = out_path
        self.result = {}

    def run(self):
        with open(self.file_path, 'r') as txt:
            for line in tqdm(txt.readlines()):
                image_path, label_path = line.strip().split(',')

                gray_label = cv.imread(os.path.join(self.dataset_path, label_path), cv.IMREAD_UNCHANGED)

                gray_labels, nums = np.unique(gray_label, return_counts = True)

                for train_id, num in zip(gray_labels, nums):
                    class_name = trainId2name[train_id]

                    if class_name in ['void', 'ignored'] or num < 5000:
                        continue

                    if class_name not in self.result.keys():
                        self.result[class_name] = [(image_path, label_path)]
                        continue

                    self.result[class_name].append((image_path, label_path))

        file_name = self.file_path.split('/')[-1].split('.')[0]
        out_json_path = os.path.join(self.out_path, '%s_split_by_class.json' % file_name)
        json.dump(self.result, open(out_json_path, 'w'))

if __name__ == '__main__':
    service = ClassSplit(
        dataset_root = '/media/stuart/data/dataset/Apollo/Lane_Detection',
        file_path = '/home/stuart/PycharmProjects/LCNet/dataset/train_apollo_gray.txt',
        out_path = '/home/stuart/PycharmProjects/LCNet/dataset'
    )

    service.run()
