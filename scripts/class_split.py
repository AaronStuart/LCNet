import numpy as np
import cv2 as cv
import json
import os

class ClassSplit:
    def __init__(self, dataset_root, file_path, out_path):
        self.dataset_path = dataset_root
        self.file_path = file_path
        self.out_path = out_path
        self.result = {}

    def run(self):
        with open(self.file_path, 'r') as txt:
            for line in txt.readlines():
                image_path, label_path = line.strip().split(',')


                gray_label = cv.imread(os.path.join(self.dataset_path, label_path), cv.IMREAD_UNCHANGED)

                gray_labels = np.unique(gray_label)

                for train_id in gray_labels:
                    if train_id not in self.result.keys():
                        self.result[train_id] = [(image_path, label_path)]
                        continue
                    self.result[train_id].append((image_path, label_path))

        file_name = self.file_path.split('/')[-1].split('.')[0]
        out_json_path = os.path.join(self.out_path, '%s_split_by_class.json' % file_name)
        json.dump(self.result, out_json_path, indent = 4)

if __name__ == '__main__':
    service = ClassSplit(
        dataset_root = '/media/stuart/data/dataset/Apollo/Lane_Detection',
        file_path = '/home/stuart/PycharmProjects/LCNet/dataset/small.txt',
        out_path = '/home/stuart/PycharmProjects/LCNet/dataset'
    )

    service.run()
