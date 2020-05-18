import os
import random

import cv2 as cv
import numpy as np
from tqdm import tqdm

from scripts.apollo_label import trainId2name


class TrainValSplit(object):
    def __init__(self, root_dir, val_ratio, total_file, train_file, val_file):
        """Divide the dataset by category to ensure that all categories of data exist in the validation set

        :param root_dir: Root directory of the dataset
        :param val_ratio: percentage of validation set
        :param train_file: train file path
        :param val_file: val file path
        """
        self.root_dir = root_dir
        self.val_ratio = val_ratio
        self.train_file = train_file
        self.val_file = val_file

        # collect images' path and labels'path
        self.image_label_pairs = self.getImageLabelPairs(root_dir)

        # save total paths to txt
        with open(total_file, 'w') as total:
            for image_path, label_path in self.image_label_pairs:
                total.writelines(image_path + ',' + label_path + '\n')

        # get split dict
        self.split_dict = self.getSplitDict()

        # do train val split
        self.train_set, self.val_set = self.trainValSplit()

    def getImageLabelPairs(self, root_dir):
        image_dirs = [
            os.path.join(root_dir, 'ColorImage_road02', 'ColorImage'),
            os.path.join(root_dir, 'ColorImage_road03', 'ColorImage'),
            os.path.join(root_dir, 'ColorImage_road04', 'ColorImage')
        ]
        result = []
        for image_dir in image_dirs:
            for dirpath, dirnames, filenames in os.walk(image_dir):
                for filename in filenames:
                    if filename.endswith('.jpg'):
                        image_path = os.path.join(dirpath, filename)
                        label_path = image_path.replace('ColorImage_road', 'Labels_road').replace('ColorImage',
                                                                                                  'Label').replace(
                            '.jpg', '_bin.png')
                        if os.path.exists(label_path):
                            result.append((image_path[len(self.root_dir) + 1:], label_path[len(self.root_dir) + 1:]))
                        else:
                            print(image_path, 'does not exist', label_path)
        random.shuffle(result)
        return result

    def getSplitDict(self):
        split_dict = {}

        # split by class
        for image_path, label_path in tqdm(self.image_label_pairs):

            # parse gray label
            try:
                gray_label = cv.imread(os.path.join(self.root_dir, label_path.replace('.png', '_gray.png')),
                                       cv.IMREAD_UNCHANGED)
                train_ids, counts = np.unique(gray_label, return_counts=True)
            except:
                continue

            for train_id, count in zip(train_ids, counts):
                class_name = trainId2name[train_id]
                if class_name in ['void', 'ignored'] or count < 5000:
                    continue

                if class_name not in split_dict.keys():
                    split_dict[class_name] = []
                split_dict[class_name].append((image_path, label_path))

        return split_dict

    def trainValSplit(self):
        all_set, val_set = set(), set()

        for label_name, label_list in self.split_dict.items():
            all_set.update(set(label_list))

            if label_name in ['s_w_d', 's_y_d', 's_w_p', 'b_y_g', 'b_w_g', 'c_wy_z']:
                continue

            # sample val set
            val_set.update(random.sample(label_list, int(self.val_ratio * len(label_list))))

        train_set = all_set - val_set

        return train_set, val_set

    def outputTxt(self):
        # write to train file
        with open(self.train_file, 'w') as train:
            for image_path, label_path in self.train_set:
                train.writelines(image_path + ',' + label_path + '\n')

        # write to val file
        with open(self.val_file, 'w') as val:
            for image_path, label_path in self.val_set:
                val.writelines(image_path + ',' + label_path + '\n')

    def run(self):
        self.outputTxt()


if __name__ == '__main__':
    TrainValSplit(
        root_dir='/media/stuart/data/dataset/Apollo/Lane_Detection',
        val_ratio=0.2,
        total_file='/home/stuart/PycharmProjects/LCNet/dataset/apollo.txt',
        train_file='/home/stuart/PycharmProjects/LCNet/dataset/apollo_train.txt',
        val_file='/home/stuart/PycharmProjects/LCNet/dataset/apollo_val.txt'
    ).run()
