import json
import os
import random

import cv2 as cv
import cupy as cp
from tqdm import tqdm

from scripts.apollo_label import labels, name2color


class TrainValSplit(object):
    def __init__(self, root_dir, val_ratio, train_file, val_file):
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

        # all samples
        self.image_label_pairs = self.getImageLabelPairs(root_dir)

        # Results in ascending order of pixels
        self.frequency_ascending_order = (
            'a_n_lu',
            'a_w_lr',
            'sb_y_do',
            'sb_w_do',
            'db_y_g',
            'b_n_sr',
            'a_w_u',
            's_w_c',
            'd_wy_za',
            's_y_c',
            'ds_w_dn',
            'a_w_r',
            'a_w_tl',
            'b_y_g',
            'om_n_n',
            's_w_p',
            'a_w_t',
            'a_w_tr',
            'r_wy_np',
            's_w_s',
            'vom_wy_n',
            'a_w_l',
            'ds_y_dn',
            's_y_d',
            'b_w_g',
            's_w_d',
            'c_wy_z',
            'ignored',
            'void'
        )

        # the final split result
        self.train_set = []
        self.val_set = []

        # auxliary dict
        self.split_dict = self.getSplitDict()

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
                        label_path = image_path.replace('ColorImage_road', 'Labels_road').replace('ColorImage', 'Label').replace('.jpg', '_bin.png')
                        if os.path.exists(label_path):
                            result.append((image_path, label_path))
                        else:
                            print(image_path, 'does not exist', label_path)
        return result

    def getSplitDict(self):
        # define initial split dict
        split_dict = {label.name: [] for label in labels}
        processed = set()

        # restore checkpoint
        split_dict = json.load(open('../output/split.json', 'r'))
        processed = set(json.load(open('../output/processed.json', 'r')))

        # split by class
        for image_path, label_path in tqdm(self.image_label_pairs):
            # skip processed image
            if image_path in processed:
                continue

            # read label
            print('Begin to process %s' % image_path)
            label = cv.imread(label_path)
            for label_name in self.frequency_ascending_order:
                bgr = name2color[label_name][::-1]
                # If it contains pixels of this class
                if cp.any(cp.all(cp.array(label == bgr), axis = 2)):
                    relative_image_path = image_path.replace(self.root_dir + os.sep, '')
                    relative_label_path = label_path.replace(self.root_dir + os.sep, '')
                    split_dict[label_name].append((relative_image_path, relative_label_path))
                    processed.add(image_path)
                    break
            print('Process %s finished' % image_path)

            # store processed result
            with open('../output/processed.json', 'w') as json_file:
                json.dump(list(processed), json_file, indent = 4)
            with open('../output/split.json', 'w') as json_file:
                json.dump(split_dict, json_file, indent = 4)

        return split_dict
            
    def trainValSplit(self):
        for label_name, label_list in self.split_dict.items():
            if not label_list:
                continue
            for path_tuple in label_list:
                rand_num = random.random()
                if rand_num < self.val_ratio:
                    self.val_set.append(path_tuple)
                else:
                    self.train_set.append(path_tuple)

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
        self.trainValSplit()
        self.outputTxt()

if __name__ == '__main__':

    TrainValSplit(
        root_dir = '/media/stuart/data/dataset/Apollo/Lane_Detection',
        val_ratio = 0.1,
        train_file = '../dataset/train_apollo.txt',
        val_file = '../dataset/val_apollo.txt'
    ).run()