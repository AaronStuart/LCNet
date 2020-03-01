import os

import cv2
import torch
import numpy as np
import cupy as cp
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from scripts.apollo_label import color2trainId

from line_profiler import LineProfiler

class ApolloLaneDataset(Dataset):
    def __init__(self, root_dir, path_file):
        self.root_dir = root_dir
        self.path_file = path_file

        # load file
        self.path_list = []
        with open(self.path_file) as file:
            for line in file.readlines():
                self.path_list.append(line.strip().split(','))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.path_list[index][0])
        label_path = os.path.join(self.root_dir, self.path_list[index][1])

        # open image and label, change image from BGR to RGB
        orgin_image = cp.asarray(cv2.imread(image_path))
        origin_label = cp.asarray(cv2.imread(label_path, cv2.IMREAD_UNCHANGED))

        # concat along channel axis
        image_label_concat = cp.concatenate([orgin_image, origin_label], axis=-1)

        # random crop
        crop_result = self.randomCrop(image_label_concat, crop_ratio = 1 / 3)
        crop_image, crop_label = crop_result[:, :, : 3], crop_result[:, :, 3 : ]

        train_label = self.bgrToGray(crop_label)
        train_label = cp.expand_dims(train_label, axis = 0)

        # change input from BGR to RGB
        input = F.to_tensor(cp.asnumpy(crop_image)[:, :, [2, 1, 0]])
        train_label = torch.tensor(train_label, dtype = torch.int)


        return {
            'input': input,
            'crop_label': cp.asnumpy(crop_label)[:, :, [2, 1, 0]] ,
            'train_label': train_label

        }

    def bgrToGray(self, bgr_label):
        train_label = cp.zeros(bgr_label.shape[:2])
        for color, trainId in color2trainId.items():
            # color is RGB format, label is BGR format
            color = cp.asarray(color)[::-1]
            mask = (bgr_label == color).all(axis=2)
            train_label[mask] = trainId
        return train_label

    def randomCrop(self, image, crop_ratio):
        H, W, C = image.shape

        # get target size
        target_height = int(H * crop_ratio)
        target_width = int(W * crop_ratio)

        # random select crop area
        h_begin = cp.random.randint(low = 0, high = H - target_height)
        w_begin = cp.random.randint(low = 0, high = W - target_width)

        return image[h_begin : h_begin + target_height, w_begin : w_begin + target_width, :]



def main():
    dataset = ApolloLaneDataset(
        root_dir='/media/stuart/data/dataset/Apollo/Lane_Detection',
        path_file='/home/stuart/PycharmProjects/LCNet/dataset/train_apollo.txt'
    )
    print("len of dataset: ", len(dataset))

    data = dataset[0]
    print("input shape: ", data['input'].shape)
    print("crop_label shape: ", data['crop_label'].shape)
    print("train_label shape: ", data['train_label'].shape)
    print("train_label unique values: ", data['train_label'].unique())

if __name__ == '__main__':
    lp = LineProfiler()
    lp.add_function(ApolloLaneDataset.__getitem__)
    lp.add_function(ApolloLaneDataset.bgrToGray)
    lp_wrapper = lp(main)
    lp_wrapper()
    lp.print_stats()