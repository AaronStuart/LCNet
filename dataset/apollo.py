import json
import os
from itertools import cycle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ApolloBalanceTrainDataLoader:
    def __init__(self, root_dir, json_path, batch_size):
        self.root_dir = root_dir
        self.class_paths = self.get_class_paths(json_path)
        self.batch_size = batch_size
        self.i = 0

    def get_class_paths(self, json_path):
        class_dict = json.load(open(json_path, 'r'))

        class_paths = []
        for class_name, paths in class_dict.items():
            class_paths.append(cycle(paths))

        return class_paths

    def __iter__(self):
        return self

    def __next__(self):
        inputs, labels = [], []
        for _ in range(self.batch_size):
            image_path, label_path = next(self.class_paths[self.i])

            ###### preprocess image #######
            image_path = os.path.join(self.root_dir, image_path)
            image = cv2.imread(image_path)
            image = cv2.resize(image, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

            ###### preprocess label ########
            label_path = os.path.join(self.root_dir, label_path)
            gray_label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            gray_label = cv2.resize(gray_label, dsize=image.shape[:-1][::-1], interpolation=cv2.INTER_NEAREST)

            # change from [H, W, C] to [C, H, W]
            image = np.transpose(image, axes=[2, 0, 1])
            gray_label = gray_label[np.newaxis, :, :]

            # change form BGR to RGB
            image = np.ascontiguousarray(image[::-1, :, :])

            inputs.append(image)
            labels.append(gray_label)

            # circle train every class
            self.i = (self.i + 1) % len(self.class_paths)
        batch_image, batch_label = np.stack(inputs), np.stack(labels)
        return batch_image, batch_label


class ApolloBalanceTrainEvalDataLoader:
    def __init__(self, root_dir, file_path, batch_size):
        self.root_dir = root_dir
        self.paths = self.get_paths(file_path)
        self.batch_size = batch_size

    def get_paths(self, file_path):
        result = []
        with open(file_path, 'r') as txt:
            for line in txt.readlines():
                image_path, gray_label_path = line.strip().split(',')

                image_path = os.path.join(self.root_dir, image_path)
                gray_label_path = os.path.join(self.root_dir, gray_label_path)
                result.append((image_path, gray_label_path))
        return result

    def __iter__(self):
        return self

    def __next__(self):
        image_batch, label_batch = [], []
        for _ in range(self.batch_size):
            if len(self.paths) == 0:
                raise StopIteration
            image_path, gray_label_path = self.paths.pop()

            ###### preprocess image #######
            image = cv2.imread(image_path)
            image = cv2.resize(image, (800, 641), interpolation=cv2.INTER_AREA)

            ###### preprocess label ########
            gray_label = cv2.imread(gray_label_path, cv2.IMREAD_UNCHANGED)

            # change from [H, W, C] to [C, H, W]
            image = np.transpose(image, axes=[2, 0, 1])
            gray_label = gray_label[np.newaxis, :, :]

            # change form BGR to RGB
            image = np.ascontiguousarray(image[::-1, :, :])

            image_batch.append(image)
            label_batch.append(gray_label)

        batch_image, batch_label = np.stack(image_batch), np.stack(label_batch)
        return batch_image, batch_label


class ApolloDataset(Dataset):
    def __init__(self, root_dir, path_file, is_train=True):
        self.root_dir = root_dir
        self.path_file = path_file
        self.is_train = is_train

        # load file
        self.path_list = []
        with open(self.path_file) as file:
            for line in file.readlines():
                self.path_list.append(line.strip().split(','))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        ###### preprocess image #######
        image_path = os.path.join(self.root_dir, self.path_list[index][0])
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

        ###### preprocess label ########
        label_path = os.path.join(self.root_dir, self.path_list[index][1])
        gray_label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        if self.is_train:
            gray_label = cv2.resize(gray_label, dsize=image.shape[:-1][::-1], interpolation=cv2.INTER_NEAREST)

        # change from [H, W, C] to [C, H, W]
        image = np.transpose(image, axes=[2, 0, 1])
        gray_label = gray_label[np.newaxis, :, :]

        # change form BGR to RGB
        image = np.ascontiguousarray(image[::-1, :, :])

        return {
            'image': torch.tensor(image).to(torch.float),
            'label': torch.tensor(gray_label).to(torch.uint8)
        }


if __name__ == '__main__':
    # train_dataloader = ApolloBalanceTrainDataLoader(
    #     root_dir='/media/stuart/data/dataset/Apollo/Lane_Detection',
    #     json_path='/home/stuart/PycharmProjects/LCNet/dataset/test_split_by_class.json',
    #     batch_size=2
    # )
    #
    # for image, gray_label in train_dataloader:
    #     pass

    dataset = ApolloDataset(
        root_dir='/media/stuart/data/dataset/Apollo/Lane_Detection',
        path_file='/home/stuart/PycharmProjects/LCNet/dataset/apollo_train_gray.txt'
    )
    print("len of dataset: ", len(dataset))

    data = dataset[0]
    print("image shape: ", data['image'].shape)
    print("label shape: ", data['label'].shape)

    trainloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    for data in trainloader:
        print("image shape: ", data['image'].shape)
        print("label shape: ", data['label'].shape)
