import os

import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from scripts.apollo_label import color2trainId


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
        ###### preprocess image #######
        image_path = os.path.join(self.root_dir, self.path_list[index][0])
        image = Image.open(image_path)
        image_transform = transforms.Compose(
            [
                transforms.Resize([512, 1024]),
                transforms.ToTensor()
            ]
        )
        input = image_transform(image)

        ###### preprocess label ########
        label_path = os.path.join(self.root_dir, self.path_list[index][1])
        origin_label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # create a black train_id_label
        canvas = np.zeros(origin_label.shape[:2], dtype=np.uint8)
        for color, trainId in color2trainId.items():
            # map color to trainId
            mask = (origin_label == color).all(axis=2)
            canvas[mask] = trainId
        canvas = np.expand_dims(canvas, axis=0)
        label_trainId = torch.tensor(canvas)
        
        # change from [H, W, C] to [C, H, W]
        origin_image = np.transpose(np.array(image), axes=[2, 0, 1])
        origin_label = np.transpose(origin_label, axes=[2, 0, 1])
        # change form BGR to RGB
        origin_label = np.ascontiguousarray(origin_label[::-1, :, :])

        return {
            'input': input,
            'label_trainId': label_trainId,
            'origin_image': origin_image,
            'origin_label': origin_label
        }


if __name__ == '__main__':
    dataset = ApolloLaneDataset(
        root_dir='/media/stuart/data/dataset/Apollo/Lane_Detection',
        path_file='/home/stuart/PycharmProjects/EDANet/dataset/train_apollo.txt'
    )
    print("len of dataset: ", len(dataset))

    data = dataset[0]
    print("input shape: ", data['input'].shape)
    print("label_trainId shape: ", data['label_trainId'].shape)
    print("origin_input shape: ", data['origin_label'].shape)
    print("origin_label shape: ", data['origin_label'].shape)
    print("label_for_train unique values: ", data['label_trainId'].unique())