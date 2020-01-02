import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from scripts.apollo_label import color2trainId


class ApolloLaneDataset(Dataset):
    def __init__(self, path_file, is_train):
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
        image_path = self.path_list[index][0]
        label_path = self.path_list[index][1]

        ###### preprocess image #######
        image = Image.open(image_path)
        image_transform = transforms.Compose(
            [
                transforms.Resize([512, 1024]),
                transforms.ToTensor()
            ]
        )
        image = image_transform(image)

        ###### preprocess image ########
        label_bgr = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if self.is_train:
            label_bgr = cv2.resize(label_bgr, (1024, 512), interpolation=cv2.INTER_NEAREST)

        # create a black train_id_label
        canvas = np.zeros(label_bgr.shape[:2], dtype=np.uint8)
        for bgr_color, trainId in color2trainId.items():
            # map color to trainId
            mask = (label_bgr == bgr_color).all(axis=2)
            canvas[mask] = trainId
        canvas = np.expand_dims(canvas, axis = 0)

        label_trainId = torch.tensor(canvas)
        label_bgr = np.transpose(label_bgr, axes = [2, 0, 1])

        return {'input': image, 'label_trainId': label_trainId, 'label_bgr' : label_bgr}

if __name__ == '__main__':
    path_file = '/home/stuart/PycharmProjects/EDANet/dataset/train_apollo.txt'
    
    dataset = ApolloLaneDataset(path_file)
    print("len of dataset: ", len(dataset))
    
    data = dataset[0]
    print("input shape: ", data['input'].shape)
    print("label_trainId shape: ", data['label_trainId'].shape)
    print("label_bgr shape: ", data['label_bgr'].shape)
    print("label_for_train unique values: ", data['label_trainId'].unique())