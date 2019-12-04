import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BDD100K(Dataset):
    def __init__(self, path_file):
        self.path_file = path_file

        # load file
        self.path_list = []
        with open(self.path_file) as file:
            for line in file.readlines():
                self.path_list.append(line.lstrip().rstrip().split(' '))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        image_path = self.path_list[index][0]
        label_path = self.path_list[index][1]

        # preprocess image
        image = Image.open(image_path)
        image_transform = transforms.Compose(
            [
                transforms.Resize([512, 1024]),
                transforms.ToTensor()
            ]
        )
        image = image_transform(image)

        # preprocess label
        label = Image.open(label_path)
        label_transform = transforms.Compose(
            [
                transforms.Resize([512, 1024],interpolation = Image.NEAREST),
                transforms.Grayscale(),
                transforms.ToTensor()
            ]
        )
        label = label_transform(label)

        return {'image': image, 'label': label}

if __name__ == '__main__':
    path_file = '/home/stuart/PycharmProjects/EDANet/dataset/train_bdd100k.txt'
    dataset = BDD100K(path_file)
    print("len of dataset: ", len(dataset))
    print("image shape: ", dataset[0]['image'].shape)
    print("label shape: ", dataset[0]['label'].shape)
    print("label unique values: ", dataset[0]['label'].unique())