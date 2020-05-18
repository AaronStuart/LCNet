import json
import os
import types
from itertools import cycle
from random import shuffle

import cv2
import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator


class ExternalInputIterator(object):
    def __init__(self, batch_size, root_dir, file_path):
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.file_path = file_path
        with open(file_path, 'r') as f:
            self.files = [line.rstrip() for line in f if line is not '']
        shuffle(self.files)
        self.i = 0
        self.n = len(self.files)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i > self.n:
            raise StopIteration

        jpegs, labels = [], []
        for _ in range(self.batch_size):
            jpeg_filename, label_filename = self.files[self.i].split(',')

            # open encoded jpg
            f = open(os.path.join(self.root_dir, jpeg_filename), 'rb')
            jpegs.append(np.frombuffer(f.read(), dtype=np.uint8))

            # open encoded png
            f = open(os.path.join(self.root_dir, label_filename), 'rb')
            labels.append(np.frombuffer(f.read(), dtype=np.uint8))

            self.i = (self.i + 1) % self.n

        return (jpegs, labels)

    @property
    def size(self):
        return len(self.files)

    next = __next__


class ApolloPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, iterator, is_train):
        super(ApolloPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.iterator = iterator
        self.is_train = is_train

        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()

        self.rgb_decode = ops.ImageDecoder(device="mixed", output_type=types.BGR)
        self.gray_decode = ops.ImageDecoder(device="mixed", output_type=types.GRAY)

        self.input_resize = ops.Resize(
            device='gpu', image_type=types.RGB,
            resize_longer=800,
            interp_type=types.INTERP_LINEAR
        )
        self.label_resize = ops.Resize(
            device='gpu', image_type=types.GRAY,
            resize_longer=800,
            interp_type=types.INTERP_NN
        )
        self.transpose = ops.Transpose(
            device='gpu',
            perm=[2, 0, 1]
        )

        self.change_type = ops.Cast(device='gpu', dtype=types.FLOAT)

    def define_graph(self):
        # input image preprocess
        self.jpegs = self.input()
        inputs = self.rgb_decode(self.jpegs)
        inputs = self.input_resize(inputs)
        inputs = self.change_type(inputs)
        inputs = self.transpose(inputs)

        # label preprocess
        self.labels = self.input_label()
        labels = self.gray_decode(self.labels)
        if self.is_train:
            labels = self.label_resize(labels)
        labels = self.transpose(labels)

        return (inputs, labels)

    def iter_setup(self):
        try:
            (images, labels) = self.iterator.next()
            self.feed_input(self.jpegs, images)
            self.feed_input(self.labels, labels)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration


class ApolloDaliDataset(object):
    def __init__(self, root_dir, file_path, batch_size, num_threads, is_train):
        self.root_dir = root_dir
        self.file_path = file_path
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.is_train = is_train

    def getIterator(self):
        # create disk file iterator
        file_iterator = ExternalInputIterator(
            batch_size=self.batch_size,
            root_dir=self.root_dir,
            file_path=self.file_path
        )

        # create DALI's pipeline
        dataset_pipeline = ApolloPipeline(
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            device_id=0,
            iterator=file_iterator,
            is_train=self.is_train
        )

        # create iterator of pytorch
        dataset_iterator = PyTorchIterator(
            dataset_pipeline,
            output_map=['input', 'label'],
            size=file_iterator.size,
            auto_reset=True,
            last_batch_padded=True,
            fill_last_batch=False
        )

        return dataset_iterator


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
            image = cv2.resize(image, dsize = (0, 0), fx = 0.3, fy = 0.3, interpolation=cv2.INTER_AREA)

            ###### preprocess label ########
            label_path = os.path.join(self.root_dir, label_path)
            gray_label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            gray_label = cv2.resize(gray_label, dsize = image.shape[:-1][::-1], interpolation=cv2.INTER_NEAREST)

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


if __name__ == '__main__':
    # ####################### DALI DEBUG ###############################################
    # train_iterator = ApolloDaliDataset(
    #     root_dir='/media/stuart/data/dataset/Apollo/Lane_Detection',
    #     file_path='/home/stuart/PycharmProjects/LCNet/dataset/train_apollo_gray.txt',
    #     batch_size=1,
    #     num_threads=12,
    #     is_train=True
    # ).getIterator()
    #
    # for iter, data in enumerate(train_iterator):
    #     input, label = data[0]['input'], data[0]['label']
    #     print("input shape is ", input.shape)
    #     print("label shape is ", label.shape)

    train_dataloader = ApolloBalanceTrainDataLoader(
        root_dir='/media/stuart/data/dataset/Apollo/Lane_Detection',
        json_path='/home/stuart/PycharmProjects/LCNet/dataset/test_split_by_class.json',
        batch_size=2
    )

    for image, gray_label in train_dataloader:
        pass
