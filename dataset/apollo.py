import os
import types
from random import shuffle

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
            resize_longer = 800,
            interp_type=types.INTERP_LINEAR
        )
        self.label_resize = ops.Resize(
            device='gpu', image_type=types.GRAY,
            resize_longer = 800,
            interp_type=types.INTERP_NN
        )

        self.change_type = ops.Cast(device='gpu', dtype=types.FLOAT)

    def define_graph(self):
        # input image preprocess
        self.jpegs = self.input()
        inputs = self.rgb_decode(self.jpegs)
        inputs = self.input_resize(inputs)
        inputs = self.change_type(inputs)

        # label preprocess
        self.labels = self.input_label()
        labels = self.gray_decode(self.labels)
        if self.is_train:
            labels = self.label_resize(labels)

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
            is_train = self.is_train
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


if __name__ == '__main__':
    train_iterator = ApolloDaliDataset(
        root_dir='/media/stuart/data/dataset/Apollo/Lane_Detection',
        file_path='/home/stuart/PycharmProjects/LCNet/dataset/train_apollo_gray.txt',
        batch_size=1,
        num_threads=12,
        is_train=True
    ).getIterator()

    for iter, data in enumerate(train_iterator):
        input, label = data[0]['input'], data[0]['label']
        print("input shape is ", input.shape)
        print("label shape is ", label.shape)