import os
import types
from random import shuffle

import numpy as np
import cv2 as cv
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torch
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator as PyTorchIterator

from scripts.apollo_label import valid_trainIds


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
    def __init__(self, batch_size, num_threads, device_id, iterator):
        super(ApolloPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.iterator = iterator
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.rgb_decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.gray_decode = ops.ImageDecoder(device="mixed", output_type=types.GRAY)

        self.input_resize = ops.Resize(device='gpu', resize_longer=1024, interp_type=types.INTERP_LINEAR)
        self.label_resize = ops.Resize(device='gpu', resize_longer=1024, interp_type=types.INTERP_NN)

        self.change_type = ops.Cast(device='gpu', dtype=types.FLOAT)
        self.transpose = ops.Transpose(device="gpu", perm=[2, 0, 1])

    def define_graph(self):
        self.jpegs = self.input()
        images = self.rgb_decode(self.jpegs)
        resized_images = self.input_resize(images)
        resized_images = self.change_type(resized_images)
        resized_images = self.transpose(resized_images)

        self.labels = self.input_label()
        labels = self.gray_decode(self.labels)
        resize_labels = self.label_resize(labels)
        resize_labels = self.transpose(resize_labels)

        return (resized_images, resize_labels)

    def iter_setup(self):
        try:
            (images, labels) = self.iterator.next()
            self.feed_input(self.jpegs, images, layout="HWC")
            self.feed_input(self.labels, labels, layout="HWC")
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration


class ApolloDataset:
    def __init__(self, root_dir, file_path, batch_size, num_threads):
        self.root_dir = root_dir
        self.file_path = file_path
        self.batch_size = batch_size
        self.num_threads = num_threads

    def getIterator(self):
        #
        file_iterator = ExternalInputIterator(
            batch_size=self.batch_size,
            root_dir=self.root_dir,
            file_path=self.file_path
        )

        dataset_pipeline = ApolloPipeline(
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            device_id=0,
            iterator=file_iterator
        )

        dataset_iterator = PyTorchIterator(
            dataset_pipeline,
            size=file_iterator.size,
            last_batch_padded=True,
            fill_last_batch=False
        )

        return dataset_iterator

if __name__ == '__main__':
    apollo_iterator = ApolloDataset(
        root_dir='/media/stuart/data/dataset/Apollo/Lane_Detection',
        file_path='/home/stuart/PycharmProjects/LCNet/dataset/train_apollo_gray.txt',
        batch_size=4,
        num_threads=12
    ).getIterator()

    for iter, data in enumerate(apollo_iterator):
        print("input shape is ", data[0]['data'].shape)
        print("label shape is ", data[0]['label'].shape)
        # if iter == 0:
        #     gray_label = torch.squeeze(data[0]['label'][0])
        #     for gray_value in valid_trainIds:
        #         canvas = torch.zeros_like(gray_label, dtype=torch.uint8)
        #         canvas[gray_label == gray_value] = 255
        #         cv.imwrite("%d.png" % gray_value, canvas.cpu().numpy())

