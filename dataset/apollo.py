import os
import types
from random import shuffle
import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import matplotlib.pyplot as plt

class ExternalInputIterator(object):
    def __init__(self, batch_size, root_dir, file_path):
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.file_path = file_path
        with open(file_path, 'r') as f:
            self.files = [line.rstrip() for line in f if line is not '']
        shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        jpegs, labels = [], []
        for _ in range(self.batch_size):
            jpeg_filename, label_filename = self.files[self.i].split(',')

            # open encoded jpg
            f = open(os.path.join(self.root_dir, jpeg_filename), 'rb')
            jpegs.append(np.frombuffer(f.read(), dtype = np.uint8))

            # open encoded png
            f = open(os.path.join(self.root_dir, label_filename), 'rb')
            labels.append(np.frombuffer(f.read(), dtype=np.uint8))

            self.i = (self.i + 1) % self.n

        return (jpegs, labels)

    next = __next__


class ApolloPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, iterator):
        super(ApolloPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.iterator = iterator
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.cast = ops.Cast(device="gpu", dtype=types.INT32)

    def define_graph(self):
        self.jpegs = self.input()
        images = self.decode(self.jpegs)

        self.labels = self.input_label()
        labels = self.decode(self.labels)

        return (images, labels)

    def iter_setup(self):
        (images, labels) = self.iterator.next()
        self.feed_input(self.jpegs, images, layout="HWC")
        self.feed_input(self.labels, labels, layout="HWC")


def main():
    eii = ExternalInputIterator(
        batch_size=4,
        root_dir='/media/stuart/data/dataset/Apollo/Lane_Detection',
        file_path='/home/stuart/PycharmProjects/LCNet/dataset/train_apollo.txt'
    )
    iterator = iter(eii)

    pipe = ApolloPipeline(
        batch_size=4,
        num_threads=2,
        device_id=0,
        iterator = iterator
    )

    pipe.build()
    pipe_out = pipe.run()
    print(pipe_out)

    inputs = pipe_out[0].as_cpu()
    labels = pipe_out[1].as_cpu()

    image = inputs.at(2)
    label = labels.at(2)
    print(image.shape)
    print(label.shape)
    plt.imshow(image.astype('uint8'))
    plt.imshow(label.astype('uint8'))
    plt.show()


if __name__ == '__main__':
    main()