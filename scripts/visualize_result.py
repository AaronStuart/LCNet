import json
import matplotlib.pyplot as plt
import numpy as np


def single_json_visiualize(json_file):
    # load json file to dict
    result_dict = json.load(open(json_file, 'r'))

    # sort dict by keys
    labels = sorted(result_dict.keys())
    IoUs = [result_dict[label]['IoU'] for label in labels]

    fig, ax = plt.subplots()
    # set x axis
    plt.xticks(np.arange(len(labels)), labels, rotation = 'vertical')
    # set y axis
    plt.bar(np.arange(len(labels)), IoUs)

    plt.savefig(json_file.replace('json', 'png'), transparent = False, bbox_inches = 'tight')
    plt.show()


if __name__ == '__main__':
    json_file = '/home/stuart/PycharmProjects/EDANet/output/UNet/UNet.json'
    single_json_visiualize(json_file)