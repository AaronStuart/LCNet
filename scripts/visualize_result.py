import json
import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

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

def two_json_visualize(fcn_json, unet_json):
    fcn_result = json.load(open(fcn_json, 'r'))
    unet_result = json.load(open(unet_json, 'r'))

    # sort dict by keys
    labels = sorted(fcn_result.keys())
    fcn_miou = [fcn_result[label]['IoU'] for label in labels]
    unet_miou = [unet_result[label]['IoU'] for label in labels]

    fig, ax = plt.subplots()
    # the label locations
    x = np.arange(len(labels))
    # the width of the bars
    width = 0.35

    ax.bar(x - width / 2, fcn_miou, width, label='FCN')
    ax.bar(x + width / 2, unet_miou, width, label='U-Net')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('mIoU')
    plt.xticks(np.arange(len(labels)), labels, rotation='vertical')
    ax.legend()

    fig.tight_layout()
    plt.savefig('/home/stuart/PycharmProjects/EDANet/output/compare.png', transparent=True, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    fcn_json = '/home/stuart/PycharmProjects/EDANet/output/FCN/FCN.json'
    unet_json = '/home/stuart/PycharmProjects/EDANet/output/UNet/UNet.json'
    two_json_visualize(fcn_json, unet_json)