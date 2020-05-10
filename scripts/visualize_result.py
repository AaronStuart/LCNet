import json

import plotly as py
import plotly.graph_objs as go


def multi_json_visualize(json_path_dict):
    # read json file
    json_file_dict = {}
    for model_name, json_path in json_path_dict.items():
        json_file_dict[model_name] = json.load(open(json_path, 'r'))

    # define traces
    traces = []
    for model_name, json_file in json_file_dict.items():
        labels = sorted(json_file.keys())
        traces.append(
            go.Bar(
                x=labels,
                y=[json_file[label]['IoU'] for label in labels],
                name=model_name
            )
        )

    # define layout
    layout = go.Layout(title='mIoU')

    # generate figure
    figure = go.Figure(
        data=traces,
        layout=layout
    )

    # draw figure
    py.offline.plot(figure, filename='baseline.html')


if __name__ == '__main__':
    # multi_json_visualize(
    #     json_path_dict={
    #         'U-Net': '/home/stuart/PycharmProjects/LCNet/experiments/UNet/UNet.json',
    #         'FCN': '/home/stuart/PycharmProjects/LCNet/experiments/FCN/FCN.json',
    #         'DeepLabV3': '/home/stuart/PycharmProjects/LCNet/experiments/DeepLabV3/DeepLabV3.json'
    #     }
    # )
    # multi_json_visualize(
    #     json_path_dict={
    #         'metric': '/home/stuart/PycharmProjects/LCNet/experiments/DeepLabV3/metric_100000_iter.json',
    #         'focal': '/home/stuart/PycharmProjects/LCNet/experiments/DeepLabV3/focal_100000_iter.json',
    #     }
    # )
    multi_json_visualize(
        json_path_dict={
            'dynamic': '/home/stuart/PycharmProjects/LCNet/experiments/DeepLabV3/dynamic_weighted_cluster_100000_iter_pretrained.json',
            'cluster': '/home/stuart/PycharmProjects/LCNet/experiments/DeepLabV3/cluster_100000_iter_pretrained.json',
            'pretrained': '/home/stuart/PycharmProjects/LCNet/experiments/DeepLabV3/focal_100000_iter_pretrained.json',
            'no_pretrained': '/home/stuart/PycharmProjects/LCNet/experiments/DeepLabV3/focal_100000_iter.json'
        }
    )
