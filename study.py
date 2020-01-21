import torch
import numpy as np

def batch_norm(feature_map, epsilon = 1e-5):
    mean = np.mean(feature_map, axis = (0, 2, 3), keepdims = True)
    var = np.var(feature_map, axis = (0, 2, 3), keepdims = True)
    return (feature_map - mean) / np.sqrt(var + epsilon)

def layer_norm(feature_map, epsilon = 1e-5):
    mean = np.mean(feature_map, axis = (1, 2, 3), keepdims = True)
    var = np.var(feature_map, axis = (1, 2, 3), keepdims = True)
    return (feature_map - mean) / np.sqrt(var + epsilon)

def instance_norm(feature_map, epsilon = 1e-5):
    mean = np.mean(feature_map, axis = (2, 3), keepdims = True)
    var = np.var(feature_map, axis = (2, 3), keepdims = True)
    return (feature_map - mean) / np.sqrt(var + epsilon)

def group_norm(feature_map, group, epsilon = 1e-5):
    # create group dim
    n, c, h, w = feature_map.shape
    feature_map = np.reshape(feature_map, [n, group, c // group, h, w])

    mean = np.mean(feature_map, axis = (0, 2, 3, 4), keepdims = True)
    var = np.var(feature_map, axis = (0, 2, 3, 4), keepdims = True)

    normalized = (feature_map - mean) / np.sqrt(var + epsilon)
    return np.reshape(normalized, [n, c, h, w])


if __name__ == '__main__':
    feature_map = np.random.random([10, 30, 64, 64])
    tensor = torch.from_numpy(feature_map)

    official_bn = torch.nn.BatchNorm2d(num_features = 30, affine=False, track_running_stats=False)(tensor)
    numpy_bn = batch_norm(feature_map)
    print('diff:{}'.format((official_bn.numpy() - numpy_bn).sum()))

    official_ln = torch.nn.LayerNorm(normalized_shape=[30, 64, 64], elementwise_affine=False)(tensor)
    numpy_ln = layer_norm(feature_map)
    print('diff:{}'.format((numpy_ln - official_ln.numpy()).sum()))

    official_in = torch.nn.InstanceNorm2d(num_features=30, affine=False, track_running_stats=False)(tensor)
    numpy_in = instance_norm(feature_map)
    print('diff:{}'.format((numpy_in - official_in.numpy()).sum()))

    official_gn = torch.nn.GroupNorm(num_groups=10, num_channels=30, affine=False)(tensor)
    numpy_gn = group_norm(feature_map, group = 10)
    print('diff:{}'.format((numpy_gn - official_gn.numpy()).sum()))
