import importlib

ext_module = importlib.import_module('geotransformer.ext')

def grid_subsample_with_labels(points, labels, lengths, voxel_size):
    """Grid subsampling in stack mode with labels.

    This function is implemented on CPU.

    Args:
        points (Tensor): stacked points. (N, 3)
        labels (Tensor): semantic labels of points. (N,)
        lengths (Tensor): number of points in the stacked batch. (B,)
        voxel_size (float): voxel size.

    Returns:
        s_points (Tensor): stacked subsampled points (M, 3)
        s_labels (Tensor): semantic labels of subsampled points (M,)
        s_lengths (Tensor): numbers of subsampled points in the batch. (B,)
    """
    s_points, s_labels, s_lengths = ext_module.grid_subsampling_with_labels(
        points, labels, lengths, voxel_size
    )
    return s_points, s_labels, s_lengths
