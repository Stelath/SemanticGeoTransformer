import os
import glob
import numpy as np
import pandas as pd

import pyvista as pv
pv.global_theme.trame.server_proxy_enabled = True


from geotransformer.utils.pointcloud import apply_transform

from geotransformer.utils.common import load_pickle

def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj[:,:3]

def load_from_npy(npy_path):
    obj = np.load(npy_path)
    return obj

# Load Transforms
transforms = np.load('estimated_transforms.npy')
gt_transforms = pd.read_csv('/zfs/ailab/VIPR/GeoTransformer/data/Rellis/sequences/04/poses.txt', header=None, sep=' ')

# Get all the bin files (Point Clouds)
bin_files = sorted(glob.glob(os.path.join('/zfs/ailab/VIPR/GeoTransformer/data/Rellis/sequences/04/os1_cloud_node_kitti_bin/', '*.bin')))

test_pkl = load_pickle('/zfs/ailab/VIPR/GeoTransformer/data/Rellis/metadata/test.pkl')

import numpy as np

# Initialize variables
translation_distances = []
num_transforms = len(gt_transforms)

assert(len(transforms) == len(gt_transforms))

# Iterate through each transform
for i in range(num_transforms):
    # Get the translation components of the transform
    gt_translation = np.concatenate([gt_transforms.iloc[i].to_numpy().reshape(3,4), np.array([[0,0,0,1]])], axis=0)[:3, 3]

    
    # Calculate the Euclidean distance of the translation
    translation_distance = np.linalg.norm(gt_translation)
    
    # Append the translation distance to the list
    translation_distances.append(translation_distance)

start = 25
skip = 50

first_transform = np.concatenate([gt_transforms.iloc[start].to_numpy().reshape(3,4), np.array([[0,0,0,1]])], axis=0)
gt_transforms_log = [first_transform]


end = start+(skip*6) #len(bin_files)

num_transforms = len(gt_transforms)

point_cloud = apply_transform(load_from_bin(bin_files[start]), first_transform)
for i, bin_file in enumerate(bin_files[start+skip:end:skip]):
    idx = (i + 1) * skip + start
    # Load Point Cloud
    src_point_cloud = load_from_bin(bin_file)
    
    # Get New Transform
    gt_transform = np.concatenate([gt_transforms.iloc[idx].to_numpy().reshape(3,4), np.array([[0,0,0,1]])], axis=0)
    
    gt_transforms_log.append(gt_transform)
    
    # transform = np.matmul(transform, gt_transform)
    transformed_point_cloud = apply_transform(src_point_cloud, gt_transform)
    
    # Apply Transform
    point_cloud = np.append(point_cloud, transformed_point_cloud, axis=0)#apply_transform(src_point_cloud, transform))