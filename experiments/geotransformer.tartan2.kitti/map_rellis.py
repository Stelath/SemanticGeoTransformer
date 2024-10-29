"""
Uses a pre-trained model to predict the transforms for the Rellis dataset.

Some weights are stored in /zfs/ailab/VIPR/GeoTransformer/output/geotransformer.rellis/snapshots
"""

import os
import glob
import itertools
import numpy as np
import argparse
from tqdm import tqdm

import torch

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda

from config import make_cfg
from model import create_model

from dataset import test_data_loader

from geotransformer.utils.registration import (
    evaluate_sparse_correspondences,
    evaluate_correspondences,
    compute_registration_error,
)

def main(args):
    cfg = make_cfg()
    test_loader, neighbor_limits = test_data_loader(cfg)
    
    # Create an array to store all the transforms & the finished pointcloud
    estimated_transforms = np.zeros((len(test_loader), 4, 4))
    gt_transforms = np.zeros((len(test_loader), 4, 4))
    
    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])
    model.eval()

    rotational_error = []
    translational_error = []
    
    for i, batch in tqdm(enumerate(test_loader)):

        """
        Mess with the points here perhaps
        Batch might just be a single pair of pointclouds, but we'll see
        """

        with torch.no_grad():
            # prediction
            data_dict = to_cuda(batch)
            output_dict = model(data_dict)
            data_dict = release_cuda(data_dict)
            output_dict = release_cuda(output_dict)
        
        # get results
        ref_points_out = output_dict["ref_points"]
        src_points_out = output_dict["src_points"]
        estimated_transform = output_dict["estimated_transform"]
        
        estimated_transforms[i] = estimated_transform

        # Getting the translation between the two pointclouds so we have an idea how far apart they are
        # This is the same as the translation in the transform matrix

        # Combine all the axis of the translation into a single number
        print("Estimated distance between point clouds:", np.linalg.norm(estimated_transform[:3, 3]))


        # Save the transforms every 50 iterations
        if i % 32 == 0:
            np.save('transforms.npy', estimated_transforms)


    np.save('estimated_transforms.npy', estimated_transforms)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Rellis dataset')
    parser.add_argument('-d', '--data', type=str, help='Path to Rellis dataset folder')
    parser.add_argument('-w', "--weights", required=True, help="model weights file")
    args = parser.parse_args()
    
    main(args)
