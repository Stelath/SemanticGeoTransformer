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

def load_from_npy(npy_path):
    points = np.fromfile(npy_path, dtype=np.float32)
    
    # Randomly delete half of the indices
    indices = np.arange(len(points))
    np.random.shuffle(indices)
    indices = indices[:int(len(indices)*0.75)]
    points = np.delete(points, indices)
    
    return points

def load_data(ref_points, src_points):
    src_feats = np.ones_like(src_points)
    ref_feats = np.ones_like(ref_points)

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
        "transform": np.ones((4, 4)).astype(np.float32) # Dummy transform, we don't have ground truth
    }

    return data_dict

def main(args):
    # Get all the bin files (Point Clouds)
    files = glob.glob(os.path.join(args.data, '*.npy'))
    
    # Create an array to store all the transforms & the finished pointcloud
    transforms = np.zeros((len(files), 4, 4))
    
    cfg = make_cfg()

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])
    model.eval()
    
    
    test_loader, neighbor_limits = test_data_loader(cfg)
    
    ref_points = load_from_npy(files[0])
    for i, f in enumerate(tqdm(files[25:500:50]), start=1):
        src_points = load_from_npy(f)
        
        # prepare data
        data_dict = load_data(ref_points, src_points)
        data_dict = registration_collate_fn_stack_mode(
            [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
        )
        print("DATA LOADED")
        print(data_dict["points"])
        
        with torch.no_grad():
            # prediction
            data_dict = to_cuda(data_dict)
            output_dict = model(data_dict)
            data_dict = release_cuda(data_dict)
            output_dict = release_cuda(output_dict)

        print("MODEL PREDICTED")
        # get results
        ref_points_out = output_dict["ref_points"]
        src_points_out = output_dict["src_points"]
        estimated_transform = output_dict["estimated_transform"]
        
        transforms[i] = estimated_transform
        
        ref_points = src_points
        
        # Save the transforms every 50 iterations
        if i % 50 == 0:
            np.save('transforms.npy', transforms)
    
    np.save('transforms.npy', transforms)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Rellis dataset')
    parser.add_argument('-d', '--data', type=str, help='Path to Rellis dataset folder')
    parser.add_argument('-w', "--weights", required=True, help="model weights file")
    args = parser.parse_args()
    
    main(args)
