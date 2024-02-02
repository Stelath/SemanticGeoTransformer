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

def main(args):
    cfg = make_cfg()
    test_loader, neighbor_limits = test_data_loader(cfg)
    
    # Create an array to store all the transforms & the finished pointcloud
    transforms = np.zeros((len(test_loader), 4, 4))
    
    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])
    model.eval()
    
    for i, batch in tqdm(enumerate(test_loader)):
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
        
        transforms[i] = estimated_transform
        
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
