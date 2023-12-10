import os
import glob
import itertools
import numpy as np
import argparse
from tqdm import tqdm

import torch

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.registration import compute_registration_error

from config import make_cfg
from model import create_model

def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    
    total_elements = obj.shape[0]
    elements_to_remove = int(total_elements * 8 / 10)
    indices_to_remove = np.random.choice(total_elements, elements_to_remove, replace=False)
    obj = np.delete(obj, indices_to_remove, axis=0)
    
    normalization_num = 45
    
    # ignore reflectivity info
    return obj[:,:3] / normalization_num

def load_data(ref_points, src_points):
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

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
    files = glob.glob(os.path.join(args.data, '*.bin'))
    
    # Create an array to store all the transforms & the finished pointcloud
    transforms = np.zeros((len(files), 4, 4))
    
    cfg = make_cfg()

    # prepare model
    model = create_model(cfg).cuda()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict["model"])
    
    ref_points = load_from_bin(files[0])
    print("NUM POINTS:", ref_points.shape)
    for i, f in enumerate(tqdm(files[1:75:5]), start=1):
        src_points = load_from_bin(f)
        
        # prepare data
        data_dict = load_data(ref_points, src_points)
        neighbor_limits = [38, 36, 36, 38]  # default setting in 3DMatch
        data_dict = registration_collate_fn_stack_mode(
            [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
        )
        
        with torch.no_grad():
            # prediction
            data_dict = to_cuda(data_dict)
            output_dict = model(data_dict)
            data_dict = release_cuda(data_dict)
            output_dict = release_cuda(output_dict)

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
