import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from geotransformer.utils.common import dump_pickle
from sklearn.model_selection import train_test_split

def main(args):
    sequences_dir = os.path.join(args.dataset, 'sequences')
    
    metadata = []
    for sequence in tqdm(os.listdir(sequences_dir), desc='Sequences'):
        sequence_dir = os.path.join(sequences_dir, sequence, 'os1_cloud_node_kitti_bin')
        if not os.path.isdir(sequence_dir):
            continue
        
        gt_transforms = pd.read_csv(os.path.join(sequences_dir, sequence, 'poses.txt'), header=None, sep=' ')
        
        for pointcloud in tqdm(sorted(os.listdir(sequence_dir)[0:-1:args.spacing // 4]), desc='Pointcloud Pairs', leave=False):
            p1_name = int(pointcloud.replace('.bin', ''))
            pcd1 = os.path.join('downsampled', sequence, str(p1_name).zfill(6) + '.npy')
            p1 = os.path.join(sequence_dir, pointcloud)
            p2_name = int(int(pointcloud.replace('.bin', '')) + args.spacing)
            pcd2 = os.path.join('downsampled', sequence, str(p2_name).zfill(6) + '.npy')
            p2 = os.path.join(sequence_dir, str(p2_name).zfill(6) + '.bin')
            
            if not os.path.isfile(p1) or not os.path.isfile(p2):
                break
            
            p1_gt_transform = np.concatenate([gt_transforms.iloc[p1_name].to_numpy().reshape(3,4), np.array([[0,0,0,1]])], axis=0)
            p2_gt_transform = np.concatenate([gt_transforms.iloc[p2_name].to_numpy().reshape(3,4), np.array([[0,0,0,1]])], axis=0)
            
            metadata.append({
                'seq_id': int(sequence),
                'frame0': p1_name,
                'frame1': p2_name,
                'transform': np.linalg.inv(p1_gt_transform) @ p2_gt_transform,
                'pcd0': pcd1,
                'pcd1': pcd2
            })
    
    # Split metadata into train, val, and test sets
    os.makedirs(os.path.join(args.dataset, 'metadata'), exist_ok=True)
    train_metadata, val_test_metadata = train_test_split(metadata, test_size=0.2, random_state=42)
    val_metadata, test_metadata = train_test_split(val_test_metadata, test_size=0.5, random_state=42)
    
    dump_pickle(train_metadata, os.path.join(args.dataset, 'metadata', 'train.pkl'))
    dump_pickle(val_metadata, os.path.join(args.dataset, 'metadata', 'val.pkl'))
    dump_pickle(test_metadata, os.path.join(args.dataset, 'metadata', 'test.pkl'))
    

if __name__ == '__main__':
    # Define the command line arguments
    parser = argparse.ArgumentParser(description='Create a metadata file for the RELLIS dataset')
    parser.add_argument('-d', '--dataset', type=str, help='The path to the dataset, formatted as outlined in the README.md', required=True)
    parser.add_argument('-s', '--spacing', type=int, help='The interval with which to sample the pointclouds', default=50)

    # Parse the command line arguments
    args = parser.parse_args()

    main(args)
