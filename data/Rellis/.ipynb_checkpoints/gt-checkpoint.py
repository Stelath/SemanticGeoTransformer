import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from geotransformer.utils.common import dump_pickle

def process_sequences(sequences, sequences_dir, args):
    metadata = []
    for sequence in tqdm(sequences, desc='Sequences'):
        sequence_dir_full = os.path.join(sequences_dir, sequence, 'os1_cloud_node_kitti_bin')
        print(sequence_dir_full)
        
        if not os.path.isdir(sequence_dir_full):
            continue
        
        gt_transforms = pd.read_csv(os.path.join(sequences_dir, sequence, 'poses.txt'), header=None, sep=' ')
        
        # Select a starting point every 20 frames and pair with the pose 200 frames ahead
        for starting_index in range(0, len(gt_transforms) - 8, 32):  # Ensure frame1 does not go out of range
            p1_name = starting_index
            p2_name = starting_index + 8  # Each pair is 200 frames apart
            
            # Generate the corresponding point cloud file names and paths
            pcd1 = os.path.join('downsampled', sequence, str(p1_name).zfill(6) + '.npy')
            p1 = os.path.join(sequence_dir_full, str(p1_name).zfill(6) + '.bin')
            pcd2 = os.path.join('downsampled', sequence, str(p2_name).zfill(6) + '.npy')
            p2 = os.path.join(sequence_dir_full, str(p2_name).zfill(6) + '.bin')
            
            # Check if the .bin file paths exist
            if not os.path.isfile(p1):
                print(f"{p1} does not exist.")
            if not os.path.isfile(p2):
                print(f"{p2} does not exist.")

            # Check if the corresponding .npy files exist
            if not os.path.isfile(pcd1):
                print(f"{pcd1} does not exist.")
            if not os.path.isfile(pcd2):
                print(f"{pcd2} does not exist.")

            if not os.path.isfile(p1) or not os.path.isfile(p2):
                print(f'noooo')
                break
            
            # Retrieve the pose matrices for the two point clouds
            p1_gt_transform = np.concatenate([gt_transforms.iloc[p1_name].to_numpy().reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)
            p2_gt_transform = np.concatenate([gt_transforms.iloc[p2_name].to_numpy().reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)
            
            print(f'p1{p1_gt_transform}')
            
            # Add the paired data to the metadata
            metadata.append({
                'seq_id': int(sequence),
                'frame0': p1_name,
                'frame1': p2_name,
                'transform': np.linalg.inv(p2_gt_transform) @ p1_gt_transform,
                'pcd0': pcd1,
                'pcd1': pcd2
            })
    return metadata


def main(args):
    sequences_dir = os.path.join(args.dataset, 'sequences')
    
    # Split the dataset into training, validation, and test sets
    train_sequences = ['00', '01', '02']
    val_sequences = ['03']
    test_sequences = ['04']
    
    print(f'Training sequences: {train_sequences}')
    print(f'Validation sequence: {val_sequences}')
    print(f'Test sequence: {test_sequences}')
    
    # Process the training set
    train_metadata = process_sequences(train_sequences, sequences_dir, args)
    
    # Process the validation set
    val_metadata = process_sequences(val_sequences, sequences_dir, args)
    
    # Create the metadata directory if it doesn't exist
    os.makedirs(os.path.join(args.dataset, 'metadata'), exist_ok=True)
    
    # Save the training and validation metadata
    dump_pickle(train_metadata, os.path.join(args.dataset, 'metadata', 'train.pkl'))
    dump_pickle(val_metadata, os.path.join(args.dataset, 'metadata', 'val.pkl'))
    
    # Process the test set
    test_metadata = process_sequences(test_sequences, sequences_dir, args)
    
    # Save the test metadata
    dump_pickle(test_metadata, os.path.join(args.dataset, 'metadata', 'test.pkl'))

if __name__ == '__main__':
    # Define the command line arguments
    parser = argparse.ArgumentParser(description='Create a metadata file for the RELLIS dataset')
    parser.add_argument('-d', '--dataset', type=str, help='The path to the dataset, formatted as outlined in the README.md', required=True)
    parser.add_argument('-s', '--spacing', type=int, help='The interval with which to sample the pointclouds', default=32)
    parser.add_argument('-si', '--starting-indicies', type=list, help='The interval with which to sample the pointclouds', default=[0,8,16,24])
    parser.add_argument('--test-only', action='store_true', help='Only create the test metadata file')

    # Parse the command line arguments
    args = parser.parse_args()

    print(f'train, val and test')
    main(args)

