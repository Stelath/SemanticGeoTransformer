# import os
# import numpy as np
# import pandas as pd
# import argparse
# from tqdm import tqdm
# from geotransformer.utils.common import dump_pickle

# def process_sequences(sequences, sequences_dir, args):
#     metadata = []
#     for sequence in tqdm(sequences, desc='Sequences'):
#         sequence_dir_full = os.path.join(sequences_dir, sequence, 'os1_cloud_node_color_ply')
#         print(sequence_dir_full)
        
#         if not os.path.isdir(sequence_dir_full):
#             continue
        
#         gt_transforms = pd.read_csv(os.path.join(sequences_dir, sequence, 'poses.txt'), header=None, sep=' ')
        
#         # Select a starting point every 5 frames and pair with the pose 50 frames ahead
#         for starting_index in range(0, len(gt_transforms) - 50, 5):  # Ensure frame1 does not go out of range
#             p1_name = starting_index
#             p2_name = starting_index + 50  # Each pair is 50 frames apart
            
#             # Generate the corresponding point cloud file names and paths
#             pcd1 = os.path.join('downsampled', sequence, str(p1_name).zfill(6) + '.npy')
#             p1 = os.path.join(sequence_dir_full, str(p1_name).zfill(6) + '.ply')
#             pcd2 = os.path.join('downsampled', sequence, str(p2_name).zfill(6) + '.npy')
#             p2 = os.path.join(sequence_dir_full, str(p2_name).zfill(6) + '.ply')
            
#             # Check if the .ply file paths exist
#             if not os.path.isfile(p1):
#                 print(f"{p1} does not exist.")
#             if not os.path.isfile(p2):
#                 print(f"{p2} does not exist.")

#             # Check if the corresponding .npy files exist
#             if not os.path.isfile(pcd1):
#                 print(f"{pcd1} does not exist.")
#             if not os.path.isfile(pcd2):
#                 print(f"{pcd2} does not exist.")

#             if not os.path.isfile(p1) or not os.path.isfile(p2):
#                 print(f'noooo')
#                 break
            
#             # Retrieve the pose matrices for the two point clouds
#             p1_gt_transform = np.concatenate([gt_transforms.iloc[p1_name].to_numpy().reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)
#             p2_gt_transform = np.concatenate([gt_transforms.iloc[p2_name].to_numpy().reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)
            
#             print(f'p1{p1_gt_transform}')
            
#             # Add the paired data to the metadata
#             metadata.append({
#                 'seq_id': int(sequence),
#                 'frame0': p1_name,
#                 'frame1': p2_name,
#                 'transform': np.linalg.inv(p1_gt_transform) @ p2_gt_transform,
#                 'pcd0': pcd1,
#                 'pcd1': pcd2
#             })
#     return metadata


# def main(args):
#     sequences_dir = os.path.join(args.dataset, 'sequences')
    
#     # Split the dataset into training, validation, and test sets
#     train_sequences = ['00000', '00001', '00002']
#     val_sequences = ['00003']
#     test_sequences = ['00004']
    
#     print(f'Training sequences: {train_sequences}')
#     print(f'Validation sequence: {val_sequences}')
#     print(f'Test sequence: {test_sequences}')
    
#     # Process the training set
#     train_metadata = process_sequences(train_sequences, sequences_dir, args)
    
#     # Process the validation set
#     val_metadata = process_sequences(val_sequences, sequences_dir, args)
    
#     # Create the metadata directory if it doesn't exist
#     os.makedirs(os.path.join(args.dataset, 'metadata'), exist_ok=True)
    
#     # Save the training and validation metadata
#     dump_pickle(train_metadata, os.path.join(args.dataset, 'metadata', 'train.pkl'))
#     dump_pickle(val_metadata, os.path.join(args.dataset, 'metadata', 'val.pkl'))
    
#     # Process the test set
#     test_metadata = process_sequences(test_sequences, sequences_dir, args)
    
#     # Save the test metadata
#     dump_pickle(test_metadata, os.path.join(args.dataset, 'metadata', 'test.pkl'))

# if __name__ == '__main__':
#     # Define the command line arguments
#     parser = argparse.ArgumentParser(description='Create a metadata file for the RELLIS dataset')
#     parser.add_argument('-d', '--dataset', type=str, help='The path to the dataset, formatted as outlined in the README.md', required=True)
#     parser.add_argument('-s', '--spacing', type=int, help='The interval with which to sample the pointclouds', default=32)
#     parser.add_argument('-si', '--starting-indicies', type=list, help='The interval with which to sample the pointclouds', default=[0,8,16,24])
#     parser.add_argument('--test-only', action='store_true', help='Only create the test metadata file')

#     # Parse the command line arguments
#     args = parser.parse_args()

#     print(f'train, val and test')
#     main(args)
import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from geotransformer.utils.common import dump_pickle

def process_sequences(sequences, sequences_dir, args):
    metadata = []
    index_delta = 100
    for sequence in tqdm(sequences, desc='Sequences'):
        sequence_lidar_dir = os.path.join(sequences_dir, sequence, 'lidar_data', '00000')
        sequence_downsampled_dir = os.path.join(sequences_dir, sequence, 'downsampled', '00000')
        print(f"Processing sequence: {sequence}")
        
        if not os.path.isdir(sequence_lidar_dir):
            print(f"{sequence_lidar_dir} is not a valid directory")
            continue
        
        gt_transforms = pd.read_csv(os.path.join(sequences_dir, sequence, 'poses.txt'), header=None, sep=' ')
        
        # Get all .bin files in the folder and sort them by file name
        bin_files = sorted([f for f in os.listdir(sequence_lidar_dir) if f.endswith('.bin')])
        
        if len(bin_files) < index_delta:
            print(f"Not enough .bin files for pairing in {sequence_lidar_dir}")
            continue
        
        # Traverse and match point cloud files
        for starting_index in range(0, len(bin_files) - index_delta, 5):
            p1_name = bin_files[starting_index]
            p2_name = bin_files[starting_index + index_delta]
            
            pcd1 = os.path.join(sequence_downsampled_dir, p1_name.replace('.bin', '.npy'))
            p1 = os.path.join(sequence_lidar_dir, p1_name)
            pcd2 = os.path.join(sequence_downsampled_dir, p2_name.replace('.bin', '.npy'))
            p2 = os.path.join(sequence_lidar_dir, p2_name)
            
            # Check if the file exists
            if not os.path.isfile(p1) or not os.path.isfile(p2):
                print(f"Skipping pair {p1_name} and {p2_name}: Files not found.")
                continue
            if not os.path.isfile(pcd1) or not os.path.isfile(pcd2):
                print(f"Skipping pair {p1_name} and {p2_name}: Downsampled files not found.")
                continue

            print(f"Pairing {p1_name} with {p2_name}")
            
            # Get the corresponding pose matrix
            p1_gt_transform = np.concatenate([gt_transforms.iloc[starting_index].to_numpy().reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)
            p2_gt_transform = np.concatenate([gt_transforms.iloc[starting_index + index_delta].to_numpy().reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)
            
            # Add paired data to metadata, set seq_id to 0 here
            metadata.append({
                'seq_id': 0,  # Set all seq_id to 0
                'frame0': p1_name,
                'frame1': p2_name,
                'transform': np.linalg.inv(p1_gt_transform) @ p2_gt_transform,
                'pcd0': pcd1,
                'pcd1': pcd2
            })

            print("transform", np.linalg.inv(p1_gt_transform) @ p2_gt_transform)
    
    print(f"Generated {len(metadata)} pairs for sequence {sequence}")
    return metadata


def main(args):
    # Update sequences_dir to directly point to the root directory of the dataset, without the need to concatenate 'Final_Dataset' again
    sequences_dir = args.dataset
    
    # Define training, validation, and test sets
    train_sequences = ['training']
    val_sequences = ['validation']
    test_sequences = ['testing']
    
    print(f'Training sequences: {train_sequences}')
    print(f'Validation sequence: {val_sequences}')
    print(f'Test sequence: {test_sequences}')
    
    # Process training set
    train_metadata = process_sequences(train_sequences, sequences_dir, args)
    
    # Process validation set
    val_metadata = process_sequences(val_sequences, sequences_dir, args)
    
    # If the metadata directory does not exist, create it
    os.makedirs(os.path.join(args.dataset, 'metadata'), exist_ok=True)
    
    # Save metadata for training set and validation set
    dump_pickle(train_metadata, os.path.join(args.dataset, 'metadata', 'train.pkl'))
    dump_pickle(val_metadata, os.path.join(args.dataset, 'metadata', 'val.pkl'))
    
    # Process test set
    test_metadata = process_sequences(test_sequences, sequences_dir, args)
    
    # Save metadata of the test set
    dump_pickle(test_metadata, os.path.join(args.dataset, 'metadata', 'test.pkl'))


if __name__ == '__main__':
    # Define command line parameters
    parser = argparse.ArgumentParser(description='Create a metadata file for the ORFD dataset')
    parser.add_argument('-d', '--dataset', type=str, help='The path to the dataset, formatted as outlined in the README.md', required=True)
    parser.add_argument('-s', '--spacing', type=int, help='The interval with which to sample the pointclouds', default=32)
    parser.add_argument('-si', '--starting-indicies', type=list, help='The interval with which to sample the pointclouds', default=[0,8,16,24])
    parser.add_argument('--test-only', action='store_true', help='Only create the test metadata file')

    # Parsing command line arguments
    args = parser.parse_args()

    print(f'train, val and test')
    main(args)

