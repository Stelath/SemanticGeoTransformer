import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from geotransformer.utils.common import dump_pickle

def process_sequences(sequences, sequences_dir, args):
    metadata = []
    used_points = set()  # Set to keep track of used point clouds
    p0_positions = []  # List to track all chosen p0 positions

    for sequence in tqdm(sequences, desc='Sequences'):
        sequence_dir_full = os.path.join(sequences_dir, sequence, 'vel_cloud_node_kitti_bin')
        print(sequence_dir_full)
        
        if not os.path.isdir(sequence_dir_full):
            continue
        
        gt_transforms = pd.read_csv(os.path.join(sequences_dir, sequence, 'poses.txt'), header=None, sep=' ')
        point_cloud_files = sorted(os.listdir(sequence_dir_full))  # Obtain all point cloud file names in this scene.
        num_point_clouds = len(point_cloud_files)  # Calculate the total number of point cloud files.
        
        # Loop over the point cloud frames to find pairs based on a minimum fixed distance
        starting_index = 0  # Start from the first frame
        while starting_index < len(gt_transforms) - 1:
            if starting_index in used_points:
                starting_index += 1
                continue  # Skip if this point cloud has already been used
            
            p0_name = starting_index
            
            # Get the pose matrix for the starting point cloud
            p0_gt_transform = np.concatenate([gt_transforms.iloc[p0_name].to_numpy().reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)
            p0_position = p0_gt_transform[:3, 3]
            
            # Check distance to all previous p0 positions with a fixed distance
            too_close = False
            for previous_p0_position in p0_positions:
                dis_to_previous_p0 = np.linalg.norm(p0_position - previous_p0_position)
                if dis_to_previous_p0 < args.min_distance_to_last_pair:
                    too_close = True
                    break

            if too_close:
                starting_index += 1
                continue  # Skip this p0 if it is too close to any previous p0
            
            # Search for a suitable second point cloud frame based on a fixed minimum distance
            found_pair = False
            for i in range(p0_name + 1, len(gt_transforms)):
                p1_name = i
                
                # Skip if this point cloud has already been used
                if p1_name in used_points:
                    continue
                
                # Retrieve the pose matrix for the candidate point cloud
                p1_gt_transform = np.concatenate([gt_transforms.iloc[p1_name].to_numpy().reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)
                
                # Calculate the Euclidean distance between the two point cloud positions
                dis = np.linalg.norm(p1_gt_transform[:3, 3] - p0_position)
                
                # Check if the distance is greater than the fixed minimum distance
                if dis >= args.min_fixed_distance_m:
                    print(f"Pair found: {p0_name} and {p1_name} with distance {dis:.2f} meters")  # Log the distance
                    found_pair = True
                    break  # Found a suitable pair, exit the inner loop
                
            # Ensure that we found a valid pair
            if not found_pair:
                starting_index += 1
                continue
            
            # Mark these points as used
            used_points.add(p0_name)
            used_points.add(p1_name)
            
            # Save the p0 position for future distance checks
            p0_positions.append(p0_position)
            
            # Generate the corresponding point cloud file names and paths
            pcd0 = os.path.join('downsampled', sequence, str(p0_name).zfill(6) + '.npy')
            p0 = os.path.join(sequence_dir_full, str(p0_name).zfill(6) + '.bin')
            pcd1 = os.path.join('downsampled', sequence, str(p1_name).zfill(6) + '.npy')
            p1 = os.path.join(sequence_dir_full, str(p1_name).zfill(6) + '.bin')
            
            # Check if the .bin file paths exist
            missing_files = []
            if not os.path.isfile(p0):
                missing_files.append(p0)
            if not os.path.isfile(p1):
                missing_files.append(p1)
            if not os.path.isfile(pcd0):
                missing_files.append(pcd0)
            if not os.path.isfile(pcd1):
                missing_files.append(pcd1)
                
            # If any of the required files are missing, skip this pair
            if missing_files:
                print(f'Missing files: {", ".join(missing_files)}. Skipping this pair.')
                starting_index += 1
                continue

            # Add the paired data to the metadata
            metadata.append({
                'seq_id': int(sequence),
                'frame0': p0_name,
                'frame1': p1_name,
                'transform': np.linalg.inv(p0_gt_transform) @ p1_gt_transform,
                'pcd0': pcd0,
                'pcd1': pcd1
            })
            
            # Move `starting_index` forward by adding `min_distance_to_last_pair` equivalent number of indices
            # Assuming that index increments roughly represent spatial distance.
            # Use an increment strategy based on your actual data or requirement.
            increment = int(args.min_distance_to_last_pair / args.min_fixed_distance_m * 5)
            starting_index += increment

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
    output_folder = '/project/bli4/autoai/joyang/GeoTransformer/data/Rellis/metadata_new'
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the training and validation metadata
    dump_pickle(train_metadata, os.path.join(output_folder, 'train.pkl'))
    dump_pickle(val_metadata, os.path.join(output_folder, 'val.pkl'))
    
    # Process the test set
    test_metadata = process_sequences(test_sequences, sequences_dir, args)
    
    # Save the test metadata
    dump_pickle(test_metadata, os.path.join(output_folder, 'test.pkl'))

if __name__ == '__main__':
    # Define the command line arguments
    parser = argparse.ArgumentParser(description='Create a metadata file for the RELLIS dataset')
    parser.add_argument('-d', '--dataset', type=str, help='The path to the dataset, formatted as outlined in the README.md', required=True)
    parser.add_argument('-fd', '--min_fixed_distance_m', type=float, help='The minimum fixed distance (in meters) between two point clouds', default= 10.0)
    parser.add_argument('-mdlp', '--min_distance_to_last_pair', type=float, help='The minimum distance (in meters) between new p0 and any previous p0', default= 1.0)
    # parser.add_argument('--test-only', action='store_true', help='Only create the test metadata file')

    # Parse the command line arguments
    args = parser.parse_args()

    print(f'train, val and test')
    main(args)


#############################################################################################################################################
# import os
# import numpy as np
# import pandas as pd
# import argparse
# from tqdm import tqdm
# from geotransformer.utils.common import dump_pickle

# def process_sequences(sequences, sequences_dir, args):
#     metadata = []
#     used_points = set()  # Set to keep track of used point clouds
#     p0_positions = []  # List to track all chosen p0 positions

#     for sequence in tqdm(sequences, desc='Sequences'):
#         sequence_dir_full = os.path.join(sequences_dir, sequence, 'vel_cloud_node_kitti_bin')
#         print(sequence_dir_full)
        
#         if not os.path.isdir(sequence_dir_full):
#             continue
        
#         gt_transforms = pd.read_csv(os.path.join(sequences_dir, sequence, 'poses.txt'), header=None, sep=' ')
#         point_cloud_files = sorted(os.listdir(sequence_dir_full))  # Obtain all point cloud file names in this scene.
#         num_point_clouds = len(point_cloud_files)  # Calculate the total number of point cloud files.
        
#         # Loop over the point cloud frames to find pairs based on a minimum fixed distance
#         starting_index = 0  # Start from the first frame
#         while starting_index < len(gt_transforms) - 1:
#             if starting_index in used_points:
#                 starting_index += 1
#                 continue  # Skip if this point cloud has already been used
            
#             p0_name = starting_index
            
#             # Get the pose matrix for the starting point cloud
#             p0_gt_transform = np.concatenate([gt_transforms.iloc[p0_name].to_numpy().reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)
#             p0_position = p0_gt_transform[:3, 3]
            
#             # Check distance to all previous p0 positions
#             too_close = False
#             for previous_p0_position in p0_positions:
#                 dis_to_previous_p0 = np.linalg.norm(p0_position - previous_p0_position)
#                 if dis_to_previous_p0 < args.min_distance_to_last_pair:
#                     too_close = True
#                     break

#             if too_close:
#                 starting_index += 1
#                 continue  # Skip this p0 if it is too close to any previous p0
            
#             # Search for a suitable second point cloud frame based on a fixed minimum distance
#             found_pair = False
#             for i in range(starting_index + 1, len(gt_transforms)):
#                 p1_name = i
                
#                 # Skip if this point cloud has already been used
#                 if p1_name in used_points:
#                     continue
                
#                 # Retrieve the pose matrix for the candidate point cloud
#                 p1_gt_transform = np.concatenate([gt_transforms.iloc[p1_name].to_numpy().reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)
                
#                 # Calculate the Euclidean distance between the two point cloud positions
#                 dis = np.linalg.norm(p1_gt_transform[:3, 3] - p0_position)
                
#                 # Check if the distance is greater than the fixed minimum distance
#                 if dis >= args.min_fixed_distance_m:
#                     print(f"Pair found: {p0_name} and {p1_name} with distance {dis:.2f} meters")  # Log the distance
#                     found_pair = True
#                     break  # Found a suitable pair, exit the inner loop
                
#             # Ensure that we found a valid pair
#             if not found_pair:
#                 starting_index += 1
#                 continue
            
#             # Mark these points as used
#             used_points.add(p0_name)
#             used_points.add(p1_name)
            
#             # Save the p0 position for future distance checks
#             p0_positions.append(p0_position)
            
#             # Generate the corresponding point cloud file names and paths
#             pcd0 = os.path.join('downsampled', sequence, str(p0_name).zfill(6) + '.npy')
#             p0 = os.path.join(sequence_dir_full, str(p0_name).zfill(6) + '.bin')
#             pcd1 = os.path.join('downsampled', sequence, str(p1_name).zfill(6) + '.npy')
#             p1 = os.path.join(sequence_dir_full, str(p1_name).zfill(6) + '.bin')
            
#             # Check if the .bin file paths exist
#             missing_files = []
#             if not os.path.isfile(p0):
#                 missing_files.append(p0)
#             if not os.path.isfile(p1):
#                 missing_files.append(p1)
#             if not os.path.isfile(pcd0):
#                 missing_files.append(pcd0)
#             if not os.path.isfile(pcd1):
#                 missing_files.append(pcd1)
                
#             # If any of the required files are missing, skip this pair
#             if missing_files:
#                 print(f'Missing files: {", ".join(missing_files)}. Skipping this pair.')
#                 starting_index += 1
#                 continue

#             # Add the paired data to the metadata
#             metadata.append({
#                 'seq_id': int(sequence),
#                 'frame0': p0_name,
#                 'frame1': p1_name,
#                 'transform': np.linalg.inv(p1_gt_transform) @ p0_gt_transform,
#                 'pcd0': pcd0,
#                 'pcd1': pcd1
#             })
            
#             # Move `starting_index` forward to avoid closely located frames
#             starting_index = p1_name + 1

#     return metadata

# def main(args):
#     sequences_dir = os.path.join(args.dataset, 'sequences')
    
#     # Split the dataset into training, validation, and test sets
#     train_sequences = ['00', '01', '02']
#     val_sequences = ['03']
#     test_sequences = ['04']
    
#     print(f'Training sequences: {train_sequences}')
#     print(f'Validation sequence: {val_sequences}')
#     print(f'Test sequence: {test_sequences}')
    
#     # Process the training set
#     train_metadata = process_sequences(train_sequences, sequences_dir, args)
    
#     # Process the validation set
#     val_metadata = process_sequences(val_sequences, sequences_dir, args)
    
#     # Create the metadata directory if it doesn't exist
#     output_folder = '/project/bli4/maps/wacv/GeoTransformer/data/Rellis/metadata_new'
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Save the training and validation metadata
#     dump_pickle(train_metadata, os.path.join(output_folder, 'train.pkl'))
#     dump_pickle(val_metadata, os.path.join(output_folder, 'val.pkl'))
    
#     # Process the test set
#     test_metadata = process_sequences(test_sequences, sequences_dir, args)
    
#     # Save the test metadata
#     dump_pickle(test_metadata, os.path.join(output_folder, 'test.pkl'))

# if __name__ == '__main__':
#     # Define the command line arguments
#     parser = argparse.ArgumentParser(description='Create a metadata file for the RELLIS dataset')
#     parser.add_argument('-d', '--dataset', type=str, help='The path to the dataset, formatted as outlined in the README.md', required=True)
#     parser.add_argument('-fd', '--min_fixed_distance_m', type=float, help='The minimum fixed distance (in meters) between two point clouds', default=10.0)
#     parser.add_argument('-mdlp', '--min_distance_to_last_pair', type=float, help='The minimum distance (in meters) between new p0 and any previous p0', default=5.0)
#     parser.add_argument('--test-only', action='store_true', help='Only create the test metadata file')

#     # Parse the command line arguments
#     args = parser.parse_args()

#     print(f'train, val and test')
#     main(args)

    
############################################################################################################################################# 
# import os
# import numpy as np
# import pandas as pd
# import argparse
# from tqdm import tqdm
# from geotransformer.utils.common import dump_pickle

# def process_sequences(sequences, sequences_dir, args):
#     metadata = []
#     used_points = set()  # Set to keep track of used point clouds
#     last_p1_transform = None  # To keep track of the last p1's transformation
#     last_p1_name = None  # To keep track of the last p1's name

#     for sequence in tqdm(sequences, desc='Sequences'):
#         sequence_dir_full = os.path.join(sequences_dir, sequence, 'vel_cloud_node_kitti_bin')
#         print(sequence_dir_full)
        
#         if not os.path.isdir(sequence_dir_full):
#             continue
        
#         gt_transforms = pd.read_csv(os.path.join(sequences_dir, sequence, 'poses.txt'), header=None, sep=' ')
#         point_cloud_files = sorted(os.listdir(sequence_dir_full))  # Obtain all point cloud file names in this scene.
#         num_point_clouds = len(point_cloud_files)  # Calculate the total number of point cloud files.
        
#         # Loop over the point cloud frames to find pairs based on a minimum fixed distance
#         starting_index = 0  # Start from the first frame
#         while starting_index < len(gt_transforms) - 1:
#             if starting_index in used_points:
#                 starting_index += 1
#                 continue  # Skip if this point cloud has already been used
            
#             p0_name = starting_index
            
#             # Get the pose matrix for the starting point cloud
#             p0_gt_transform = np.concatenate([gt_transforms.iloc[p0_name].to_numpy().reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)
            
#             # Check distance to the last p1, if it exists
#             if last_p1_transform is not None:
#                 dis_to_last_p1 = np.linalg.norm(p0_gt_transform[:3, 3] - last_p1_transform[:3, 3])
#                 if dis_to_last_p1 < args.min_distance_to_last_pair:
#                     starting_index += 1
#                     continue  # Skip this p0 if it is too close to the last p1
            
#             # Search for a suitable second point cloud frame based on a fixed minimum distance
#             found_pair = False
#             for i in range(starting_index + 1, len(gt_transforms)):
#                 p1_name = i
                
#                 # Skip if this point cloud has already been used
#                 if p1_name in used_points:
#                     continue
                
#                 # Retrieve the pose matrix for the candidate point cloud
#                 p1_gt_transform = np.concatenate([gt_transforms.iloc[p1_name].to_numpy().reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)
                
#                 # Calculate the Euclidean distance between the two point cloud positions
#                 dis = np.linalg.norm(p1_gt_transform[:3, 3] - p0_gt_transform[:3, 3])
                
#                 # Check if the distance is greater than the fixed minimum distance
#                 if dis >= args.min_fixed_distance_m:
#                     print(f"Pair found: {p0_name} and {p1_name} with distance {dis:.2f} meters")  # Log the distance
#                     if last_p1_name is not None:
#                         print(f"Distance between {last_p1_name} and {p0_name}: {dis_to_last_p1:.2f} meters")
#                     found_pair = True
#                     break  # Found a suitable pair, exit the inner loop
                
#             # Ensure that we found a valid pair
#             if not found_pair:
#                 starting_index += 1
#                 continue
            
#             # Mark these points as used
#             used_points.add(p0_name)
#             used_points.add(p1_name)
            
#             # Update last_p1_transform and last_p1_name for next round of comparisons
#             last_p1_transform = p1_gt_transform
#             last_p1_name = p1_name
            
#             # Generate the corresponding point cloud file names and paths
#             pcd0 = os.path.join('downsampled', sequence, str(p0_name).zfill(6) + '.npy')
#             p0 = os.path.join(sequence_dir_full, str(p0_name).zfill(6) + '.bin')
#             pcd1 = os.path.join('downsampled', sequence, str(p1_name).zfill(6) + '.npy')
#             p1 = os.path.join(sequence_dir_full, str(p1_name).zfill(6) + '.bin')
            
#             # Check if the .bin file paths exist
#             missing_files = []
#             if not os.path.isfile(p0):
#                 missing_files.append(p0)
#             if not os.path.isfile(p1):
#                 missing_files.append(p1)
#             if not os.path.isfile(pcd0):
#                 missing_files.append(pcd0)
#             if not os.path.isfile(pcd1):
#                 missing_files.append(pcd1)
                
#             # If any of the required files are missing, skip this pair
#             if missing_files:
#                 print(f'Missing files: {", ".join(missing_files)}. Skipping this pair.')
#                 starting_index += 1
#                 continue

#             # Add the paired data to the metadata
#             metadata.append({
#                 'seq_id': int(sequence),
#                 'frame0': p0_name,
#                 'frame1': p1_name,
#                 'transform': np.linalg.inv(p1_gt_transform) @ p0_gt_transform,
#                 'pcd0': pcd0,
#                 'pcd1': pcd1
#             })
            
#             # Move `starting_index` forward to avoid closely located frames
#             starting_index = p1_name + 1

#     return metadata

# def main(args):
#     sequences_dir = os.path.join(args.dataset, 'sequences')
    
#     # Split the dataset into training, validation, and test sets
#     train_sequences = ['00', '01', '02']
#     val_sequences = ['03']
#     test_sequences = ['04']
    
#     print(f'Training sequences: {train_sequences}')
#     print(f'Validation sequence: {val_sequences}')
#     print(f'Test sequence: {test_sequences}')
    
#     # Process the training set
#     train_metadata = process_sequences(train_sequences, sequences_dir, args)
    
#     # Process the validation set
#     val_metadata = process_sequences(val_sequences, sequences_dir, args)
    
#     # Create the metadata directory if it doesn't exist
#     output_folder = '/project/bli4/autoai/joyang/GeoTransformer/data/Rellis/metadata'
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Save the training and validation metadata
#     dump_pickle(train_metadata, os.path.join(output_folder, 'train.pkl'))
#     dump_pickle(val_metadata, os.path.join(output_folder, 'val.pkl'))
    
#     # Process the test set
#     test_metadata = process_sequences(test_sequences, sequences_dir, args)
    
#     # Save the test metadata
#     dump_pickle(test_metadata, os.path.join(output_folder, 'test.pkl'))

# if __name__ == '__main__':
#     # Define the command line arguments
#     parser = argparse.ArgumentParser(description='Create a metadata file for the RELLIS dataset')
#     parser.add_argument('-d', '--dataset', type=str, help='The path to the dataset, formatted as outlined in the README.md', required=True)
#     parser.add_argument('-fd', '--min_fixed_distance_m', type=float, help='The minimum fixed distance (in meters) between two point clouds', default=10.0)
#     parser.add_argument('-mdlp', '--min_distance_to_last_pair', type=float, help='The minimum distance (in meters) between new p0 and the previous p1', default= - 5.0)
#     parser.add_argument('--test-only', action='store_true', help='Only create the test metadata file')

#     # Parse the command line arguments
#     args = parser.parse_args()

#     print(f'train, val and test')
#     main(args)
