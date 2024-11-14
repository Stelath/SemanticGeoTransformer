import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from geotransformer.utils.common import dump_pickle
import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def get_positions(sequence_dir):
    """
    Gets the positions of the point clouds in a sequence.
    """
    positions = {}
    poses_file = os.path.join(sequence_dir, 'poses.txt')
    if not os.path.isfile(poses_file):
        print(f"poses.txt not found in {sequence_dir}")
        return positions

    gt_transforms = pd.read_csv(poses_file, header=None, sep=' ')
    for idx in range(len(gt_transforms)):
        pose = gt_transforms.iloc[idx].to_numpy()
        pose = np.concatenate([pose.reshape(3, 4), np.array([[0, 0, 0, 1]])], axis=0)
        positions[idx] = pose

    return positions

def process_sequence(sequence, positions, args):
    metadata = []
    distances = []
    attempts = 0

    frame_indices = sorted(positions.keys())
    num_frames = len(frame_indices)

    # First pass: process with regular spacing
    iteration = tqdm(range(0, num_frames - 1, args.spacing), desc=f"Processing {sequence} (1st pass)")
    for starting_index in iteration:
        metadata, distances, attempts = process_pair(sequence, positions, args, frame_indices, starting_index, metadata, distances, attempts)

    # Second pass: process with offset spacing
    offset = args.spacing // 2
    iteration = tqdm(range(offset, num_frames - 1, args.spacing), desc=f"Processing {sequence} (2nd pass)")
    for starting_index in iteration:
        metadata, distances, attempts = process_pair(sequence, positions, args, frame_indices, starting_index, metadata, distances, attempts)

    if distances:
        avg_distance = np.mean(distances)
        median_distance = np.median(distances)
        print(f"Processed {sequence} with {len(metadata)} pairs out of {attempts} attempts")
        print(f"Average distance between frames: {avg_distance:.2f} meters")
        print(f"Median distance between frames: {median_distance:.2f} meters")
    else:
        print(f"No valid pairs found in {sequence}")

    return metadata, distances

def process_pair(sequence, positions, args, frame_indices, starting_index, metadata, distances, attempts):
    attempts += 1
    p1_idx = frame_indices[starting_index]
    p1_gt_pose = positions.get(p1_idx)
    if p1_gt_pose is None:
        return metadata, distances, attempts

    p1_name = str(p1_idx).zfill(6)  # Retain leading zeros for pcd file path
    p1_frame = str(p1_idx)  # Remove leading zeros for frame

    # Find all point clouds within the specified distance range
    valid_pairs = []
    for next_idx in range(starting_index + 1, len(frame_indices)):
        p2_idx = frame_indices[next_idx]
        p2_gt_pose = positions.get(p2_idx)
        if p2_gt_pose is None:
            continue

        dis_squared = np.sum((p2_gt_pose[:3, 3] - p1_gt_pose[:3, 3]) ** 2)
        distance = np.sqrt(dis_squared)

        if args.min_spacing_m <= distance <= args.max_spacing_m:
            valid_pairs.append((p2_idx, p2_gt_pose, distance))

    # If valid pairs found, randomly select one if random option enabled, otherwise take the first one
    if valid_pairs:
        if args.random:
            p2_idx, p2_gt_pose, distance = valid_pairs[np.random.randint(len(valid_pairs))]
        else:
            p2_idx, p2_gt_pose, distance = valid_pairs[0]

        p2_name = str(p2_idx).zfill(6)  # Retain leading zeros for pcd file path
        p2_frame = str(p2_idx)  # Remove leading zeros for frame

        transform = np.linalg.inv(p1_gt_pose) @ p2_gt_pose

        this_metadata = {
            'seq_id': sequence.lstrip("0"),  # Remove leading zeros from sequence ID
            'frame0': p1_frame,
            'frame1': p2_frame,
            'transform': transform,
            'pcd0': os.path.join('downsampled', sequence, f"{p1_name}.npy"),
            'pcd1': os.path.join('downsampled', sequence, f"{p2_name}.npy"),
            'pc0_pose': p1_gt_pose,
            'pc1_pose': p2_gt_pose,
            'distance': distance
        }

        metadata.append(this_metadata)
        distances.append(distance)

    return metadata, distances, attempts

def process_sequences(sequences, args):
    metadata = []
    all_distances = []

    for sequence in sequences:
        sequence_dir = os.path.join('.', 'sequences', sequence)
        print(f"Processing sequence {sequence}")
        positions = get_positions(sequence_dir)
        if not positions:
            continue

        seq_metadata, distances = process_sequence(sequence, positions, args)
        metadata.extend(seq_metadata)
        all_distances.extend(distances)

    if all_distances:
        overall_avg_distance = np.mean(all_distances)
        overall_median_distance = np.median(all_distances)
        print(f"Overall average distance between frames: {overall_avg_distance:.2f} meters")
        print(f"Overall median distance between frames: {overall_median_distance:.2f} meters")
    else:
        print("No valid pairs found in any sequence")

    return metadata, overall_avg_distance, overall_median_distance

def main(args):
    sequences_dir = os.path.join('.', 'sequences')
    sequences = sorted(os.listdir(sequences_dir))

    train_sequences = sequences[:6]
    val_sequences = sequences[6:8]
    test_sequences = sequences[8:11]

    print(f'Training sequences: {train_sequences}')
    print(f'Validation sequence: {val_sequences}')
    print(f'Test sequence: {test_sequences}')

    print("Processing Training Sequences")
    train_metadata, train_avg_distance, train_median_distance = process_sequences(train_sequences, args)

    print("Processing Validation Sequence")
    val_metadata, val_avg_distance, val_median_distance = process_sequences(val_sequences, args)

    print("Processing Test Sequence")
    test_metadata, test_avg_distance, test_median_distance = process_sequences(test_sequences, args)

    metadata_dir = os.path.join('.', 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)
    dump_pickle(train_metadata, os.path.join(metadata_dir, 'train.pkl'))
    dump_pickle(val_metadata, os.path.join(metadata_dir, 'val.pkl'))
    dump_pickle(test_metadata, os.path.join(metadata_dir, 'test.pkl'))

    with open(os.path.join(metadata_dir, 'metadata.info'), 'w') as f:
        f.write(f"DO NOT EDIT THIS FILE (AUTOMATICALLY GENERATED)\n\n")
        f.write(f"Created on {datetime.datetime.now()}\n")
        f.write(f"Total number of sequences: {len(sequences)}\n")
        f.write(f"Sequences in the training set: {train_sequences}\n")
        f.write(f"Sequences in the validation set: {val_sequences}\n")
        f.write(f"Sequences in the test set: {test_sequences}\n")
        f.write(f"Number of pairs in the training set: {len(train_metadata)}\n")
        f.write(f"Number of pairs in the validation set: {len(val_metadata)}\n")
        f.write(f"Number of pairs in the test set: {len(test_metadata)}\n")
        f.write(f"Average distance in training set: {train_avg_distance:.2f} meters\n")
        f.write(f"Median distance in training set: {train_median_distance:.2f} meters\n")
        f.write(f"Average distance in validation set: {val_avg_distance:.2f} meters\n")
        f.write(f"Median distance in validation set: {val_median_distance:.2f} meters\n")
        f.write(f"Average distance in test set: {test_avg_distance:.2f} meters\n")
        f.write(f"Median distance in test set: {test_median_distance:.2f} meters\n")
        f.write(f"All args: {args}\n")

        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a metadata file for the KITTI dataset')
    parser.add_argument('-s_min', '--min_spacing_m', type=float, help='The minimum distance between point clouds', default=14.8)
    parser.add_argument('-s_max', '--max_spacing_m', type=float, help='The maximum distance between point clouds', default=15.2)
    parser.add_argument('--spacing', type=int, help='The number of frames to skip between each starting frame', default=8)
    parser.add_argument('--random', action='store_true', help='Randomly select point clouds within the distance range')
    args = parser.parse_args()

    main(args)



