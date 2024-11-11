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

    iteration = tqdm(range(0, num_frames - 1, args.spacing), desc=f"Processing {sequence}")
    for starting_index in iteration:
        attempts += 1
        p1_idx = frame_indices[starting_index]
        p1_gt_pose = positions.get(p1_idx)
        if p1_gt_pose is None:
            continue

        p1_name = str(p1_idx).zfill(6)  # Retain leading zeros for pcd file path
        p1_frame = str(p1_idx)  # Remove leading zeros for frame

        # Find the next point cloud within the specified distance range
        for next_idx in range(starting_index + 1, num_frames):
            p2_idx = frame_indices[next_idx]
            p2_gt_pose = positions.get(p2_idx)
            if p2_gt_pose is None:
                continue

            dis_squared = np.sum((p2_gt_pose[:3, 3] - p1_gt_pose[:3, 3]) ** 2)
            distance = np.sqrt(dis_squared)

            if args.min_spacing_m <= distance <= args.max_spacing_m:
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
                break  # Move to next p1_idx after finding a valid pair

    if distances:
        avg_distance = np.mean(distances)
        median_distance = np.median(distances)
        print(f"Processed {sequence} with {len(metadata)} pairs out of {attempts} attempts")
        print(f"Average distance between frames: {avg_distance:.2f} meters")
        print(f"Median distance between frames: {median_distance:.2f} meters")
    else:
        print(f"No valid pairs found in {sequence}")

    return metadata, distances

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

    return metadata

def plot_sequences(metadata):
    # Create two subplots
    sequences = [m['seq_id'] for m in metadata]

    # Remove duplicates while preserving order
    seen = set()
    sequences = [x for x in sequences if not (x in seen or seen.add(x))]

    sequences = sequences[:3]

    print(sequences)

    fig, axs = plt.subplots(1, len(sequences), figsize=(15, 5))

    all_points = {}

    # Plot where each point cloud is located as dots on a 2D plane
    distances = {}
    for i in range(len(metadata)):
        seq_id = metadata[i]['seq_id']
        if seq_id not in distances:
            distances[seq_id] = []
        if seq_id not in all_points:
            all_points[seq_id] = []

        x1 = metadata[i]['pc0_pose'][0, 3]
        y1 = metadata[i]['pc0_pose'][1, 3]

        x2 = metadata[i]['pc1_pose'][0, 3]
        y2 = metadata[i]['pc1_pose'][1, 3]

        distances[seq_id].append(metadata[i]['distance'])
        frame = metadata[i]['frame0']

        if seq_id not in sequences:
            continue
        axs[sequences.index(seq_id)].scatter(x1, y1, label=frame, zorder=100)
        axs[sequences.index(seq_id)].scatter(x2, y2, label=frame, zorder=100)
        # Draw a line connected
        axs[sequences.index(seq_id)].plot([x1, x2], [y1, y2], 'k-', zorder=101)

        # Plot the two point clouds (black dots)
        pc1 = np.load(metadata[i]['pcd0'])
        pc2 = np.load(metadata[i]['pcd1'])

        # Filter out the ground
        pc1_height_threshold = np.percentile(pc1[:, 2], 95)
        pc2_height_threshold = np.percentile(pc2[:, 2], 95)

        pc1 = pc1[pc1[:, 2] > pc1_height_threshold]
        pc2 = pc2[pc2[:, 2] > pc2_height_threshold]

        # Add the x and y values
        pc1[:, 0] += x1
        pc1[:, 1] += y1

        # Normalize z values
        norm = Normalize(vmin=min(pc1[:, 2].min(), pc2[:, 2].min()), vmax=max(pc1[:, 2].max(), pc2[:, 2].max()))

        all_points[seq_id].extend(pc1)
        all_points[seq_id].extend(pc2)

    # For each sequence, sort all the points by zorder and then plot them
    for seq_id in sequences:
        all_points[seq_id] = np.array(all_points[seq_id])
        all_points[seq_id] = all_points[seq_id][all_points[seq_id][:, 2].argsort()]

        points = all_points[seq_id]

        # Create a colormap
        cmap = plt.get_cmap('viridis')

        # Map z values to colors
        colors = cmap(norm(points[:, 2]))

        # Plot the points with colors based on z values
        axs[sequences.index(seq_id)].scatter(points[:, 0], points[:, 1], c=colors, s=0.4, marker=',')

    # Set titles for the subplots
    for i, ax in enumerate(axs):
        ax.set_title(sequences[i] + f"\n(Average distance (m): {np.mean(distances[sequences[i]]):.2f})")

    # Show the plot
    plt.tight_layout()
    plt.savefig('point_clouds_rellis.png')
    print(os.getcwd() + '/point_clouds_rellis.png')

def main(args):
    sequences_dir = os.path.join('.', 'sequences')
    sequences = sorted(os.listdir(sequences_dir))

    train_sequences = sequences[:-2]
    val_sequences = [sequences[-2]]
    test_sequences = [sequences[-1]]

    print(f'Training sequences: {train_sequences}')
    print(f'Validation sequence: {val_sequences}')
    print(f'Test sequence: {test_sequences}')

    print("Processing Training Sequences")
    train_metadata = process_sequences(train_sequences, args)

    print("Processing Validation Sequence")
    val_metadata = process_sequences(val_sequences, args)

    print("Processing Test Sequence")
    test_metadata = process_sequences(test_sequences, args)

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
        f.write(f"All args: {args}\n")

    if args.plot_sequences:
        plot_sequences(pd.read_pickle(os.path.join('.', 'metadata', 'train.pkl')))
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a metadata file for the RELLIS dataset')
    parser.add_argument('-s_min', '--min_spacing_m', type=float, help='The minimum distance between point clouds', default=14.8)
    parser.add_argument('-s_max', '--max_spacing_m', type=float, help='The maximum distance between point clouds', default=15.2)
    parser.add_argument('--spacing', type=int, help='The number of frames to skip between each starting frame', default=8)
    parser.add_argument('--plot-sequences', action='store_true', help='Plot the sequences')
    args = parser.parse_args()

    main(args)
