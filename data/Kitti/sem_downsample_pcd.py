import os
import os.path as osp
import open3d as o3d
import numpy as np
import glob
from tqdm import tqdm
import multiprocessing


def process_file(file_name, seq_id):
    frame = file_name.split('/')[-1][:-4]
    new_file_name = osp.join('downsampled', seq_id, frame + '.npy')
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(0.3)
    points = np.array(pcd.points).astype(np.float32)
    os.makedirs(osp.dirname(new_file_name), exist_ok=True)
    np.save(new_file_name, points)

def process_pcd_with_labels(file_name, label_file_name, seq_id):
    try:
        print(f"Processing: {osp.basename(file_name)} - {osp.basename(label_file_name)}")
        
        frame = file_name.split('/')[-1][:-4]
        new_file_name = osp.join('downsampled', seq_id, frame + '.npy')
        
        # Load points and labels
        points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
        positions = points[:, :3]
        labels = load_kitti_labels(label_file_name)
        
        voxel_size = 0.3
        # Compute voxel indices
        voxel_indices = np.floor(positions / voxel_size).astype(np.int32)
        # Create keys by combining voxel indices and labels
        keys = np.c_[voxel_indices, labels]
        # Find unique keys and inverse indices
        unique_keys, inverse_indices = np.unique(keys, axis=0, return_inverse=True)
        num_unique = len(unique_keys)
        # Initialize sums and counts
        sums = np.zeros((num_unique, 3), dtype=np.float64)
        counts = np.zeros(num_unique, dtype=np.int32)
        # Accumulate sums and counts
        np.add.at(sums, inverse_indices, positions)
        np.add.at(counts, inverse_indices, 1)
        # Compute mean positions
        mean_positions = sums / counts[:, None]
        # Get labels
        downsampled_labels = unique_keys[:, 3]
        # Combine mean positions and labels
        downsampled_points = np.hstack([mean_positions, downsampled_labels.reshape(-1,1)])
        
        # Save downsampled points
        os.makedirs(osp.dirname(new_file_name), exist_ok=True)
        np.save(new_file_name, downsampled_points.astype(np.float32))
    except Exception as e:
        print(f"Error processing {osp.basename(file_name)} - {osp.basename(label_file_name)}: {str(e)}")
        raise  # Re-raise the exception to be caught by the main function

def load_kitti_labels(label_file_path, return_label_array=False):
    # Load labels as 32-bit unsigned integers
    labels = np.fromfile(label_file_path, dtype=np.uint32)

    # Extract the semantics by masking out the instance id (only keep the lower 16 bits)
    semantic_labels = labels & 0xFFFF  # Only lower 16 bits are semantic label

    semantic_map = {
        0 : "unlabeled",
        1 : "outlier",
        10: "car",
        11: "bicycle",
        13: "bus",
        15: "motorcycle",
        16: "on-rails",
        18: "truck",
        20: "other-vehicle",
        30: "person",
        31: "bicyclist",
        32: "motorcyclist",
        40: "road",
        44: "parking",
        48: "sidewalk",
        49: "other-ground",
        50: "building",
        51: "fence",
        52: "other-structure",
        60: "lane-marking",
        70: "vegetation",
        71: "trunk",
        72: "terrain",
        80: "pole",
        81: "traffic-sign",
        99: "other-object",
        252: "moving-car",
        253: "moving-bicyclist",
        254: "moving-person",
        255: "moving-motorcyclist",
        256: "moving-on-rails",
        257: "moving-bus",
        258: "moving-truck",
        259: "moving-other-vehicle",
    }
    
    learning_map = {
        0 : 0,     # "unlabeled"
        1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
        10: 1,     # "car"
        11: 2,     # "bicycle"
        13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
        15: 3,     # "motorcycle"
        16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
        18: 4,     # "truck"
        20: 5,     # "other-vehicle"
        30: 6,     # "person"
        31: 7,     # "bicyclist"
        32: 8,     # "motorcyclist"
        40: 9,     # "road"
        44: 10,    # "parking"
        48: 11,    # "sidewalk"
        49: 12,    # "other-ground"
        50: 13,    # "building"
        51: 14,    # "fence"
        52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
        60: 9,     # "lane-marking" to "road" ---------------------------------mapped
        70: 15,    # "vegetation"
        71: 16,    # "trunk"
        72: 17,    # "terrain"
        80: 18,    # "pole"
        81: 19,    # "traffic-sign"
        99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
        252: 20,    # "moving-car"
        253: 21,    # "moving-bicyclist"
        254: 22,    # "moving-person"
        255: 23,    # "moving-motorcyclist"
        256: 24,    # "moving-on-rails" mapped to "moving-other-vehicle" ------mapped
        257: 24,    # "moving-bus" mapped to "moving-other-vehicle" -----------mapped
        258: 25,    # "moving-truck"
        259: 24,    # "moving-other-vehicle"
    }
    
    # Create a mapping from old label values to new label values
    old_to_new = {0: 0}  # 'void' keeps its value
    current_new_value = 1
    for old_value in sorted(set(semantic_map.values()) - {0}):
        old_to_new[old_value] = current_new_value
        current_new_value += 1

    # Use numpy's vectorize to apply the remapping efficiently
    remapped_labels = np.vectorize(lambda x: learning_map[x])(semantic_labels)

    if return_label_array:
        # If you need the labels as strings, you can create a reverse mapping
        reverse_semantic_map = {learning_map[v]: k for k, v in semantic_map.items()}
        remapped_label_strings = np.vectorize(lambda x: reverse_semantic_map.get(x, 'unknown'))(remapped_labels)
        
        return remapped_labels, remapped_label_strings
    
    return remapped_labels

def old_main():
    num_cores = 16 #multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    
    for i in range(5):
        seq_id = '{:02d}'.format(i)
        file_names = glob.glob(osp.join('sequences', seq_id, 'velodyne', '*.bin'))
        
        pool.starmap(process_file, [(file_name, seq_id) for file_name in file_names])
    
    pool.close()
    pool.join()
    
def main():
    num_cores = 16 #multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    
    for i in range(5):
        seq_id = '{:02d}'.format(i)
        file_names = sorted(glob.glob(osp.join('sequences', seq_id, 'velodyne', '*.bin')))
        label_names = sorted(glob.glob(osp.join('sequences', seq_id, 'labels', '*.label')))
        
        args = []
        for j, file_name in enumerate(file_names):
            args.append((file_name, label_names[j], seq_id))
        
        pool.starmap(process_pcd_with_labels, args)
    
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
