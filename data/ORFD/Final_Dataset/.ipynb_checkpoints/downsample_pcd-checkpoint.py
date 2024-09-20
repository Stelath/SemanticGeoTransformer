import os
import numpy as np
import glob
from tqdm import tqdm
import open3d as o3d

def main():
    # Set the root directory path for the dataset
    dataset_root = '/project/bli4/autoai/joyang/GeoTransformer/data/ORFD/Final_Dataset/validation/lidar_data'  
    output_root = '/project/bli4/autoai/joyang/GeoTransformer/data/ORFD/Final_Dataset/validation/downsampled'  

    # Create the output folder if it does not exist
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Created directory: {output_root}")
    
    # Search for .bin files
    file_names = glob.glob(os.path.join(dataset_root, '*.bin'))
    
    # Print the number of files
    print(f'Looking for .bin files in: {dataset_root}')
    print(f'Number of files found: {len(file_names)}')
    
    if not file_names:
        print('No .bin files found, exiting...')
        return

    # Iterate over each .bin file
    for file_name in tqdm(file_names):
        frame = os.path.splitext(os.path.basename(file_name))[0]
        new_file_name = os.path.join(output_root, frame + '.npy')
        
        # Read the point cloud data from the .bin file, where each point has 5 values (x, y, z, intensity, category/other feature)
        points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 5)
        
        # Create an Open3D point cloud object, using only the first 3 values (x, y, z)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Perform voxel downsampling
        pcd = pcd.voxel_down_sample(0.3)
        points_downsampled = np.array(pcd.points).astype(np.float32)
        
        # Save the downsampled point cloud data
        np.save(new_file_name, points_downsampled)

if __name__ == '__main__':
    main()
