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

def main():
    num_cores = 16 #multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    
    for i in range(5):
        seq_id = '{:02d}'.format(i)
        file_names = glob.glob(osp.join('sequences', seq_id, 'os1_cloud_node_kitti_bin', '*.bin'))
        
        pool.starmap(process_file, [(file_name, seq_id) for file_name in file_names])
    
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
