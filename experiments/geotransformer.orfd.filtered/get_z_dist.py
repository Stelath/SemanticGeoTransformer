# Written by Ethan Anderson 8/30/2024

# Operation
# 1. Create the train data loader
# 2. Iterate through each point cloud
# 3. Collect the z value of each point
# 4. Create a histogram of the z values
# 5. Plot the histogram

import argparse
import time
import torch
import torch.optim as optim

from geotransformer.engine import EpochBasedTrainer

from config import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from loss import OverallLoss, Evaluator

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import os 

def filter_z_value(cfg):
    """
    Points less than z_max are filtered out
    Outputs "non_filtered.npy" and "filtered.npy"
    """
    print("Building data loader...")
    loader, _, _ = train_valid_data_loader(cfg, False)
    print("Built data loader...")

    for i, data_dict in enumerate(loader):
        points = data_dict['points'][0]
        # Save the original point cloud
        np.save("non_filtered.npy", points)
        print("Original point cloud saved...")

        # Filter using standard deviation
        points_filtered = remove_point_by_stdd(points, 1)

        np.save("filtered.npy", points_filtered)
        print("Filtered point cloud saved...")

        print("Original point cloud size:", points.shape)
        print("Filtered point cloud size:", points_filtered.shape)
        print("Proportion of points kept:", points_filtered.shape[0] / points.shape[0])
        print()

        if i > 100:
            break


def get_all_z_values(cfg):
    train_loader, _, _ = train_valid_data_loader(cfg, False)
    print("Built data loader...")
    z_bins = {}
    for i, data_dict in enumerate(tqdm(train_loader)):
        z_values = data_dict['points'][0][:, 2].tolist()
        for z in z_values:
            x = round(z, 2)
            if x in z_bins:
                z_bins[x] += 1
            else:
                z_bins[x] = 1
        
        if i > 100:
            break

    # Sort the dictionary by key
    z_bins = dict(sorted(z_bins.items()))
    return z_bins


def visualize_point_cloud(file_path):
    """
    Given a path to a .npy file, visualize the point cloud
    """
    name = file_path.split(".")[0]

    points = np.load(file_path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    norm = plt.Normalize(z.min(), z.max())  # For color scale
    ax.scatter(x, y, z, c=z, cmap='viridis', marker='.', norm=norm)
    # Make sure that the x y z scales are the same
    ax.set_box_aspect([np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2])])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


    plt.savefig(f'{name}_v1.png')
    ax.view_init(45, 45)
    plt.savefig(f'{name}_v2.png')
    # Front facing perspective
    ax.view_init(3, 0)
    plt.savefig(f'{name}_v3.png')
    # Side facing perspective
    ax.view_init(3, 90)
    plt.savefig(f'{name}_v4.png')

def remove_ground_plane(points, std_dev_threshold):
    """
    Remove points within the standard deviation threshold of the mean
    """
    # Extract z-values
    z_values = points[:, 2].numpy()
    
    # Calculate mean and standard deviation of the z-values
    z_mean = np.mean(z_values)
    z_std = np.std(z_values)
    
    # Filter out points within 1 standard deviation of the mean (i.e., the ground plane)
    # Also remove points that are below the ground plane
    mask = np.abs(z_values - z_mean) > std_dev_threshold * z_std
    filtered_points = points[mask]
    
    return filtered_points


def main():
    cfg = make_cfg()

    # between .2 and .5
    filter_z_value(cfg)
    visualize_point_cloud("filtered.npy")
    visualize_point_cloud("non_filtered.npy")


def plot_z_dist():

    if os.path.exists("z_bins.npy"):
        z_bins = np.load("z_bins.npy", allow_pickle=True).item()
    else:
        z_bins = get_all_z_values(cfg)

    # Save the bins
    np.save("z_bins.npy", z_bins)
   
    # Sort the z bins by key
    z_bins = dict(sorted(z_bins.items()))

    # Plot a cumulative proportion histogram i.e. the far right should reach 100$
    total = sum(z_bins.values())  # Total number of points
    cumulative = 0
    chosen_prop = 0
    proportions = []
    for key, value in z_bins.items():
        cumulative += value
        prop = cumulative / total
        proportions.append(prop)

        if prop < .68: # 1 standard deviation
            chosen_z = key
            chosen_prop = prop

        



    plt.plot(list(z_bins.keys()), proportions)

    plt.xticks(np.arange(-5, 10, 1))
    
    # limit x axis from -5 to 10
    plt.xlim(-5, 10)

    plt.axvline(x=chosen_z, color='r')
    plt.text(chosen_z + 1, chosen_prop, f"{chosen_z, chosen_prop}")


    plt.xlabel("Z Value")
    plt.ylabel("Cumulative Proportion (%)")
    plt.title("Cumulative Proportion of Z Values in Point Cloud")
    plt.savefig("z_dist.png")

if __name__ == '__main__':
    main()