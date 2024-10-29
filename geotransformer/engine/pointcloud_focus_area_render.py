"""
Renders a point cloud with a red rectangle and then zooms into the contents of the red rectangle, displayed in the bottom right of the full point cloud.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def apply_transform(points: np.ndarray, transform: np.ndarray, normals=None):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    else:
        return points

def pointcloud_focus_area_render(name, ref_path, src_path, transform_path, output_path, rect, ignore_transform=False):
    """
    Renders a point cloud with a red rectangle and then zooms into the contents of the red rectangle, displayed in the bottom right of the full point cloud.
    :param ref_path: Path to the reference point cloud. (.npy)
    :param src_path: Path to the source point cloud. (.npy)
    :param transform_path: Path to the transformation file. (.npy)
    :param output_path: Path to save the rendered image. (.png)
    :param rect: Coordinates of the rectangle. (x, y, width, height)
    """

    ref_points = np.load(ref_path)
    src_points = np.load(src_path)
    transform_pr = np.load(transform_path)

    print("total translation mag: ", np.linalg.norm(transform_pr[:3, 3]))

    # Apply the transformation to the src points
    if not ignore_transform:
        src_points_trans = apply_transform(src_points, transform_pr)
    else:
        print("Ignoring transform")
        src_points_trans = src_points

    fig, ax1 = plt.subplots(figsize=(4, 10))

    # Plot 1 - Full point cloud in top-down view (X, Y)
    ax1.scatter(ref_points[:, 0], ref_points[:, 1], c='cyan', s=0.1, label="Ref Points")
    ax1.scatter(src_points_trans[:, 0], src_points_trans[:, 1], c='orange', s=0.1, label="Transformed Src Points")
    # ax1.set_title(name, fontsize=20, pad=20, loc='center')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('off')

    # Draw the red rectangle (in X, Y plane)
    x, y, w, h = rect
    ax1.plot([x, x + w], [y, y], 'r')
    ax1.plot([x, x], [y, y + h], 'r')
    ax1.plot([x + w, x + w], [y, y + h], 'r')
    ax1.plot([x, x + w], [y + h, y + h], 'r')

    window_width = 20 + w
    ax1.set_xlim(x - 10, x + window_width)

    # Work with the 4 x 10
    window_height = window_width * 10 / 4
    y_center = 0
    ax1.set_ylim(y_center - window_height / 2, y_center + window_height / 2)

    # Plot 2 - Inset zoomed-in area in the bottom right corner
    ax_inset = inset_axes(ax1, width="80%", height="40%", loc='lower right')  # Inset at bottom-right corner
    ax_inset.scatter(ref_points[:, 0], ref_points[:, 1], c='cyan', s=30)
    ax_inset.scatter(src_points_trans[:, 0], src_points_trans[:, 1], c='orange', s=30)
    ax_inset.set_xlim(rect[0], rect[0] + rect[2])
    ax_inset.set_ylim(rect[1], rect[1] + rect[3])

    # Hide labels for the inset plot
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    print("Saved to: ", os.path.abspath(output_path))
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":





    rect = [-30, 10, 5, 5]
    sequence = "figure_8_to_turnpike_pt2_2023-09-12-12-28-48"
   #  pair = "2820_2880"
    base_path = f"output/geotransformer.tartan2/features/{sequence}"
    base_path_reducedaug = f"output/geotransformer.tartan2.reducedaug/features/{sequence}"
    base_path_kitti = f"output/geotransformer.tartan2.kitti/features/{sequence}"

    # Find the pair with the largest gt translation
    max_translation = 0
    max_pair = None
    for pair in os.listdir(base_path):
        print("pair: ", pair)
        if "transform_gt.npy" in pair:
            transform = np.load(f"{base_path}/{pair}")
            translation = np.linalg.norm(transform[:3, 3])
            if translation > max_translation:
                max_translation = translation
                print("pair: ", pair)
                max_pair = pair.split("_")[0] + "_" + pair.split("_")[1]
                print("max_pair: ", max_pair)
    print("Max translation: ", max_translation, " for pair: ", max_pair)
    

    pair = max_pair

    pointcloud_focus_area_render("Input",
                                f"{base_path}/{pair}_ref.npy",
                                f"{base_path}/{pair}_src.npy",
                                f"{base_path}/{pair}_transform_pr.npy",
                                "focus_area_render_input_tartan.png",
                                rect,
                                ignore_transform=True)
    
    pointcloud_focus_area_render("Ground Truth",
                                f"{base_path}/{pair}_ref.npy",
                                f"{base_path}/{pair}_src.npy",
                                f"{base_path}/{pair}_transform_gt.npy",
                                "focus_area_render_gt_tartan.png",
                                rect)
    
    pointcloud_focus_area_render("GeoTransformer KITTI",
                                f"{base_path_kitti}/{pair}_ref.npy",
                                f"{base_path_kitti}/{pair}_src.npy",
                                f"{base_path_kitti}/{pair}_transform_pr.npy",
                                "focus_area_render_geotransformer_KITTI_tartan.png",
                                rect)

    pointcloud_focus_area_render("GeoTransformer",
                                f"{base_path}/{pair}_ref.npy",
                                f"{base_path}/{pair}_src.npy",
                                f"{base_path}/{pair}_transform_pr.npy",
                                "focus_area_render_geotransformer_tartan.png",
                                rect)

    
    pointcloud_focus_area_render("Ours",
                                f"{base_path_reducedaug}/{pair}_ref.npy",
                                f"{base_path_reducedaug}/{pair}_src.npy",
                                f"{base_path_reducedaug}/{pair}_transform_pr.npy",
                                "focus_area_render_z_aug_geotransformer_tartan.png",
                                rect)
    # # Z-Priority
    # ref_path = "output/geotransformer.rellis.reducedaug/features/4/0_32_ref.npy"
    # src_path = "output/geotransformer.rellis.reducedaug/features/4/0_32_src.npy"
    # transform_path = "output/geotransformer.rellis.reducedaug/features/4/0_32_transform_pr.npy"
    # output_path = "focus_area_render_z-aug.png"
    # rect = [-8, 4, 5, 5]
    # # rect = [25, 0, 5, 5]
    # pointcloud_focus_area_render("GeoTransformer Z-Priority", ref_path, src_path, transform_path, output_path, rect)

    # # Base GeoTransformer
    # ref_path = "output/geotransformer.rellis/features/4/0_32_ref.npy"
    # src_path = "output/geotransformer.rellis/features/4/0_32_src.npy"
    # transform_path = "output/geotransformer.rellis/features/4/0_32_transform_pr.npy"
    # output_path = "focus_area_render_base.png"

    # pointcloud_focus_area_render("GeoTransformer", ref_path, src_path, transform_path, output_path, rect)
