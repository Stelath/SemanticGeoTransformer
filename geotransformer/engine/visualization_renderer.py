# Written by Ethan 9/9/2024
import matplotlib.pyplot as plt
import os 
import numpy as np 
from tqdm import tqdm
import copy 
import imageio

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

def apply_transform(points: np.ndarray, transform: np.ndarray, normals= None):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    else:
        return points
    
def process_output(output_path, num_groups, jump_groups=9):
    groups = []
    files = os.listdir(output_path)
    files = sorted(files)
    # Sort the files by the two frame numbers
    files = sorted(files, key=lambda x: int(x.split('_')[1]))
    files = sorted(files, key=lambda x: int(x.split('_')[0]))


    for i in range(0, len(files), 4):
        ref_file = files[i]
        src_file = files[i+1]
        transform_file_gt = files[i+2]
        transform_file_pr = files[i+3]
        groups.append((ref_file, src_file, transform_file_gt, transform_file_pr))
    groups = groups[::jump_groups]
    groups = groups[:num_groups]

    return groups


def render_point_cloud_registration_strip(output_path):
    groups = process_output(output_path, 7)

    # Set up figure for vertical strip (adjust height based on number of groups)
    num_groups = len(groups)
    fig = plt.figure(figsize=(5, num_groups * 3))  # Width 5, height proportional to number of groups

    for idx, group in enumerate(tqdm(groups)):
        ref_points = np.load(os.path.join(output_path, group[0]))
        src_points = np.load(os.path.join(output_path, group[1]))
        transform_gt = np.load(os.path.join(output_path, group[2]))
        transform_pr = np.load(os.path.join(output_path, group[3]))

        print("Transform differnce: ", np.linalg.norm(transform_gt - transform_pr))
                               


        # Apply the transformation to the src points
        src_points_est = apply_transform(src_points, transform_pr)
        src_points_gt = apply_transform(src_points, transform_gt)
        src_points_in = src_points  # Original src points

        # Create a subplot for each point cloud registration (stacked vertically)
        ax = fig.add_subplot(num_groups, 1, idx + 1, projection='3d')

        # Remove background, grid, and axes
        ax.set_facecolor((0, 0, 0, 0))  # Transparent background
        ax.grid(False)
        ax.set_axis_off()

        # Plot the ref points in blue and transformed src points in red
        ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], c='b', marker=',', s=0.05, alpha=0.5)
        ax.scatter(src_points_est[:, 0], src_points[:, 1], src_points[:, 2], c='r', marker=',', s=0.05, alpha=0.5)
        ax.scatter(src_points_gt[:, 0], src_points[:, 1], src_points[:, 2], c='g', marker=',', s=0.05, alpha=0.5)
        ax.scatter(src_points_in[:, 0], src_points[:, 1], src_points[:, 2], c='y', marker=',', s=0.05, alpha=0.5)


        # Make the point cloud a bit bigger
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)

        ax.view_init(elev=50, azim=-90)

    # Save the figure with a transparent background
    plt.subplots_adjust(hspace=.1)  # Remove space between subplots

    strip_path = "registration_strip.png"
    plt.savefig(strip_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)

    print("saved to: ", os.path.abspath(strip_path))


def create_point_cloud_registration_sequence_gif(output_path):
    gt_pr_offset = 100
    group_count = 50
    groups = process_output(output_path, group_count, jump_groups=1)

    # Group comp: ref_file, src_file, transform_file_gt, transform_file_pr

    # Create a GIF of each point cloud being registered to the sequence, i.e. 10 gif frames

    # Getting the absolute position of each point cloud, the ref of the first group is the origin
    ref_points = np.load(os.path.join(output_path, groups[0][0]))

    # draw this point cloud on it's own as the first frame
    fig = plt.figure(figsize=(150, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(True)
    # set_axes_equal(ax)
    ax.set_box_aspect([3, 1, 1])  # Equal aspect ratio for x, y, z
    ax.set_facecolor((0, 0, 0, 0))  # Transparent background
    ax.grid(False)
    ax.set_axis_off()
    ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], c='b', marker=',', s=0.05, alpha=0.5)

    # plot another but with y + gt_pr_offset for the gt side-by-side display
    ax.scatter(ref_points[:, 0], ref_points[:, 1] + gt_pr_offset, ref_points[:, 2], c='b', marker=',', s=0.05, alpha=0.5)

    ax.set_xlim(-200, 20)
    ax.set_ylim(-20, 100)
    # Adjust view so we are looking towards +x
    view_angles = [(50, -90),
                   (20, 0)]
    ax.view_init(elev=50, azim=-120)

    # Gif building dir
    gif_dir = "gif_temp"
    os.makedirs(gif_dir, exist_ok=True)
    # Clear the directory
    for f in os.listdir(gif_dir):
        os.remove(os.path.join(gif_dir, f))

    plt.subplots_adjust(hspace=.1)  # Remove space between subplots

    # plt.savefig(os.path.join(gif_dir, f"frame_0.png"), transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)

    colors = ['orange', 'g', 'y', 'c', 'm', 'k', 'b', 'purple', 'brown']
    # colors = ['g']

    # Loop through the rest of the groups and create a frame for each
    cum_transform_pr = np.eye(4)
    cum_transform_gt = np.eye(4)
    frame = 0
    for i, group in enumerate(groups[1:]):
        
        src_points = np.load(os.path.join(output_path, group[1]))
        # print("Src_file: ", group[1])
        new_transform_pr = np.load(os.path.join(output_path, group[3]))
        new_transform_gt = np.load(os.path.join(output_path, group[2]))

        print("Ground truth x change: ", new_transform_gt[0, 3], f" from {group[0]} to {group[1]}")


        # Add in the cumulative transformation
        transform_pr = np.matmul(cum_transform_pr, new_transform_pr)
        transform_gt = np.matmul(cum_transform_gt, new_transform_gt)

        # print("pos (pr): ", transform_pr[:3, 3])
        # print("pos (gt): ", transform_gt[:3, 3])

        src_points_est = apply_transform(src_points, transform_pr)
        src_points_gt = apply_transform(src_points, transform_gt)

        # Update the cumulative transformation
        cum_transform_pr = copy.deepcopy(transform_pr)
        cum_transform_gt = copy.deepcopy(transform_gt)

        # Plot onto the same figure (only render 10)
        if (i % (group_count // 20)) == 0:
            # pr
            ax.scatter(src_points_est[:, 0], src_points_est[:, 1], src_points_est[:, 2], c=colors[frame%len(colors)], marker=',', s=0.05, alpha=0.5)
            # gt
            ax.scatter(src_points_gt[:, 0], src_points_gt[:, 1] + gt_pr_offset, src_points_gt[:, 2], c=colors[frame%len(colors)], marker=',', s=0.05, alpha=0.5)

            translation_error = np.linalg.norm(transform_gt[:3, 3] - transform_pr[:3, 3])
            rotational_error_deg = np.arccos((np.trace(transform_gt[:3, :3].T @ transform_pr[:3, :3]) - 1) / 2) * 180 / np.pi
            total_translation = np.linalg.norm(transform_gt[:3, 3])
            total_rotation_deg = np.arccos((np.trace(transform_gt[:3, :3].T @ np.eye(3)) - 1) / 2) * 180 / np.pi

           



            # Change xlim and ylim to follow the point cloud as it moves
            pos = transform_gt[:3, 3]

            ax.set_xlim(pos[0] - 15, pos[0] + 35)
            ax.set_ylim(pos[1] - 25 + gt_pr_offset / 2, pos[1] + 25 + gt_pr_offset / 2)
            ax.set_zlim(-5, pos[2] + 5)

          
        
        

            print("camera-based pos: ", pos)
            for angle in view_angles:
                 # Remove old text
                for txt in ax.texts:
                    txt.set_visible(False)

                # ref number
                point_cloud_number = group[0].split('_')[0]
                
                # Annotate with error
                # Problem, the text keeps writing over itself and doesn't clear
                ax.text2D(0.05, 0.92, f"Translation error (m): {translation_error:.2f} / {total_translation:.2f}", transform=ax.transAxes, color='black', zorder=100, fontsize=7)
                ax.text2D(0.05, 0.89, f"Rotational error (deg): {rotational_error_deg:.2f} / {total_rotation_deg:.2f}", transform=ax.transAxes, color='black', zorder=100, fontsize=7)
                
                ax.text2D(0.05, 0.95, f"Frame: {frame} | Point Cloud # {point_cloud_number}", transform=ax.transAxes, color='black', zorder=100, fontsize=7)
                if angle[0] == 20:
                    ax.text2D(0.05, .10, f"Predicted", transform=ax.transAxes, color='black', zorder=100)
                    ax.text2D(0.65, .10, f"Ground Truth", transform=ax.transAxes, color='black', zorder=100)

                if angle[0] == 50:
                    ax.text2D(0.05, .50, f"Ground Truth", transform=ax.transAxes, color='black', zorder=100)
                    ax.text2D(0.65, .10, f"Predicted", transform=ax.transAxes, color='black', zorder=100)


                ax.view_init(elev=angle[0], azim=angle[1])
                abs_path = os.path.abspath(os.path.join(gif_dir, f"frame_{frame+1}_{angle[0]}_{angle[1]}.png"))
                plt.savefig(abs_path, transparent=False, bbox_inches='tight', pad_inches=0, dpi=300)
                print(f"Saving to: {abs_path}")
            frame += 1

    # Create the gif for each view angle
    for angle in view_angles:
        images = []
        for i in range(1, frame + 1):
            abs_path = os.path.abspath(os.path.join(gif_dir, f"frame_{i}_{angle[0]}_{angle[1]}.png"))
            print("reading: ", abs_path)
            images.append(imageio.imread(abs_path))
        out = os.path.abspath(os.path.join(gif_dir, f"registration_sequence_{angle[0]}_{angle[1]}.gif"))

        # Show each frame for .5 seconds, repeat forever
        imageio.mimsave(out, images, loop=0, fps=1.5)
        print(f"Saved gif to registration_sequence_{angle[0]}_{angle[1]}.gif")
    



if __name__ == "__main__":
    path = "output/geotransformer.rellis.reducedaug/features/4"
    create_point_cloud_registration_sequence_gif(path)
