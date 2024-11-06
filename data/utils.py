"""
File created by Ethan on 11/6 for some gt.py utils to help standardize our metadata
"""

import numpy as np

def find_next_restricted_distance_point_cloud(p1_pose, p1_timestamp, timestamp_poses, min_distance, max_distance):
    """
    Given the p1_pose and timestamp and dictionary of timestamp: pose pairs, find the next point cloud that is within the desired distance range.

    Return:
    (p2_name, p2_pose, timestamp) if the point cloud is within the range

    p2_name is equivalent to the index of that point cloud's timestamp in the timestamps list

    Pose format:

        Size of the array:
        4x4 matrix

        Labled example with non-zero translation matrix: x+3 y+7 z+2
        [[ 0.99999999 -0.00000001  0.00000001  3.00000001]
        [ 0.00000001  0.99999999 -0.00000001  7.00000001]
        [-0.00000001  0.00000001  0.99999999  2.00000001]
        [ 0.          0.          0.          1.        ]]



        The first three columns are the rotation matrix
        The last column is the translation matrix
        In this example the translation matrix is [3, 7, 2]

    """

    timestamps_list_str = list(timestamp_poses.keys())

    # Convert timestamps_list to numpy array and convert string to float
    timestamps_list = np.array([float(timestamp) for timestamp in timestamps_list_str])


    # Assert that the list is in ascending order
    # assert all(timestamps_list[i] < timestamps_list[i + 1] for i in range(len(timestamps_list) - 1)), "The timestamps are not in ascending order"

    # Find the index of the p1_timestamp
    p1_index = timestamps_list_str.index(p1_timestamp)
    assert p1_index != -1, "Could not find the p1_timestamp in the timestamp_poses dictionary"

    # Getting a list of timestamps to check
    timestamps_to_check = timestamps_list_str[p1_index + 1:]

    for i, timestamp in enumerate(timestamps_to_check):
        # Ensure the timestamp is larger 
        p2_pose = timestamp_poses[timestamp]
        p2_name = i + p1_index + 1

        # assert p2_name == timestamps_list_str.index(timestamp), "The index of the timestamp does not match the index in the timestamps list"


        assert timestamp > p1_timestamp 

        dis_squared = np.sum((p2_pose[:3, 3] - p1_pose[:3, 3]) ** 2)
        if min_distance**2 < dis_squared < max_distance**2:
            # Within the range, return the index and distance
            print(dis_squared ** 0.5)
            return p2_name, p2_pose, timestamp

        # If the distance is greater than the max distance, return None i.e. give up
        if dis_squared > max_distance**2:
            return None, None, None

    return None, None, None