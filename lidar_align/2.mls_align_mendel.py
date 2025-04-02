import laspy
import open3d as o3d
import numpy as np
import copy
import os
import re

# Compute transformation using Procrustes analysis (rigid alignment)
def compute_transformation(source, target):
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    # Compute optimal rotation using SVD
    U, _, Vt = np.linalg.svd(target_centered.T @ source_centered)
    R = U @ Vt  # Rotation matrix

    # Ensure a proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Compute translation
    t = target_mean - R @ source_mean

    # Create transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t

    return transformation

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
def icp(photo_cloud, lidar_cloud, threshold, identity_matrix,radius, max_nn, k):

    lidar_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

    initial_eval_result = o3d.pipelines.registration.evaluate_registration(
        photo_cloud, lidar_cloud, threshold, identity_matrix)
    
    initial= str(initial_eval_result)
    initial_fitness_match = re.search(r'fitness=(.*?),', initial)
    initial_fitness_match = float(initial_fitness_match.group(1))
    final_fitness_match=initial_fitness_match+0.001
    while initial_fitness_match<1 and final_fitness_match>initial_fitness_match:
        #initial evaluation
        initial_eval_result = o3d.pipelines.registration.evaluate_registration(
            photo_cloud, lidar_cloud, threshold, identity_matrix)
        initial= str(initial_eval_result)
        initial_fitness_match = re.search(r'fitness=(.*?),', initial)
        initial_fitness_match = float(initial_fitness_match.group(1))
        #icp iteration
        loss = o3d.pipelines.registration.TukeyLoss(k=k)
        p2p_loss = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        reg_result = o3d.pipelines.registration.registration_icp(
            photo_cloud, lidar_cloud, threshold, identity_matrix, p2p_loss)
        reg_result_str = str(reg_result)
        final_fitness_match = re.search(r'fitness=(.*?),', reg_result_str)
        final_fitness_match = float(final_fitness_match.group(1))
        if final_fitness_match<1 and final_fitness_match>initial_fitness_match:
            print('fitness improved from', initial_fitness_match, 'to', final_fitness_match)
            photo_cloud=photo_cloud.transform(reg_result.transformation)
            #save the reg result
            higuest_reg_result=reg_result
            higuest_reg_result_matrix=reg_result.transformation
            print('photo_cloud has been transformed')
        else:
            print('fitness did not improve, while loop will stop')
            print('final fitness is', final_fitness_match)
            print('the last photo cloud was not transformed,it is the correct one')
    return photo_cloud, lidar_cloud, higuest_reg_result, higuest_reg_result_matrix
def lazO3d(file_path):
    cloud = laspy.read(file_path)
    points = np.vstack((cloud.x, cloud.y, cloud.z)).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

path=r"/home/vasquezv/BCI_50ha"


#to ensure aligment of sequential tiles I am going to need a coffee, I have no idea how complicated this is going to get
#reference tile placed on 0,0

def evaluate_registration(reference_overlap, target_overlap, threshold, init_matrix):
    eval_result = o3d.pipelines.registration.evaluate_registration(
        reference_overlap, target_overlap, threshold, init_matrix)
    return eval_result.fitness, eval_result

# Function to transform and save the point cloud
def save_point_cloud(cloud, output_path):
    points = np.asarray(cloud.points)
    header = laspy.LasHeader(point_format=3, version="1.4")
    las = laspy.LasData(header)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    las.write(output_path)

def calculate_fitness(reference, target, threshold, init_matrix):
    bbox_reference = reference.get_axis_aligned_bounding_box()
    bbox_target = target.get_axis_aligned_bounding_box()

    min_bound = np.maximum(bbox_reference.min_bound, bbox_target.min_bound)
    max_bound = np.minimum(bbox_reference.max_bound, bbox_target.max_bound)
    crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    reference_overlap = reference.crop(crop_box)
    target_overlap = target.crop(crop_box)

    fitness, _ = evaluate_registration(reference_overlap, target_overlap, threshold, init_matrix)
    return fitness, reference_overlap, target_overlap


plots = [
    '3020', '3021', '3022', '3023', '3024',
    '3120', '3121', '3122', '3123', '3124',
    '3220', '3221', '3222', '3223', '3224',
    '3320', '3321', '3322', '3323', '3324',
    '3420', '3421', '3422', '3423', '3424'
]

threshold = 0.1
init_matrix = np.identity(4)
score_threshold= 0.99

# Dictionary to store fitness scores
fitness_dict = {}

# Evaluate overlaps for adjacent plots
for plot in plots:
    print("the plot is", plot)
    reference_path = os.path.join(path, "plot1", f"{plot}_cloud.laz")
    reference = lazO3d(reference_path)
    # Get the plot indices
    plot_row = int(plot[0:2])
    plot_col = int(plot[2:4])
    adjacent_plots = []
    if plot_col + 1 <= 24:
        adjacent_plots.append(f"{plot_row}{plot_col + 1:02d}")
    if plot_row + 1 <= 34:
        adjacent_plots.append(f"{plot_row + 1:02d}{plot_col:02d}")

    for adj_plot in adjacent_plots:
        target_path = os.path.join(path, "plot1", f"{adj_plot}_cloud.laz")
        target = lazO3d(target_path)

        # Calculate fitness score
        fitness, reference_overlap, target_overlap = calculate_fitness(reference, target, threshold, init_matrix)
        fitness_dict[(plot, adj_plot)] = fitness
        print(f"Fitness score for {plot} -> {adj_plot}: {fitness}")
        if fitness < score_threshold:
            _,_,_,result_matrix = icp(reference_overlap, target_overlap, threshold, init_matrix, 4, 100, 3.7)
            if result_matrix is not None:
                target_transformed = target.transform(result_matrix)
                save_point_cloud(target_transformed, target_path)
                print(f"Aligned and saved: {adj_plot}")
            else:
                print("The ICP failed, no improvement")
        else:
            print("Fitness is enough")



