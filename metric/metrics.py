import numpy as np
from scipy.spatial import cKDTree as KDTree

def completion_ratio(gt_points, rec_points, dist_th=0.01):
    gen_points_kd_tree = KDTree(rec_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    completion = np.mean((one_distances < dist_th).astype(np.float))
    return completion


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(rec_points)
    gen_to_gt_chamfer = np.mean(two_distances)
    return gen_to_gt_chamfer


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    one_distances, two_vertex_ids = gt_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(one_distances)
    return gt_to_gen_chamfer


def chamfer(gt_points, rec_points):
    # one direction
    gen_points_kd_tree = KDTree(rec_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(one_distances)

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(rec_points)
    gen_to_gt_chamfer = np.mean(two_distances)

    return (gt_to_gen_chamfer + gen_to_gt_chamfer) / 2.

