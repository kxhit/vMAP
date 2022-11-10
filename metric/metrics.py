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


import logging
import torch
logger = logging.getLogger('debug')
# check if two state dicts equal
def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    if len(model_state_dict_1) != len(model_state_dict_2):
        logger.info(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
        model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            logger.info(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            logger.info(f"Tensor mismatch: {v_1} vs {v_2}")
            return False
    return True
# rec_pc = trimesh.sample.sample_surface(mesh_rec, 100000)
# rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])
#
# gt_pc = trimesh.sample.sample_surface(mesh_gt, 100000)
# gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
#
# accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
