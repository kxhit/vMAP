import numpy as np
import open3d as o3d
import torch
import trimesh
from metrics import accuracy, completion, completion_ratio
import os

def calc_3d_metric(mesh_rec, mesh_gt, N=200000):
    """
    3D reconstruction metric.
    # todo infer network inside 3D bbox to get per obj mesh
    """
    metrics = [[] for _ in range(6)]
    # mesh_rec = trimesh.load(rec_meshfile, process=False)
    # mesh_gt = trimesh.load(gt_meshfile, process=False)
    transform, extents = trimesh.bounds.oriented_bounds(mesh_gt)
    extents = extents / 0.9 # enlarge 0.9
    box = trimesh.creation.box(extents=extents, transform=np.linalg.inv(transform))
    # box = trimesh.creation.box(extents=extents, transform=transform)
    # print("extents ", extents)
    # print("transform ", transform)
    # print("box ", box)
    # print("mesh ", mesh_rec)
    mesh_rec = mesh_rec.slice_plane(box.facets_origin, -box.facets_normal)
    # print("mesh ", mesh_rec)
    if mesh_rec.vertices.shape[0] == 0:
        print("no mesh found")
        return
    # mesh_rec.show()
    rec_pc = trimesh.sample.sample_surface(mesh_rec, N) # todo not sample globally
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, N)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices, 0.05)
    completion_ratio_rec_1 = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices, 0.01)

    # accuracy_rec *= 100  # convert to cm
    # completion_rec *= 100  # convert to cm
    # completion_ratio_rec *= 100  # convert to %
    # print('accuracy: ', accuracy_rec)
    # print('completion: ', completion_rec)
    # print('completion ratio: ', completion_ratio_rec)
    # print("completion_ratio_rec_1cm ", completion_ratio_rec_1)
    metrics[0].append(accuracy_rec)
    metrics[1].append(completion_rec)
    metrics[2].append(completion_ratio_rec_1)
    metrics[3].append(completion_ratio_rec)
    metrics[4].append(0)
    metrics[5].append(0)
    return metrics

# # exp_name = ["room0_bmap", "room1_bmap", "room2_bmap", "office0_bmap", "office1_bmap", "office2_bmap", "office3_bmap", "office4_bmap"]
# exp = "office_4"
#
# # for imap
# gt_dir = "/home/xin/data/Replica/replica_v1/"+exp+"/habitat"
# # exp_dir = "../train/examples/Replica/"+exp[:-2]+exp[-1]+"_imap"
# exp_dir = "/home/xin/data/Results/imap/Replica_kf50_md/"+exp[:-2]+exp[-1]+"_imap"
# gt_mesh_files = os.listdir(gt_dir)
# gt_meshfile_list = []
# for f in gt_mesh_files:
#     if "mesh_semantic.ply_" in f:
#         gt_meshfile_list.append(os.path.join(gt_dir, f))
# rec_meshfile = os.path.join(exp_dir, "scene_mesh/imap_frame1999_obj0.obj")
# mesh_rec = trimesh.load(rec_meshfile, process=False)
# for gt_meshfile in gt_meshfile_list:
#     print("gt mesh ", gt_meshfile)
#     mesh_gt = trimesh.load(gt_meshfile)
#     calc_3d_metric(mesh_rec, mesh_gt, N=10000)  # for objs use 10k, for scene use 200k points

# for niceslam
from tqdm import tqdm
import json

exp_name = ["h4", "h8", "h16", "h32", "h64", "h128", "h256"]

for exp in exp_name:
    gt_dir = "/home/xin/data/Replica/replica_v1/room_0/habitat"
    exp_dir = "./logs/ablation/bmap_"+ exp
    output_path = os.path.join(exp_dir, "eval_mesh")
    os.makedirs(output_path, exist_ok=True)

    gt_mesh_files = os.listdir(gt_dir)
    gt_meshfile_list = []
    for f in gt_mesh_files:
        if "mesh_semantic.ply_" in f:
            gt_meshfile_list.append(os.path.join(gt_dir, f))

    # mesh_rec.invert()   # niceslam mesh face needs invert
    metrics_3D = [[] for _ in range(4)]

    with open(os.path.join(gt_dir, "info_semantic.json")) as f:
        label_obj_list = json.load(f)["objects"]

    bg_meshes = []
    background_cls_list = [5, 12, 30, 31, 40, 60, 92, 93, 95, 97, 98, 79]
    for obj in label_obj_list:
        if int(obj["class_id"]) in background_cls_list:
            obj_file = os.path.join(gt_dir, "mesh_semantic.ply_" + str(int(obj["id"])) + ".ply")
            obj_mesh = trimesh.load(obj_file)
            bg_meshes.append(obj_mesh)

    metrics_3D = [[] for _ in range(4)]
    # output npy metrics
    output_path = os.path.join(exp_dir, "eval_mesh")
    os.makedirs(output_path, exist_ok=True)

    # get obj ids
    # bmap_exp_dir = "/home/xin/data/bmap_submit/Replica/Replica_m100_f3_f2/"+ exp +"_bmap/"
    bmap_exp_dir = "./logs/ablation/bmap_"+ exp

    mesh_list = os.listdir(os.path.join(bmap_exp_dir, "scene_mesh"))
    obj_ids = []
    for mesh_file in mesh_list:
        obj_id = mesh_file[mesh_file.find("_obj") + len("_obj"):mesh_file.find(".obj")]
        obj_ids.append(int(obj_id))

    for obj_id in tqdm(obj_ids):
        print("obj ", obj_id)
        if obj_id == 0:
            continue    # dont eval background
            N = 200000
            mesh_gt = trimesh.util.concatenate(bg_meshes)
        else:
            N = 10000
            obj_file = os.path.join(gt_dir, "mesh_semantic.ply_" + str(obj_id) + ".ply")
            mesh_gt = trimesh.load(obj_file)
            rec_meshfile = os.path.join(exp_dir, "scene_mesh/frame_1999_obj" + str(obj_id) + ".obj")
            mesh_rec = trimesh.load(rec_meshfile, process=False)

        metrics = calc_3d_metric(mesh_rec, mesh_gt, N=N)  # for objs use 10k, for scene use 200k points
        if metrics is None:
            continue
        np.save(output_path + '/imap_obj{}.npy'.format(obj_id), np.array(metrics))

        metrics_3D[0].append(metrics[0])
        metrics_3D[1].append(metrics[1])
        metrics_3D[2].append(metrics[2])
        metrics_3D[3].append(metrics[3])
    np.save(output_path + '/metrics_3D.npy', np.array(metrics_3D))
    print("-----------------------------------------")
    print("finish niceslam exp ", exp)