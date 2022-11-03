import torch
import cv2
import numpy as np
import os

from sampling_manager import *
import utils
import open3d

# todo verify on replica
if __name__ == "__main__":
    # hyper param for trainer
    # my_device = "cuda:0"
    my_device = "cpu"
    win_size = 5
    n_samples_per_frame = 120 // 5
    min_depth = 0

    # param for dataset
    bbox_scale = 0.2

    # camera
    cam_info = cameraInfo(1200, 680, 600.0, 600.0, 599.5, 339.5, my_device)
    # init obj_dict
    obj_dict = {}

    # get one frame
    root_dir = "/home/xin/data/Replica/replica_v1/room_0/imap/00/"
    pose_file = os.path.join(root_dir, "traj_w_c.txt")
    pose_all = np.loadtxt(pose_file, delimiter=" ").reshape([-1, 4, 4]).astype(np.float32)

    ### wrap into dataset get_item ###
    idx = 0
    rgb_file = os.path.join(root_dir, "rgb", "rgb_" + str(idx) + ".png")
    depth_file = os.path.join(root_dir, "depth", "depth_" + str(idx) + ".png")
    inst_file = os.path.join(root_dir, "semantic_instance", "semantic_instance_" + str(idx) + ".png")

    depth_np = (cv2.imread(depth_file, -1)/1000.0).astype(np.float32)
    rgb_np = cv2.imread(rgb_file).astype(np.uint8)
    inst_np = cv2.imread(inst_file, cv2.IMREAD_UNCHANGED).astype(np.int32)   # uint16 -> int32
    twc_np = pose_all[idx]

    inst_ids = np.unique(inst_np)

    rgb = torch.from_numpy(rgb_np).to(my_device).permute(1,0,2) # H,W,C -> W,H,C
    inst = torch.from_numpy(inst_np).to(my_device).permute(1,0)
    depth = torch.from_numpy(depth_np).to(my_device).permute(1,0)
    twc = torch.from_numpy(twc_np).to(my_device)

    batch_masks = []
    inst_list = []
    for inst_id in inst_ids:
        inst_mask = inst_id == inst
        batch_masks.append(inst_mask)
        inst_list.append(inst_id)

    batch_masks = torch.from_numpy(np.stack(batch_masks))
    cmins, cmaxs, rmins, rmaxs = utils.get_bbox2d_batch(batch_masks)
    obj = np.zeros(inst.shape)
    bbox_dict = {}
    # filter out small obj
    for i in range(batch_masks.shape[0]):
        w = rmaxs[i] - rmins[i]
        h = cmaxs[i] - cmins[i]
        if w <= 10 or h <= 10:  # too small, set to bg   todo
            obj[batch_masks[i]] = 0
            continue
        bbox_enlarged = utils.enlarge_bbox([rmins[i], cmins[i], rmaxs[i], cmaxs[i]], scale=bbox_scale, w=obj.shape[1],
                                     h=obj.shape[0])
        inst_id = inst_list[i]
        obj[batch_masks[i]] = inst_id
        # bbox_dict.update({inst_id: torch.from_numpy(np.array(bbox_enlarged).reshape(-1))})  # batch format
        bbox_dict.update({inst_id: torch.from_numpy(np.array([bbox_enlarged[1], bbox_enlarged[3], bbox_enlarged[0], bbox_enlarged[2]]))})  # batch format

    # for bg
    # bbox_dict.update(
    #     {0: torch.from_numpy(np.array([int(0), int(0), int(obj.shape[1]), int(obj.shape[0])]).reshape(-1))})
    bbox_dict.update({0: torch.from_numpy(np.array([int(0), int(obj.shape[0]), 0, int(obj.shape[1])]))})  # batch format

    ### wrap into dataset get_item ###

    obj_ids = np.unique(obj)
    # add new frame info to objs in current view
    for obj_id in obj_ids:
        obj_mask = obj_id == inst
        # todo filter out obj_id where mask is too small, move into dataset
        # convert inst mask to state
        state = torch.zeros_like(inst, dtype=torch.uint8, device=my_device)
        state[obj_mask] = 1
        state[inst == -1] = 2
        bbox = bbox_dict[obj_id]    # todo seq
        if obj_id in obj_dict.keys():
            scene_obj = obj_dict[obj_id]
            is_kf = True    # todo change condition according to kf_every
            scene_obj.append_keyframe(rgb, depth, state, bbox, twc, is_kf=is_kf)
        else: # init scene_obj
            scene_obj = sceneObject(my_device, rgb, depth, state, bbox, twc)
            obj_dict.update({obj_id: scene_obj})

    # get training data for all objs
    # batch_rets = []

    for _ in range(20):
        with performance_measure(f"Looping over {len(obj_dict.keys())} objects,"):
            for obj_id in obj_dict.keys():
                print("obj_id ", obj_id)
                gt_depth, gt_rgb, valid_depth_mask, obj_mask, input_pcs, sampled_z\
                    = obj_dict[obj_id].get_training_samples(win_size, n_samples_per_frame, cam_info.rays_dir_cache)
                print("input pcs ", input_pcs.shape)
                # vis3d
                pc = open3d.geometry.PointCloud()
                pc.points = open3d.utility.Vector3dVector(input_pcs.cpu().numpy().reshape(-1,3))
                open3d.visualization.draw_geometries([pc])
                # rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
                #     open3d.geometry.Image(im),
                #     open3d.geometry.Image(depth),
                #     depth_trunc=self.max_depth,
                #     depth_scale=1,
                #     convert_rgb_to_intensity=False,
                # )

        # batch_rets.append(ret)

    # batch_rets = torch.stack(batch_rets)