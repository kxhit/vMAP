import torch
import cv2
import numpy as np
import os

import loss
from sampling_manager import *
import utils
import open3d

# todo verify on replica
if __name__ == "__main__":
    ###################################
    # init
    # hyper param for trainer
    training_device = "cuda:0"
    data_device = "cpu"
    # vis_device = "cuda:1"
    win_size = 5
    n_samples_per_frame = 120 // 5
    min_depth = 0
    max_depth = 10.
    depth_scale = 1000.

    # param for dataset
    bbox_scale = 0.2

    # camera
    W = 1200
    H = 680
    fx = 600.0
    fy = 600.0
    cx = 599.5
    cy = 339.5

    cam_info = cameraInfo(W, H, fx, fy, cx, cy, data_device)
    intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
        width=W,
        height=H,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy)

    # init obj_dict
    obj_dict = {}

    # init for training
    learning_rate = 0.001
    weight_decay = 0.013
    optimiser = torch.optim.AdamW([torch.autograd.Variable(torch.tensor(0))], lr=learning_rate, weight_decay=weight_decay)

    # # vmap
    # from functorch import vmap, combine_state_for_ensemble


    #############################################
    # todo dataloader, dataset
    # get one frame
    root_dir = "/home/xin/data/Replica/replica_v1/room_0/imap/00/"
    pose_file = os.path.join(root_dir, "traj_w_c.txt")
    pose_all = np.loadtxt(pose_file, delimiter=" ").reshape([-1, 4, 4]).astype(np.float32)
    dataset_len = pose_all.shape[0]
    for idx in tqdm(range(dataset_len)):
        ### wrap into dataset get_item ###
        # idx = 0
        rgb_file = os.path.join(root_dir, "rgb", "rgb_" + str(idx) + ".png")
        depth_file = os.path.join(root_dir, "depth", "depth_" + str(idx) + ".png")
        inst_file = os.path.join(root_dir, "semantic_instance", "semantic_instance_" + str(idx) + ".png")

        depth_np = (cv2.imread(depth_file, -1)/depth_scale).astype(np.float32)
        rgb_np = cv2.imread(rgb_file).astype(np.uint8)
        inst_np = cv2.imread(inst_file, cv2.IMREAD_UNCHANGED).astype(np.int32)   # uint16 -> int32
        twc_np = pose_all[idx]

        inst_ids = np.unique(inst_np)

        rgb = torch.from_numpy(rgb_np).to(data_device).permute(1,0,2) # H,W,C -> W,H,C, # todo RGB or BGR?
        inst = torch.from_numpy(inst_np).to(data_device).permute(1,0)
        depth = torch.from_numpy(depth_np).to(data_device).permute(1,0)
        twc = torch.from_numpy(twc_np).to(data_device)

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
        with performance_measure(f"Appending data {len(obj_dict.keys())} objects,"):
            # append new frame info to objs in current view
            for obj_id in obj_ids:
                obj_mask = obj_id == inst
                # todo filter out obj_id where mask is too small, move into dataset
                # convert inst mask to state
                state = torch.zeros_like(inst, dtype=torch.uint8, device=data_device)
                state[obj_mask] = 1
                state[inst == -1] = 2
                bbox = bbox_dict[obj_id]    # todo seq
                if obj_id in obj_dict.keys():
                    scene_obj = obj_dict[obj_id]
                    is_kf = True    # todo change condition according to kf_every
                    scene_obj.append_keyframe(rgb, depth, state, bbox, twc, is_kf=is_kf)
                else: # init scene_obj
                    scene_obj = sceneObject(data_device, rgb, depth, state, bbox, twc)
                    obj_dict.update({obj_id: scene_obj})
                    # params = [scene_obj.trainer.fc_occ_map.parameters(), scene_obj.trainer.pe.parameters()]
                    optimiser.add_param_group({"params": scene_obj.trainer.fc_occ_map.parameters(), "lr": learning_rate, "weight_decay": weight_decay})
                    optimiser.add_param_group({"params": scene_obj.trainer.pe.parameters(), "lr": learning_rate, "weight_decay": weight_decay})


        ##################################################################
        # training data preperation, get training data for all objs
        # batch_rets = []
        batch_gt_depth = []
        batch_gt_rgb = []
        batch_depth_mask = []
        batch_obj_mask = []
        batch_input_pcs = []
        batch_sampled_z = []

        with performance_measure(f"Looping over {len(obj_dict.keys())} objects,"):
            for obj_id in obj_dict.keys():
                print("obj_id ", obj_id)
                gt_rgb, gt_depth, valid_depth_mask, obj_mask, input_pcs, sampled_z\
                    = obj_dict[obj_id].get_training_samples(win_size, n_samples_per_frame, cam_info.rays_dir_cache)
                # merge first two dims, win_size*num_per_frame
                batch_gt_depth.append(gt_depth.reshape([gt_depth.shape[0]*gt_depth.shape[1]]))
                batch_gt_rgb.append(gt_rgb.reshape([gt_rgb.shape[0]*gt_rgb.shape[1], gt_rgb.shape[2]]))
                batch_depth_mask.append(valid_depth_mask)
                batch_obj_mask.append(obj_mask)
                batch_input_pcs.append(input_pcs.reshape([input_pcs.shape[0]*input_pcs.shape[1], input_pcs.shape[2], input_pcs.shape[3]]))
                batch_sampled_z.append(sampled_z.reshape([sampled_z.shape[0]*sampled_z.shape[1], sampled_z.shape[2]]))

                print("input pcs ", input_pcs.shape)
                print("gt depth ", gt_depth.shape)

                # # vis3d   # todo the sampled pcs distribution seems weired between cam2surface
                # # sampled pcs
                # pc = open3d.geometry.PointCloud()
                # pc.points = open3d.utility.Vector3dVector(input_pcs.cpu().numpy().reshape(-1,3))
                # open3d.visualization.draw_geometries([pc])
                # rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
                #     open3d.geometry.Image(rgb_np),
                #     open3d.geometry.Image(depth_np),
                #     depth_trunc=max_depth,
                #     depth_scale=1,
                #     convert_rgb_to_intensity=False,
                # )
                # T_CW = np.linalg.inv(twc_np)
                # # input image pc
                # input_pc = open3d.geometry.PointCloud.create_from_rgbd_image(
                #     image=rgbd,
                #     intrinsic=intrinsic_open3d,
                #     extrinsic=T_CW)
                # open3d.visualization.draw_geometries([pc, input_pc])
                #

        ####################################################
        # training
        assert len(batch_input_pcs) > 0
        # move data to GPU
        batch_input_pcs = torch.stack(batch_input_pcs).to(training_device)
        batch_gt_depth = torch.stack(batch_gt_depth).to(training_device)
        batch_gt_rgb = torch.stack(batch_gt_rgb).to(training_device)
        batch_depth_mask = torch.stack(batch_depth_mask).to(training_device)
        batch_obj_mask = torch.stack(batch_obj_mask).to(training_device)
        batch_sampled_z = torch.stack(batch_sampled_z).to(training_device)

        with performance_measure(f"Training over {len(obj_dict.keys())} objects,"):

            # for loop training
            batch_alpha = []
            batch_color = []
            for k, obj_id in enumerate(obj_dict.keys()):
                obj_k = obj_dict[obj_id]
                embedding_k = obj_k.trainer.pe(batch_input_pcs[k])
                alpha_k, color_k = obj_k.trainer.fc_occ_map(embedding_k)
                batch_alpha.append(alpha_k)
                batch_color.append(color_k)

            batch_alpha = torch.stack(batch_alpha)
            batch_color = torch.stack(batch_color)
            print("batch alpha ", batch_alpha.shape)

            # # batched training
            # batch_alpha = []
            # batch_color = []
            # for k, obj_id in enumerate(obj_dict.keys()):
            #     obj_k = obj_dict[obj_id]
            #     embedding_k = obj_k.trainer.pe(batch_input_pcs[k])
            #     alpha_k, color_k = obj_k.trainer.fc_occ_map(embedding_k)
            #     batch_alpha.append(alpha_k)
            #     batch_color.append(color_k)
            #
            # batch_alpha = torch.stack(batch_alpha)
            # batch_color = torch.stack(batch_color)
            # print("batch alpha ", batch_alpha.shape)


            # step loss
            batch_loss, _ = loss.step_batch_loss(batch_alpha, batch_color,
                                 batch_gt_depth, batch_gt_rgb,
                                 batch_depth_mask, batch_obj_mask,
                                 batch_sampled_z)
            batch_loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            print("loss ", batch_loss)


