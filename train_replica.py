import torch
import cv2
import numpy as np
import os

import loss
from sampling_manager import *
import utils
import open3d
import dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import image_transforms

# todo verify on replica
if __name__ == "__main__":
    ###################################
    # init todo arg parser class
    # hyper param for trainer
    training_device = "cuda:0"
    data_device = "cpu"
    # data_device ="cuda:0"
    # vis_device = "cuda:1"
    imap_mode = False
    win_size = 5 * 20
    n_samples_per_frame = 120 // 5
    min_depth = 0.
    max_depth = 10.
    depth_scale = 1/1000.

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

    # load dataset
    rgb_transform = transforms.Compose(
        [image_transforms.BGRtoRGB()])
    depth_transform = transforms.Compose(
        [image_transforms.DepthScale(depth_scale),
         image_transforms.DepthFilter(max_depth)])
    scene_dataset = dataset.Replica(root_dir, pose_file, rgb_transform, depth_transform, imap_mode=imap_mode)
    # single worker loader
    dataloader = DataLoader(scene_dataset, batch_size=None, shuffle=False, sampler=None,
                                 batch_sampler=None, num_workers=0)
    # # multi worker loader
    # dataloader = DataLoader(scene_dataset, batch_size=1, shuffle=False, sampler=None,
    #                          batch_sampler=None, num_workers=4, collate_fn=None,  # todo
    #                          pin_memory=True, drop_last=False, timeout=0,
    #                          worker_init_fn=None, generator=None, prefetch_factor=2,
    #                          persistent_workers=True)
    dataloader_iterator = iter(dataloader)

    for idx in tqdm(range(dataset_len)):
        # get data from dataloader
        sample = next(dataloader_iterator)
        print(sample["depth"].shape)
        rgb = sample["image"]#.permute(1,0,2)
        depth = sample["depth"]#.permute(1,0)
        twc = sample["T"]
        inst = sample["obj"]#.permute(1,0)
        bbox_dict = sample["bbox_dict"]

        obj_ids = torch.unique(inst)
        with performance_measure(f"Appending data {len(obj_ids)} objects,"):
            # append new frame info to objs in current view
            for obj_id in obj_ids:
                obj_id = int(obj_id)
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
                    with performance_measure(f"single append"):
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

        with performance_measure(f"Samling over {len(obj_dict.keys())} objects,"):
            for obj_id in obj_dict.keys():
                # print("obj_id ", obj_id)
                # with performance_measure(f"Sampling single objects,"):
                gt_rgb, gt_depth, valid_depth_mask, obj_mask, input_pcs, sampled_z\
                    = obj_dict[obj_id].get_training_samples(win_size, n_samples_per_frame, cam_info.rays_dir_cache)
                # merge first two dims, win_size*num_per_frame
                batch_gt_depth.append(gt_depth.reshape([gt_depth.shape[0]*gt_depth.shape[1]]))
                batch_gt_rgb.append(gt_rgb.reshape([gt_rgb.shape[0]*gt_rgb.shape[1], gt_rgb.shape[2]]))
                batch_depth_mask.append(valid_depth_mask)
                batch_obj_mask.append(obj_mask)
                batch_input_pcs.append(input_pcs.reshape([input_pcs.shape[0]*input_pcs.shape[1], input_pcs.shape[2], input_pcs.shape[3]]))
                batch_sampled_z.append(sampled_z.reshape([sampled_z.shape[0]*sampled_z.shape[1], sampled_z.shape[2]]))

                # print("input pcs ", input_pcs.shape)
                # print("gt depth ", gt_depth.shape)
                # input pcs torch.Size([100, 24, 11, 3])
                # gt depth torch.Size([100, 24])

                # vis3d   # todo the sampled pcs distribution seems weired between cam2surface
                # sampled pcs
                pc = open3d.geometry.PointCloud()
                pc.points = open3d.utility.Vector3dVector(input_pcs.cpu().numpy().reshape(-1,3))
                open3d.visualization.draw_geometries([pc])
                rgb_np = rgb.cpu().numpy().astype(np.uint8).transpose(1,0,2)
                # print("rgb ", rgb_np.shape)
                # print(rgb_np)
                # cv2.imshow("rgb", rgb_np)
                # cv2.waitKey(1)
                depth_np = depth.cpu().numpy().astype(np.float32).transpose(1,0)
                twc_np = twc.cpu().numpy()
                rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
                    open3d.geometry.Image(rgb_np),
                    open3d.geometry.Image(depth_np),
                    depth_trunc=max_depth,
                    depth_scale=1,
                    convert_rgb_to_intensity=False,
                )
                T_CW = np.linalg.inv(twc_np)
                # input image pc
                input_pc = open3d.geometry.PointCloud.create_from_rgbd_image(
                    image=rgbd,
                    intrinsic=intrinsic_open3d,
                    extrinsic=T_CW)
                open3d.visualization.draw_geometries([pc, input_pc])


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
            print("loss ", batch_loss)

            batch_loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            print("loss ", batch_loss)


