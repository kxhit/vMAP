import time

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
import vis
import json

# vmap
from functorch import vmap

# todo verify on replica
if __name__ == "__main__":
    ###################################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    # init todo arg parser class
    # hyper param for trainer
    log_dir = "logs/TUM_vmap_swin_min200_box1.05"
    training_device = "cuda:0"
    # data_device = "cpu"
    data_device ="cuda:0"
    # vis_device = "cuda:1"
    max_n_models = 50   # max models number
    live_mode = True
    if live_mode:
        keep_live_time = 20.    # after this waiting time, finish training and then eval

    imap_mode = False #False
    training_strategy = "vmap" # "forloop" "vmap"
    win_size = 5
    n_iter_per_frame = 20
    n_samples_per_frame = 120 // 5 #120 // 5
    n_sample_per_step = n_samples_per_frame * win_size
    min_depth = 0.
    max_depth = 10.
    depth_scale = 1/1000.

    # param for vis
    vis_iter_step = 1000000000
    vis3d = open3d.visualization.Visualizer()
    vis3d.create_window(window_name="3D mesh vis",
                        width=1200,
                        height=680,
                        left=600, top=50)
    view_ctl = vis3d.get_view_control()
    view_ctl.set_constant_z_far(10.)

    # param for dataset
    bbox_scale = 0.2

    if live_mode:
        config_file = "./configs/TUM/config_TUM2_live_bMAP.json"
        with open(config_file) as json_file:
            config = json.load(json_file)
        # camera
        max_bound = config["render"]["depth_range"][1]
        min_bound = config["render"]["depth_range"][0]
        mh = config["camera"]["mh"]
        mw = config["camera"]["mw"]
        height = config["camera"]["h"]
        width = config["camera"]["w"]
        H = height - 2 * mh
        W = width - 2 * mw
        fx = config["camera"]["fx"]
        fy = config["camera"]["fy"]
        cx = config["camera"]["cx"] - mw
        cy = config["camera"]["cy"] - mh
    else:
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

    #############################################
    # todo dataloader, dataset
    if live_mode:
        # todo param
        dataset_len = 1000000
    else:
        # get one frame
        root_dir = "/home/xin/data/Replica/replica_v1/room_0/imap/00/"
        pose_file = os.path.join(root_dir, "traj_w_c.txt")
        pose_all = np.loadtxt(pose_file, delimiter=" ").reshape([-1, 4, 4]).astype(np.float32)
        dataset_len = pose_all.shape[0]

    # init data stream
    if not live_mode:
        # load dataset
        rgb_transform = transforms.Compose(
            [image_transforms.BGRtoRGB()])
        depth_transform = transforms.Compose(
            [image_transforms.DepthScale(depth_scale),
             image_transforms.DepthFilter(max_depth)])
        scene_dataset = dataset.Replica(root_dir, pose_file, rgb_transform, depth_transform, imap_mode=imap_mode)
        # # single worker loader
        # dataloader = DataLoader(scene_dataset, batch_size=None, shuffle=False, sampler=None,
        #                              batch_sampler=None, num_workers=0)
        # multi worker loader
        dataloader = DataLoader(scene_dataset, batch_size=None, shuffle=False, sampler=None,
                                 batch_sampler=None, num_workers=4, collate_fn=None,  # todo
                                 pin_memory=True, drop_last=False, timeout=0,
                                 worker_init_fn=None, generator=None, prefetch_factor=2,
                                 persistent_workers=True)
        dataloader_iterator = iter(dataloader)
    else:
        # init ros node
        torch.multiprocessing.set_start_method('spawn')  # spawn
        import ros_nodes
        track_to_map_Buffer = torch.multiprocessing.Queue(maxsize=5)  # todo
        # track_to_vis_T_WC = torch.multiprocessing.Queue(maxsize=1)
        kfs_que = torch.multiprocessing.Queue(maxsize=5)  # to store one more buffer
        track_p = torch.multiprocessing.Process(target=ros_nodes.Tracking,
                                                     args=(
                                                     (config), (track_to_map_Buffer), (None),
                                                     (kfs_que), (imap_mode), (True),))
        track_p.start()


    # init vmap
    fc_models, pe_models = [], []

    for frame_id in tqdm(range(dataset_len)):
        print("*********************************************")
        with performance_measure(f"getting data"):
            if not live_mode:
                # get data from dataloader
                sample = next(dataloader_iterator)
            else:
                sample = dataset.next_live_data(track_to_map_Buffer, frame_id!=0)
                # todo update keyframe poses from ORB-SLAM BA
                if not kfs_que.empty():
                    with performance_measure(f"updating keyframe pose"):
                        # Buffer_kfs_updated = render_trainer.kfs_que.get(block=False)
                        Buffer_kfs_updated = utils.get_latest_queue(kfs_que)
                        # print("Buffer_kfs_updated ", Buffer_kfs_updated)
                        if Buffer_kfs_updated is not None:
                            kf_update_dict, = Buffer_kfs_updated
                            del Buffer_kfs_updated
                            # loop over objs
                            for obj_id, obj_k in obj_dict.items():
                                print("obj ", obj_id)
                                print("kf id dict ", obj_k.kf_id_dict)
                                for live_kf_id, kf_id in obj_k.kf_id_dict.items():
                                    if live_kf_id in kf_update_dict.keys():  # update
                                        # print("sub ", obj_k.t_wc_batch[kf_id] - kf_update_dict[live_kf_id])
                                        obj_k.t_wc_batch[kf_id] = kf_update_dict[live_kf_id].clone()
                                        print("update obj {} live_kf_id {} kf_id {}".format(obj_id, live_kf_id, kf_id))
                                    # elif (live_kf_id < np.max(list(kf_update_dict.keys()))) and \
                                    #         len(obj_k.kf_id_dict.keys()) > obj_k.keyframe_buffer_size:  # not added to BA and reached max kf buffer_size, consider prune
                                    #     print("pruned kf ", kf_id)  # todo prune kf
                                    #     # models_dict_live[obj_id].render_trainer.kf_info_list.remove(
                                    #     #     kf)  # todo check batch size
                                    elif live_kf_id > np.max(list(kf_update_dict.keys())):  # not BA yet
                                        break
                                # models_dict_live[obj_id].render_trainer.batch_size = len(
                                #     models_dict_live[obj_id].render_trainer.kf_info_list)
        if sample is not None:  # new frame
            last_frame_time = time.time()
            with performance_measure(f"Appending data"):
                rgb = sample["image"].to(data_device)
                depth = sample["depth"].to(data_device)
                twc = sample["T"].to(data_device)
                inst = sample["obj"].to(data_device)
                bbox_dict = sample["bbox_dict"]
                if "frame_id" in sample.keys():
                    live_frame_id = sample["frame_id"]
                else:
                    live_frame_id = frame_id
                obj_ids = torch.unique(inst)
                # append new frame info to objs in current view
                for obj_id in obj_ids:
                    if obj_id == -1:    # unsured area
                        continue
                    obj_id = int(obj_id)
                    # convert inst mask to state
                    state = torch.zeros_like(inst, dtype=torch.uint8, device=data_device)
                    state[inst == obj_id] = 1
                    state[inst == -1] = 2
                    bbox = bbox_dict[obj_id]
                    if obj_id in obj_dict.keys():
                        scene_obj = obj_dict[obj_id]
                        # with performance_measure(f"single append"):
                        scene_obj.append_keyframe(rgb, depth, state, bbox, twc, live_frame_id)
                    else: # init scene_obj
                        if len(obj_dict.keys()) >= max_n_models:
                            print("models full!!!! current num ", len(obj_dict.keys()))
                            continue
                        scene_obj = sceneObject(data_device, rgb, depth, state, bbox, twc, None, live_frame_id)
                        obj_dict.update({obj_id: scene_obj})
                        # params = [scene_obj.trainer.fc_occ_map.parameters(), scene_obj.trainer.pe.parameters()]
                        optimiser.add_param_group({"params": scene_obj.trainer.fc_occ_map.parameters(), "lr": learning_rate, "weight_decay": weight_decay})
                        optimiser.add_param_group({"params": scene_obj.trainer.pe.parameters(), "lr": learning_rate, "weight_decay": weight_decay})
                        print("init new obj ", obj_id)
                        if training_strategy == "vmap":
                            update_vmap_model = True
                            fc_models.append(obj_dict[obj_id].trainer.fc_occ_map)
                            pe_models.append(obj_dict[obj_id].trainer.pe)

                        # ###################################
                        # # measure trainable params in total
                        # total_params = 0
                        # obj_k = obj_dict[obj_id]
                        # for p in obj_k.trainer.fc_occ_map.parameters():
                        #     if p.requires_grad:
                        #         total_params += p.numel()
                        # for p in obj_k.trainer.pe.parameters():
                        #     if p.requires_grad:
                        #         total_params += p.numel()
                        # print("total param ", total_params)

        # dynamically add vmap
        with performance_measure(f"add vmap"):
            if training_strategy == "vmap" and update_vmap_model == True:
                fc_model, fc_param, fc_buffer = utils.update_vmap(fc_models, optimiser)
                pe_model, pe_param, pe_buffer = utils.update_vmap(pe_models, optimiser)
                update_vmap_model = False


        ##################################################################
        # training data preperation, get training data for all objs
        Batch_N_gt_depth = []
        Batch_N_gt_rgb = []
        Batch_N_depth_mask = []
        Batch_N_obj_mask = []
        Batch_N_input_pcs = []
        Batch_N_sampled_z = []

        with performance_measure(f"Sampling over {len(obj_dict.keys())} objects,"):
            for obj_id, obj_k in obj_dict.items():
                # print("obj_id ", obj_id)
                # with performance_measure(f"Sampling single objects,"):
                gt_rgb, gt_depth, valid_depth_mask, obj_mask, input_pcs, sampled_z\
                    = obj_k.get_training_samples(n_iter_per_frame*win_size, n_samples_per_frame, cam_info.rays_dir_cache)
                # merge first two dims, sample_per_frame*num_per_frame
                Batch_N_gt_depth.append(gt_depth.reshape([gt_depth.shape[0]*gt_depth.shape[1]]))
                Batch_N_gt_rgb.append(gt_rgb.reshape([gt_rgb.shape[0]*gt_rgb.shape[1], gt_rgb.shape[2]]))
                Batch_N_depth_mask.append(valid_depth_mask)
                Batch_N_obj_mask.append(obj_mask)
                Batch_N_input_pcs.append(input_pcs.reshape([input_pcs.shape[0]*input_pcs.shape[1], input_pcs.shape[2], input_pcs.shape[3]]))
                Batch_N_sampled_z.append(sampled_z.reshape([sampled_z.shape[0]*sampled_z.shape[1], sampled_z.shape[2]]))

                # print("input pcs ", input_pcs.shape)
                # print("gt depth ", gt_depth.shape)
                # input pcs torch.Size([100, 24, 11, 3])
                # gt depth torch.Size([100, 24])

                # # vis3d   # todo the sampled pcs distribution seems weired between cam2surface
                # # sampled pcs
                # pc = open3d.geometry.PointCloud()
                # pc.points = open3d.utility.Vector3dVector(input_pcs.cpu().numpy().reshape(-1,3))
                # open3d.visualization.draw_geometries([pc])
                # rgb_np = rgb.cpu().numpy().astype(np.uint8).transpose(1,0,2)
                # # print("rgb ", rgb_np.shape)
                # # print(rgb_np)
                # # cv2.imshow("rgb", rgb_np)
                # # cv2.waitKey(1)
                # depth_np = depth.cpu().numpy().astype(np.float32).transpose(1,0)
                # twc_np = twc.cpu().numpy()
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
                # input_pc.points = open3d.utility.Vector3dVector(np.array(input_pc.points) - obj_k.obj_center.cpu().numpy())
                # open3d.visualization.draw_geometries([pc, input_pc])


        ####################################################
        # training
        assert len(Batch_N_input_pcs) > 0
        # move data to GPU  (n_obj, n_iter_per_frame, win_size*num_per_frame, 3)
        with performance_measure(f"stacking and moving to gpu: "):

            Batch_N_input_pcs = torch.stack(Batch_N_input_pcs).to(training_device)
            Batch_N_gt_depth = torch.stack(Batch_N_gt_depth).to(training_device)
            Batch_N_gt_rgb = torch.stack(Batch_N_gt_rgb).to(training_device) / 255. # todo
            Batch_N_depth_mask = torch.stack(Batch_N_depth_mask).to(training_device)
            Batch_N_obj_mask = torch.stack(Batch_N_obj_mask).to(training_device)
            Batch_N_sampled_z = torch.stack(Batch_N_sampled_z).to(training_device)

        with performance_measure(f"Training over {len(obj_dict.keys())} objects,"):
            for iter_step in range(n_iter_per_frame):
            # for _ in range(1):
            #     data_idx = slice(0, Batch_N_input_pcs.shape[1])
                data_idx = slice(iter_step*n_sample_per_step, (iter_step+1)*n_sample_per_step)
                batch_input_pcs = Batch_N_input_pcs[:, data_idx, ...]
                batch_gt_depth = Batch_N_gt_depth[:, data_idx, ...]
                batch_gt_rgb = Batch_N_gt_rgb[:, data_idx, ...]
                batch_depth_mask = Batch_N_depth_mask[:, data_idx, ...]
                batch_obj_mask = Batch_N_obj_mask[:, data_idx, ...]
                batch_sampled_z = Batch_N_sampled_z[:, data_idx, ...]
                # print("size ", batch_input_pcs.shape)
                if training_strategy == "forloop":
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
                    # print("batch alpha ", batch_alpha.shape)
                elif training_strategy == "vmap":
                    # batched training
                # with performance_measure(f"Batch PE"):
                    batch_embedding = vmap(pe_model)(pe_param, pe_buffer, batch_input_pcs)
                    batch_alpha, batch_color = vmap(fc_model)(fc_param, fc_buffer, batch_embedding)
                    # print("batch alpha ", batch_alpha.shape)
                else:
                    print("training strategy {} is not implemented ".format(training_strategy))
                    exit(-1)


            # step loss
            # with performance_measure(f"Batch LOSS"):
                batch_loss, _ = loss.step_batch_loss(batch_alpha, batch_color,
                                     batch_gt_depth.detach(), batch_gt_rgb.detach(),
                                     batch_obj_mask.detach(), batch_depth_mask.detach(),
                                     batch_sampled_z.detach())
            # with performance_measure(f"Backward"):
                batch_loss.backward()
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                # print("loss ", batch_loss.item())

        # update each origin model params
        # todo find a better way    # https://github.com/pytorch/functorch/issues/280
        with performance_measure(f"updating vmap param"):
            if training_strategy == "vmap":
                with torch.no_grad():
                    for model_id, (obj_id, obj_k) in enumerate(obj_dict.items()):
                        for i, param in enumerate(obj_k.trainer.fc_occ_map.parameters()):
                            param.copy_(fc_param[i][model_id])
                        for i, param in enumerate(obj_k.trainer.pe.parameters()):
                            param.copy_(pe_param[i][model_id])

        # live vis mesh
        if (((frame_id % vis_iter_step) == 0 or frame_id == dataset_len-1) or
            (live_mode and time.time()-last_frame_time>keep_live_time)) and frame_id >= 10:
            vis3d.clear_geometries()
            for obj_id, obj_k in obj_dict.items():
                if obj_id == 0 or obj_k.n_keyframes <= 2:   # too few detections
                    continue
                bound = obj_k.get_bound(intrinsic_open3d)
                print("bound ", bound)
                if bound is None:
                    print("get bound failed obj ", obj_id)
                    continue
                voxel_size = 0.01
                adaptive_grid_dim = int(np.minimum(np.max(bound.extent)//voxel_size+1, 256))
                mesh = obj_k.trainer.meshing(bound, obj_k.obj_center, grid_dim=adaptive_grid_dim)
                if mesh is None:
                    print("meshing failed obj ", obj_id)
                    continue
                # save to dir
                obj_mesh_output = os.path.join(log_dir, "scene_mesh")
                os.makedirs(obj_mesh_output, exist_ok=True)
                mesh.export(os.path.join(obj_mesh_output, "frame_{}_obj{}.obj".format(frame_id, str(obj_id))))

                # live vis
                open3d_mesh = vis.trimesh_to_open3d(mesh)
                vis3d.add_geometry(open3d_mesh)
                vis3d.add_geometry(bound)
                # update vis3d
                vis3d.poll_events()
                vis3d.update_renderer()

        if True:    # follow cam
            cam = view_ctl.convert_to_pinhole_camera_parameters()
            T_CW_np = np.linalg.inv(twc.cpu().numpy())
            cam.extrinsic = T_CW_np
            view_ctl.convert_from_pinhole_camera_parameters(cam)
            vis3d.poll_events()
            vis3d.update_renderer()


