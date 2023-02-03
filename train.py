import time
import loss
from vmap import *
import utils
import open3d
import dataset
import vis
from functorch import vmap
import argparse
from cfg import Config
import shutil

if __name__ == "__main__":
    #############################################
    # init config
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # setting params
    parser = argparse.ArgumentParser(description='Model training for single GPU')
    parser.add_argument('--logdir', default='./logs/debug',
                        type=str)
    parser.add_argument('--config',
                        default='./configs/Replica/config_replica_room0_vMAP.json',
                        type=str)
    parser.add_argument('--save_ckpt',
                        default=False,
                        type=bool)
    args = parser.parse_args()

    log_dir = args.logdir
    config_file = args.config
    save_ckpt = args.save_ckpt
    os.makedirs(log_dir, exist_ok=True)  # saving logs
    shutil.copy(config_file, log_dir)
    cfg = Config(config_file)       # config params
    n_sample_per_step = cfg.n_per_optim
    n_sample_per_step_bg = cfg.n_per_optim_bg

    # param for vis
    vis3d = open3d.visualization.Visualizer()
    vis3d.create_window(window_name="3D mesh vis",
                        width=cfg.W,
                        height=cfg.H,
                        left=600, top=50)
    view_ctl = vis3d.get_view_control()
    view_ctl.set_constant_z_far(10.)

    # set camera
    cam_info = cameraInfo(cfg)
    intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
        width=cfg.W,
        height=cfg.H,
        fx=cfg.fx,
        fy=cfg.fy,
        cx=cfg.cx,
        cy=cfg.cy)

    # init obj_dict
    obj_dict = {}   # only objs
    vis_dict = {}   # including bg

    # init for training
    AMP = False
    if AMP:
        scaler = torch.cuda.amp.GradScaler()  # amp https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
    optimiser = torch.optim.AdamW([torch.autograd.Variable(torch.tensor(0))], lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    #############################################
    # init data stream
    if not cfg.live_mode:
        # load dataset
        dataloader = dataset.init_loader(cfg)
        dataloader_iterator = iter(dataloader)
        dataset_len = len(dataloader)
    else:
        dataset_len = 1000000
        # # init ros node
        # torch.multiprocessing.set_start_method('spawn')  # spawn
        # import ros_nodes
        # track_to_map_Buffer = torch.multiprocessing.Queue(maxsize=5)
        # # track_to_vis_T_WC = torch.multiprocessing.Queue(maxsize=1)
        # kfs_que = torch.multiprocessing.Queue(maxsize=5)  # to store one more buffer
        # track_p = torch.multiprocessing.Process(target=ros_nodes.Tracking,
        #                                              args=(
        #                                              (cfg), (track_to_map_Buffer), (None),
        #                                              (kfs_que), (True),))
        # track_p.start()


    # init vmap
    fc_models, pe_models = [], []
    scene_bg = None

    for frame_id in tqdm(range(dataset_len)):
        print("*********************************************")
        # get new frame data
        with performance_measure(f"getting next data"):
            if not cfg.live_mode:
                # get data from dataloader
                sample = next(dataloader_iterator)
            else:
                pass

        if sample is not None:  # new frame
            last_frame_time = time.time()
            with performance_measure(f"Appending data"):
                rgb = sample["image"].to(cfg.data_device)
                depth = sample["depth"].to(cfg.data_device)
                twc = sample["T"].to(cfg.data_device)
                bbox_dict = sample["bbox_dict"]
                if "frame_id" in sample.keys():
                    live_frame_id = sample["frame_id"]
                else:
                    live_frame_id = frame_id
                if not cfg.live_mode:
                    inst = sample["obj"].to(cfg.data_device)
                    obj_ids = torch.unique(inst)
                else:
                    inst_data_dict = sample["obj"]
                    obj_ids = inst_data_dict.keys()
                # append new frame info to objs in current view
                for obj_id in obj_ids:
                    if obj_id == -1:    # unsured area
                        continue
                    obj_id = int(obj_id)
                    # convert inst mask to state
                    if not cfg.live_mode:
                        state = torch.zeros_like(inst, dtype=torch.uint8, device=cfg.data_device)
                        state[inst == obj_id] = 1
                        state[inst == -1] = 2
                    else:
                        inst_mask = inst_data_dict[obj_id].permute(1,0)
                        label_list = torch.unique(inst_mask).tolist()
                        state = torch.zeros_like(inst_mask, dtype=torch.uint8, device=cfg.data_device)
                        state[inst_mask == obj_id] = 1
                        state[inst_mask == -1] = 2
                    bbox = bbox_dict[obj_id]
                    if obj_id in vis_dict.keys():
                        scene_obj = vis_dict[obj_id]
                        scene_obj.append_keyframe(rgb, depth, state, bbox, twc, live_frame_id)
                    else: # init scene_obj
                        if len(obj_dict.keys()) >= cfg.max_n_models:
                            print("models full!!!! current num ", len(obj_dict.keys()))
                            continue
                        print("init new obj ", obj_id)
                        if cfg.do_bg and obj_id == 0:   # todo param
                            scene_bg = sceneObject(cfg, obj_id, rgb, depth, state, bbox, twc, live_frame_id)
                            # scene_bg.init_obj_center(intrinsic_open3d, depth, state, twc)
                            optimiser.add_param_group({"params": scene_bg.trainer.fc_occ_map.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                            optimiser.add_param_group({"params": scene_bg.trainer.pe.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                            vis_dict.update({obj_id: scene_bg})
                        else:
                            scene_obj = sceneObject(cfg, obj_id, rgb, depth, state, bbox, twc, live_frame_id)
                            # scene_obj.init_obj_center(intrinsic_open3d, depth, state, twc)
                            obj_dict.update({obj_id: scene_obj})
                            vis_dict.update({obj_id: scene_obj})
                            # params = [scene_obj.trainer.fc_occ_map.parameters(), scene_obj.trainer.pe.parameters()]
                            optimiser.add_param_group({"params": scene_obj.trainer.fc_occ_map.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                            optimiser.add_param_group({"params": scene_obj.trainer.pe.parameters(), "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay})
                            if cfg.training_strategy == "vmap":
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
            if cfg.training_strategy == "vmap" and update_vmap_model == True:
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
            if cfg.do_bg and scene_bg is not None:
                gt_rgb, gt_depth, valid_depth_mask, obj_mask, input_pcs, sampled_z \
                    = scene_bg.get_training_samples(cfg.n_iter_per_frame * cfg.win_size_bg, cfg.n_samples_per_frame_bg,
                                                    cam_info.rays_dir_cache)
                bg_gt_depth = gt_depth.reshape([gt_depth.shape[0] * gt_depth.shape[1]])
                bg_gt_rgb = gt_rgb.reshape([gt_rgb.shape[0] * gt_rgb.shape[1], gt_rgb.shape[2]])
                bg_valid_depth_mask = valid_depth_mask
                bg_obj_mask = obj_mask
                bg_input_pcs = input_pcs.reshape(
                    [input_pcs.shape[0] * input_pcs.shape[1], input_pcs.shape[2], input_pcs.shape[3]])
                bg_sampled_z = sampled_z.reshape([sampled_z.shape[0] * sampled_z.shape[1], sampled_z.shape[2]])

            for obj_id, obj_k in obj_dict.items():
                gt_rgb, gt_depth, valid_depth_mask, obj_mask, input_pcs, sampled_z \
                    = obj_k.get_training_samples(cfg.n_iter_per_frame * cfg.win_size, cfg.n_samples_per_frame,
                                                 cam_info.rays_dir_cache)
                # merge first two dims, sample_per_frame*num_per_frame
                Batch_N_gt_depth.append(gt_depth.reshape([gt_depth.shape[0] * gt_depth.shape[1]]))
                Batch_N_gt_rgb.append(gt_rgb.reshape([gt_rgb.shape[0] * gt_rgb.shape[1], gt_rgb.shape[2]]))
                Batch_N_depth_mask.append(valid_depth_mask)
                Batch_N_obj_mask.append(obj_mask)
                Batch_N_input_pcs.append(input_pcs.reshape([input_pcs.shape[0] * input_pcs.shape[1], input_pcs.shape[2], input_pcs.shape[3]]))
                Batch_N_sampled_z.append(sampled_z.reshape([sampled_z.shape[0] * sampled_z.shape[1], sampled_z.shape[2]]))

                # # vis sampled points in open3D
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

            Batch_N_input_pcs = torch.stack(Batch_N_input_pcs).to(cfg.training_device)
            Batch_N_gt_depth = torch.stack(Batch_N_gt_depth).to(cfg.training_device)
            Batch_N_gt_rgb = torch.stack(Batch_N_gt_rgb).to(cfg.training_device) / 255. # todo
            Batch_N_depth_mask = torch.stack(Batch_N_depth_mask).to(cfg.training_device)
            Batch_N_obj_mask = torch.stack(Batch_N_obj_mask).to(cfg.training_device)
            Batch_N_sampled_z = torch.stack(Batch_N_sampled_z).to(cfg.training_device)
            if cfg.do_bg:
                bg_input_pcs = bg_input_pcs.to(cfg.training_device)
                bg_gt_depth = bg_gt_depth.to(cfg.training_device)
                bg_gt_rgb = bg_gt_rgb.to(cfg.training_device) / 255.
                bg_valid_depth_mask = bg_valid_depth_mask.to(cfg.training_device)
                bg_obj_mask = bg_obj_mask.to(cfg.training_device)
                bg_sampled_z = bg_sampled_z.to(cfg.training_device)

        with performance_measure(f"Training over {len(obj_dict.keys())} objects,"):
            for iter_step in range(cfg.n_iter_per_frame):
                data_idx = slice(iter_step*n_sample_per_step, (iter_step+1)*n_sample_per_step)
                batch_input_pcs = Batch_N_input_pcs[:, data_idx, ...]
                batch_gt_depth = Batch_N_gt_depth[:, data_idx, ...]
                batch_gt_rgb = Batch_N_gt_rgb[:, data_idx, ...]
                batch_depth_mask = Batch_N_depth_mask[:, data_idx, ...]
                batch_obj_mask = Batch_N_obj_mask[:, data_idx, ...]
                batch_sampled_z = Batch_N_sampled_z[:, data_idx, ...]
                if cfg.training_strategy == "forloop":
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
                elif cfg.training_strategy == "vmap":
                    # batched training
                    batch_embedding = vmap(pe_model)(pe_param, pe_buffer, batch_input_pcs)
                    batch_alpha, batch_color = vmap(fc_model)(fc_param, fc_buffer, batch_embedding)
                    # print("batch alpha ", batch_alpha.shape)
                else:
                    print("training strategy {} is not implemented ".format(cfg.training_strategy))
                    exit(-1)


            # step loss
            # with performance_measure(f"Batch LOSS"):
                batch_loss, _ = loss.step_batch_loss(batch_alpha, batch_color,
                                     batch_gt_depth.detach(), batch_gt_rgb.detach(),
                                     batch_obj_mask.detach(), batch_depth_mask.detach(),
                                     batch_sampled_z.detach())

                if cfg.do_bg:
                    bg_data_idx = slice(iter_step * n_sample_per_step_bg, (iter_step + 1) * n_sample_per_step_bg)
                    bg_embedding = scene_bg.trainer.pe(bg_input_pcs[bg_data_idx, ...])
                    bg_alpha, bg_color = scene_bg.trainer.fc_occ_map(bg_embedding)
                    bg_loss, _ = loss.step_batch_loss(bg_alpha[None, ...], bg_color[None, ...],
                                                     bg_gt_depth[None, bg_data_idx, ...].detach(), bg_gt_rgb[None, bg_data_idx].detach(),
                                                     bg_obj_mask[None, bg_data_idx, ...].detach(), bg_valid_depth_mask[None, bg_data_idx, ...].detach(),
                                                     bg_sampled_z[None, bg_data_idx, ...].detach())
                    batch_loss += bg_loss

            # with performance_measure(f"Backward"):
                if AMP:
                    scaler.scale(batch_loss).backward()
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    batch_loss.backward()
                    optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                # print("loss ", batch_loss.item())

        # update each origin model params
        # todo find a better way    # https://github.com/pytorch/functorch/issues/280
        with performance_measure(f"updating vmap param"):
            if cfg.training_strategy == "vmap":
                with torch.no_grad():
                    for model_id, (obj_id, obj_k) in enumerate(obj_dict.items()):
                        for i, param in enumerate(obj_k.trainer.fc_occ_map.parameters()):
                            param.copy_(fc_param[i][model_id])
                        for i, param in enumerate(obj_k.trainer.pe.parameters()):
                            param.copy_(pe_param[i][model_id])


        ####################################################################
        # live vis mesh
        if (((frame_id % cfg.n_vis_iter) == 0 or frame_id == dataset_len-1) or
            (cfg.live_mode and time.time()-last_frame_time>cfg.keep_live_time)) and frame_id >= 10:
            vis3d.clear_geometries()
            for obj_id, obj_k in vis_dict.items():
                bound = obj_k.get_bound(intrinsic_open3d)
                if bound is None:
                    print("get bound failed obj ", obj_id)
                    continue
                adaptive_grid_dim = int(np.minimum(np.max(bound.extent)//cfg.live_voxel_size+1, cfg.grid_dim))
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

        if False:    # follow cam
            cam = view_ctl.convert_to_pinhole_camera_parameters()
            T_CW_np = np.linalg.inv(twc.cpu().numpy())
            cam.extrinsic = T_CW_np
            view_ctl.convert_from_pinhole_camera_parameters(cam)
            vis3d.poll_events()
            vis3d.update_renderer()

        with performance_measure("saving ckpt"):
            if save_ckpt and ((((frame_id % cfg.n_vis_iter) == 0 or frame_id == dataset_len - 1) or
                               (cfg.live_mode and time.time() - last_frame_time > cfg.keep_live_time)) and frame_id >= 10):
                for obj_id, obj_k in vis_dict.items():
                    ckpt_dir = os.path.join(log_dir, "ckpt", str(obj_id))
                    os.makedirs(ckpt_dir, exist_ok=True)
                    bound = obj_k.get_bound(intrinsic_open3d)   # update bound
                    obj_k.save_checkpoints(ckpt_dir, frame_id)
                # save current cam pose
                cam_dir = os.path.join(log_dir, "cam_pose")
                os.makedirs(cam_dir, exist_ok=True)
                torch.save({"twc": twc,}, os.path.join(cam_dir, "twc_frame_{}".format(frame_id) + ".pth"))


