import json
import numpy as np
import os
import utils

class Config:
    def __init__(self, config_file):
        # setting params
        with open(config_file) as json_file:
            config = json.load(json_file)

        # training strategy
        self.do_bg = bool(config["trainer"]["do_bg"])
        self.training_device = config["trainer"]["train_device"]
        self.data_device = config["trainer"]["data_device"]
        self.max_n_models = config["trainer"]["n_models"]
        self.live_mode = bool(config["dataset"]["live"])
        self.keep_live_time = config["dataset"]["keep_alive"]
        self.imap_mode = config["trainer"]["imap_mode"]
        self.training_strategy = config["trainer"]["training_strategy"]  # "forloop" "vmap"
        self.obj_id = -1

        # dataset setting
        self.dataset_format = config["dataset"]["format"]
        self.dataset_dir = config["dataset"]["path"]
        self.depth_scale = 1 / config["trainer"]["scale"]
        # camera setting
        self.max_depth = config["render"]["depth_range"][1]
        self.min_depth = config["render"]["depth_range"][0]
        self.mh = config["camera"]["mh"]
        self.mw = config["camera"]["mw"]
        self.height = config["camera"]["h"]
        self.width = config["camera"]["w"]
        self.H = self.height - 2 * self.mh
        self.W = self.width - 2 * self.mw
        if "fx" in config["camera"]:
            self.fx = config["camera"]["fx"]
            self.fy = config["camera"]["fy"]
            self.cx = config["camera"]["cx"] - self.mw
            self.cy = config["camera"]["cy"] - self.mh
        else:   # for scannet
            intrinsic = utils.load_matrix_from_txt(os.path.join(self.dataset_dir, "intrinsic/intrinsic_depth.txt"))
            self.fx = intrinsic[0, 0]
            self.fy = intrinsic[1, 1]
            self.cx = intrinsic[0, 2] - self.mw
            self.cy = intrinsic[1, 2] - self.mh
        if "distortion" in config["camera"]:
            self.distortion_array = np.array(config["camera"]["distortion"])
        elif "k1" in config["camera"]:
            k1 = config["camera"]["k1"]
            k2 = config["camera"]["k2"]
            k3 = config["camera"]["k3"]
            k4 = config["camera"]["k4"]
            k5 = config["camera"]["k5"]
            k6 = config["camera"]["k6"]
            p1 = config["camera"]["p1"]
            p2 = config["camera"]["p2"]
            self.distortion_array = np.array([k1, k2, p1, p2, k3, k4, k5, k6])
        else:
            self.distortion_array = None

        # training setting
        self.win_size = config["model"]["window_size"]
        self.n_iter_per_frame = config["render"]["iters_per_frame"]
        self.n_per_optim = config["render"]["n_per_optim"]
        self.n_samples_per_frame = self.n_per_optim // self.win_size
        self.win_size_bg = config["model"]["window_size_bg"]
        self.n_per_optim_bg = config["render"]["n_per_optim_bg"]
        self.n_samples_per_frame_bg = self.n_per_optim_bg // self.win_size_bg
        self.keyframe_buffer_size = config["model"]["keyframe_buffer_size"]
        self.keyframe_step = config["model"]["keyframe_step"]
        self.keyframe_step_bg = config["model"]["keyframe_step_bg"]
        self.obj_scale = config["model"]["obj_scale"]
        self.bg_scale = config["model"]["bg_scale"]
        self.hidden_feature_size = config["model"]["hidden_feature_size"]
        self.hidden_feature_size_bg = config["model"]["hidden_feature_size_bg"]
        self.n_bins_cam2surface = config["render"]["n_bins_cam2surface"]
        self.n_bins_cam2surface_bg = config["render"]["n_bins_cam2surface_bg"]
        self.n_bins = config["render"]["n_bins"]
        self.n_unidir_funcs = config["model"]["n_unidir_funcs"]
        self.surface_eps = config["model"]["surface_eps"]
        self.stop_eps = config["model"]["other_eps"]

        # optimizer setting
        self.learning_rate = config["optimizer"]["args"]["lr"]
        self.weight_decay = config["optimizer"]["args"]["weight_decay"]

        # vis setting
        self.vis_device = config["vis"]["vis_device"]
        self.n_vis_iter = config["vis"]["n_vis_iter"]
        self.live_voxel_size = config["vis"]["live_voxel_size"]
        self.grid_dim = config["vis"]["grid_dim"]