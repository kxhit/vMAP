import numpy as np
import torch
from pdb import set_trace
from time import perf_counter_ns

# from scalene import scalene_profiler
# import scalene
from tqdm import tqdm
import trainer
import open3d

class performance_measure:

    def __init__(self, name) -> None:
        self.name = name

    def __enter__(self):
        self.start_time = perf_counter_ns()

    def __exit__(self, type, value, tb):
        self.end_time = perf_counter_ns()
        self.exec_time = self.end_time - self.start_time

        print(f"{self.name} excution time: {(self.exec_time)/1000000:.2f} ms")


def origin_dirs_W(T_WC, dirs_C):

    assert T_WC.shape[0] == dirs_C.shape[0]
    assert T_WC.shape[1:] == (4, 4)
    assert dirs_C.shape[2] == 3

    dirs_W = (T_WC[:, None, :3, :3] @ dirs_C[..., None]).squeeze()

    origins = T_WC[:, :3, -1]

    return origins, dirs_W


# bin_limits_scale = torch.linspace(0, 1, n_bins, dtype=torch.float32, device=device)

# depth_range = max_depth - min_depth
# # set_trace()
# # lower_limits_scale = bin_limits_scale * depth_range + min_depth
# lower_limits_scale = bin_limits_scale[..., None] * depth_range + min_depth
# lower_limits_scale = lower_limits_scale.squeeze(dim=-1)

# bin_length_scale = depth_range / n_bins
# increments_scale = torch.rand(
#     n_rays, n_bins, device=device,
#     dtype=torch.float32) * bin_length_scale[:, None]

# z_vals_scale = lower_limits_scale[:, None] + increments_scale

# @torch.jit.script
def stratified_bins(min_depth, max_depth, n_bins, n_rays, type=torch.float32, device = "cpu"):
    # type: (Tensor, Tensor, int, int) -> Tensor

    bin_limits_scale = torch.linspace(0, 1, n_bins+1, dtype=type, device=device)

    if not torch.is_tensor(min_depth):
        min_depth = torch.ones(n_rays, dtype=type, device=device) * min_depth
    
    if not torch.is_tensor(max_depth):
        max_depth = torch.ones(n_rays, dtype=type, device=device) * max_depth

    depth_range = max_depth - min_depth
  
    lower_limits_scale = depth_range[..., None] * bin_limits_scale + min_depth[..., None]
    lower_limits_scale = lower_limits_scale[:, :-1]

    assert lower_limits_scale.shape == (n_rays, n_bins)

    bin_length_scale = depth_range / n_bins
    increments_scale = torch.rand(
        n_rays, n_bins, device=device,
        dtype=torch.float32) * bin_length_scale[..., None]

    z_vals_scale = lower_limits_scale + increments_scale

    assert z_vals_scale.shape == (n_rays, n_bins)

    return z_vals_scale

# @torch.jit.script
def normal_bins_sampling(depth, n_bins, n_rays, delta, device = "cpu"):
    # type: (Tensor, int, int, float) -> Tensor

    # device = "cpu"
    # bins = torch.normal(0.0, delta / 3., size=[n_rays, n_bins], devi
        # self.keyframes_batch = torch.empty(self.n_keyframes,ce=device).sort().values
    bins = torch.empty(n_rays, n_bins, dtype=torch.float32, device=device).normal_(mean=0.,std=delta / 3.).sort().values
    bins = torch.clip(bins, -delta, delta)
    z_vals = depth[:, None] + bins

    assert z_vals.shape == (n_rays, n_bins)

    return z_vals


class sceneObject:
    """
    TODO fill...
    """

    def __init__(self, device, rgb:torch.tensor, depth:torch.tensor, mask:torch.tensor, bbox_2d:torch.tensor, t_wc:torch.tensor) -> None:
        self.device = device

        assert rgb.shape[:2] == depth.shape
        assert rgb.shape[:2] == mask.shape
        assert bbox_2d.shape == (4,)
        assert t_wc.shape == (4, 4,)

        self.frames_width = rgb.shape[0]
        self.frames_height = rgb.shape[1]

        # TODO: what should these be set as??
        self.min_bound = 0.0
        self.max_bound = 10.0

        self.n_keyframes = 1  # Number of keyframes
        self.keyframe_buffer_size = 20
        self.keyframe_step = 25 # for bg
        self.frame_cnt = 0  # number of frames taken in

        self.bbox = torch.empty(  # obj bounding bounding box in the frame
            self.keyframe_buffer_size,
            4,
            device=self.device)  # [u low, u high, v low, v high]
        self.bbox[0] = bbox_2d

        # RGB + pixel state batch
        self.rgb_idx = slice(0, 3)
        self.state_idx = slice(3, 4)
        self.rgbs_batch = torch.empty(self.keyframe_buffer_size,
                                      self.frames_width,
                                      self.frames_height,
                                      4,
                                      dtype=torch.uint8,
                                      device=self.device)

        # Pixel states:
        self.other_obj = 0  # pixel doesn't belong to obj
        self.this_obj = 1  # pixel belong to obj 
        self.unkown_obj = 2  # pixel state is unknown 

        # Initialize first frame rgb and pixel state
        self.rgbs_batch[0, :, :, self.rgb_idx] = rgb
        self.rgbs_batch[0, :, :, self.state_idx] = mask[..., None]

        self.depth_batch = torch.empty(self.keyframe_buffer_size,
                                       self.frames_width,
                                       self.frames_height,
                                       dtype=torch.float32,
                                       device=self.device)

        # Initialize first frame's depth 
        self.depth_batch[0] = depth

        self.t_wc_batch = torch.empty(
            self.keyframe_buffer_size, 4, 4,
            dtype=torch.float32,
            device=self.device)  # world to camera transform

        # Initialize first frame's world2cam transform
        self.t_wc_batch[0] = t_wc

        # network map
        self.trainer = trainer.Trainer()

        # 3D boundary
        self.bbox3d = None
        self.pc = []

    # @profile
    def append_keyframe(self, rgb:torch.tensor, depth:torch.tensor, mask:torch.tensor, bbox_2d:torch.tensor, t_wc:torch.tensor, is_kf:bool):
        # todo if kf: append, else: replace
        assert rgb.shape[:2] == depth.shape
        assert rgb.shape[:2] == mask.shape
        assert bbox_2d.shape == (4,)
        assert t_wc.shape == (4, 4,)
        assert self.n_keyframes < self.keyframe_buffer_size - 1
        assert rgb.dtype == torch.uint8
        assert mask.dtype == torch.uint8
        assert depth.dtype == torch.float32

        # every kf_step choose one kf
        is_kf = self.frame_cnt % self.keyframe_step == 0

        if not is_kf:   # not kf, replace
            self.rgbs_batch[self.n_keyframes-1, :, :, self.rgb_idx] = rgb
            self.rgbs_batch[self.n_keyframes-1, :, :, self.state_idx] = mask[..., None]
            self.depth_batch[self.n_keyframes-1, ...] = depth
            self.t_wc_batch[self.n_keyframes-1, ...] = t_wc
            self.bbox[self.n_keyframes-1, ...] = bbox_2d

        else:   # is kf, add new kf
            self.rgbs_batch[self.n_keyframes, :, :, self.rgb_idx] = rgb
            self.rgbs_batch[self.n_keyframes, :, :, self.state_idx] = mask[..., None]
            self.depth_batch[self.n_keyframes, ...] = depth
            self.t_wc_batch[self.n_keyframes, ...] = t_wc
            self.bbox[self.n_keyframes, ...] = bbox_2d

            self.n_keyframes +=1

        if self.n_keyframes == self.keyframe_buffer_size - 1:
            self.n_keyframes -= 1

        self.frame_cnt += 1

    def get_bound(self, intrinsic_open3d):
        # get 3D boundary from posed depth img
        pcs = open3d.geometry.PointCloud()
        for kf_id in range(self.n_keyframes):
            mask = self.rgbs_batch[kf_id, : , :, self.state_idx].squeeze() == self.this_obj
            depth = self.depth_batch[kf_id].cpu().numpy().copy()
            twc = self.t_wc_batch[kf_id].cpu().numpy()
            depth[~mask] = 0
            T_CW = np.linalg.inv(twc)
            pc = open3d.geometry.PointCloud.create_from_depth_image(
                depth=open3d.geometry.Image(depth),
                intrinsic=intrinsic_open3d,
                extrinsic=T_CW,
            )
            # self.pc += pc
            pcs += pc

        # get minimal oriented 3d bbox
        try:
            bbox3d = open3d.geometry.OrientedBoundingBox.create_from_points(pcs.points)
        except RuntimeError:
            print("too few pcs obj ")
            # self.pc = []
            return None

        # self.pc = []
        return bbox3d



    def get_training_samples(self, n_frames, n_samples, cached_rays_dir):
        # Sample pixels
        keyframe_ids = torch.randint(low=0,
                                     high=self.n_keyframes,
                                     size=(n_frames,),
                                     dtype=torch.long,
                                     device=self.device)    # todo make sure latest 2 frames are sampled
        keyframe_ids = torch.unsqueeze(keyframe_ids, dim=-1)

        idx_w = torch.rand(n_frames, n_samples, device=self.device)
        idx_h = torch.rand(n_frames, n_samples, device=self.device)

        # resizing idx_w and idx_h to be in the bbox range
        idx_w = idx_w * (self.bbox[keyframe_ids, 1] - self.bbox[keyframe_ids, 0]) + self.bbox[keyframe_ids, 0]
        idx_h = idx_h * (self.bbox[keyframe_ids, 3] - self.bbox[keyframe_ids, 2]) + self.bbox[keyframe_ids, 2]

        idx_w = idx_w.long()
        idx_h = idx_h.long()

        sampled_rgbs = self.rgbs_batch[keyframe_ids, idx_w, idx_h]
        sampled_depth = self.depth_batch[keyframe_ids, idx_w, idx_h]

        # Get ray directions for sampled pixels
        sampled_ray_dirs = cached_rays_dir[idx_w, idx_h]

        # Get sampled keyframe poses
        sampled_twc = self.t_wc_batch[keyframe_ids[:, 0], :, :]

        origins, dirs_w = origin_dirs_W(sampled_twc, sampled_ray_dirs)

        return self.sample_3d_points(sampled_rgbs, sampled_depth, origins, dirs_w)

    def sample_3d_points(self, sampled_rgbs, sampled_depth, origins, dirs_w):
        """
        3D sampling strategy

        * For pixels with invalid depth:
            - N+M from minimum bound to max (stratified)
        
        * For pixels with valid depth:
            # Pixel belongs to this object
                - N from cam to surface (stratified)
                - M around surface (stratified/normal)
            # Pixel belongs that don't belong to this object
                - N from cam to surface (stratified)
                - M around surface (stratified)
            # Pixel with unknown state
                - Do nothing!
        """

        # TODO parametrize those vars
        n_bins_cam2surface = 2
        n_bins = 9
        eps = 0.03
        other_objs_max_eps = 0.02
        # print("max depth ", torch.max(sampled_depth))
        sampled_z = torch.zeros(
            sampled_rgbs.shape[0] * sampled_rgbs.shape[1],
            n_bins_cam2surface + n_bins,
            dtype=self.depth_batch.dtype,
            device=self.device)  # shape (N*n_rays, n_bins_cam2surface + n_bins)

        # TODO: parametrize what is considered as zero depth
        invalid_depth_mask = (sampled_depth < 0.001).view(-1)

        # sampling for points with invalid depth
        invalid_depth_count = invalid_depth_mask.count_nonzero()
        if invalid_depth_count:
            print("invalid_depth_count ", invalid_depth_count)
            sampled_z[invalid_depth_mask, :] = stratified_bins(
                self.min_bound, self.max_bound,
                n_bins_cam2surface + n_bins, invalid_depth_count)

        # sampling for valid depth rays
        valid_depth_mask = ~invalid_depth_mask
        valid_depth_count = valid_depth_mask.count_nonzero()


        if valid_depth_count:
            # Sample between min bound and depth for all pixels with valid depth
            sampled_z[valid_depth_mask, :n_bins_cam2surface] = stratified_bins(
                self.min_bound, sampled_depth.view(-1)[valid_depth_mask],
                n_bins_cam2surface, valid_depth_count)

            # sampling around depth for this object
            obj_mask = (sampled_rgbs[..., -1] == self.this_obj).view(-1) & valid_depth_mask # todo obj_mask
            assert sampled_z.shape[0] == obj_mask.shape[0]
            obj_count = obj_mask.count_nonzero()

            if obj_count:
                sampling_method = "stratified"  # stratified or normal
                if sampling_method == "stratified":
                    sampled_z[obj_mask, n_bins_cam2surface:] = stratified_bins(
                        sampled_depth.view(-1)[obj_mask] - eps, sampled_depth.view(-1)[obj_mask] + eps,
                        n_bins, obj_count)

                elif sampling_method == "normal":
                    sampled_z[obj_mask, n_bins_cam2surface:] = normal_bins_sampling(
                        sampled_depth.view(-1)[obj_mask],
                        n_bins,
                        obj_count,
                        delta=eps)

                else:
                    raise (
                        f"sampling method not implemented {sampling_method}, \
                            stratified and normal sampling only currenty implemented."
                    )

            # sampling around depth of other objects
            other_obj_mask = (sampled_rgbs[..., -1] != self.this_obj).view(-1) & valid_depth_mask
            other_objs_count = other_obj_mask.count_nonzero()
            if other_objs_count:
                sampled_z[other_obj_mask, n_bins_cam2surface:] = stratified_bins(
                    sampled_depth.view(-1)[other_obj_mask] - eps,
                    sampled_depth.view(-1)[other_obj_mask] + other_objs_max_eps,
                    n_bins, other_objs_count)

        sampled_z = sampled_z.view(sampled_rgbs.shape[0],
                                   sampled_rgbs.shape[1],
                                   -1)  # view as (n_rays, n_samples, 10)
        input_pcs = origins[..., None, None, :] + (dirs_w[:, :, None, :] *
                                                   sampled_z[..., None])
        # todo obj_center
        obj_labels = (sampled_rgbs[..., -1] == self.this_obj).view(-1)
        # todo standard output tensor shape!
        return sampled_rgbs[..., :3], sampled_depth, valid_depth_mask, obj_labels, input_pcs, sampled_z


class cameraInfo:

    def __init__(self, w, h, a, b, c, d, device) -> None:
        self.device = device
        self.width = w  # Frame width
        self.height = h  # Frame height

        self.fx = a
        self.fy = b
        self.cx = c
        self.cy = d

        self.rays_dir_cache = self.get_rays_dirs()

    def get_rays_dirs(self, depth_type="z"):
        idx_w = torch.arange(end=self.width, device=self.device)
        idx_h = torch.arange(end=self.height, device=self.device)

        dirs = torch.ones((self.width, self.height, 3), device=self.device)

        dirs[:, :, 0] = ((idx_w - self.cx) / self.fx)[:, None]
        dirs[:, :, 1] = ((idx_h - self.cy) / self.fy)

        if depth_type == "euclidean":
            raise Exception(
                "Get camera rays directions with euclidean depth not yet implemented"
            )
            norm = torch.norm(dirs, dim=-1)
            dirs = dirs * (1. / norm)[:, :, :, None]

        return dirs


if __name__ == "__main__":
    # scalene_profiler.start()
    # my_device = "cuda:0"
    my_device = "cpu"

    cam_info = cameraInfo(1280, 620, 600.0, 600.0, 599.5, 339.5, my_device)

    n_objs = 100
    n_frames = 5
    n_samples_per_frame = 24

    # Creating 100 objects
    # scene_objects = [scene_obj] * n_objs
    scene_objects = []

    rgb = torch.ones(1280, 620, 3, dtype=torch.uint8, device=my_device)
    state = torch.ones(1280, 620, dtype=torch.uint8, device=my_device)
    depth = torch.ones(1280, 620, dtype=torch.float32, device=my_device)
    twc = torch.eye(4, dtype=torch.float32, device=my_device)
    bbox = torch.tensor([200, 600, 200, 600], dtype=torch.float32, device=my_device)    # todo which seq? w_min, w_max, h_min, h_max?

    for i in tqdm(range(n_objs)):
        scene_objects.append(sceneObject(my_device, rgb, depth, state, bbox, twc))

    # set_trace()
    print("generated objects")

    for _ in range(20):
        with performance_measure(f"Looping over {len(scene_objects)} objects,"):
            for obj in scene_objects:
                ret = obj.get_training_samples(n_frames, n_samples_per_frame,
                                               cam_info.rays_dir_cache)

    # scalene_profiler.stop()