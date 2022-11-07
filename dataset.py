import imgviz
from torch.utils.data import Dataset
import torch
import numpy as np
import trimesh
import cv2
import os
import pickle
from utils import enlarge_bbox, get_bbox2d, get_bbox2d_batch
import glob
import h5py

# from ScenePriors.depth.monodepth import MonoDepth, scale_depth_obj
# from mmdet.apis import inference_detector, init_detector
# from utils import postprocess_mmdet_results, track_instance, box_filter
import open3d
import json
import time


def next_live_data(track_to_map_IDT, inited):
    while True:
        if track_to_map_IDT.empty():
            if inited:
                return None  # no new frame, use kf buffer
            else:   # blocking until get the first frame
                continue
        else:
            Buffer_data = track_to_map_IDT.get(block=False)
            break

    # else:
    #     # get data from rosbag
    #     Buffer_data = self.ros_nodes.read_rosbag(self.frame_id + 1)

    if Buffer_data is not None:
        image, depth, T, obj, bbox_dict, kf_id = Buffer_data
        del Buffer_data
        T_obj = torch.eye(4)
        sample = {"image": image, "depth": depth, "T": T, "T_obj": T_obj,
                  "obj": obj, "bbox_dict": bbox_dict, "frame_id": kf_id}

        return sample
    else:
        print("getting nothing?")
        exit(-1)
        # return None

class Replica(Dataset):
    def __init__(self,
                 root_dir,
                 traj_file=None,
                 rgb_transform=None,
                 depth_transform=None,
                 col_ext=".jpg",
                 imap_mode=False):  # todo debug use monodepth to pred
        self.imap_mode=imap_mode
        self.Ts = None
        self.To = None
        if traj_file is not None:
            self.Ts = np.loadtxt(traj_file, delimiter=" ").reshape([-1, 4, 4])
        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.col_ext = col_ext
        # things semantic-class: table-80 chair-20 sofa-76 stool-78 vase-91
        # self.sem_cls_list = [20,80,76,78,91]
        # background semantic classes: undefined--1, undefined-0 beam-5 blinds-12 curtain-30 ceiling-31 floor-40 pillar-60 vent-92 wall-93 wall-plug-95 window-97 rug-98
        self.background_cls_list = [5,12,30,31,40,60,92,93,95,97,98,79]
        # recon undefined objs
        # self.background_cls_list = [-1,0,5,12,30,40,60,93,95,97,98,79] # make ceiling and vent to be obj
        # self.background_cls_list = []        # todo debug do all objs
        # Not sure: door-37 handrail-43 lamp-47 pipe-62 rack-66 shower-stall-73 stair-77 switch-79 wall-cabinet-94 picture-59
        self.bbox_scale = 0.2  # 1 #1.5 0.9== s=1/9, s=0.2

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir, "depth")))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        bbox_dict = {}
        rgb_file = os.path.join(self.root_dir, "rgb", "rgb_" + str(idx) + ".png")
        depth_file = os.path.join(self.root_dir, "depth", "depth_" + str(idx) + ".png")
        inst_file = os.path.join(self.root_dir, "semantic_instance", "semantic_instance_" + str(idx) + ".png")
        obj_file = os.path.join(self.root_dir, "semantic_class", "semantic_class_" + str(idx) + ".png")
        depth = cv2.imread(depth_file, -1).astype(np.float32).transpose(1,0)
        image = cv2.imread(rgb_file).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(1,0,2)
        obj = cv2.imread(obj_file, cv2.IMREAD_UNCHANGED).astype(np.int32).transpose(1,0)   # uint16 -> int32
        inst = cv2.imread(inst_file, cv2.IMREAD_UNCHANGED).astype(np.int32).transpose(1,0)  # uint16 -> int32

        bbox_scale = self.bbox_scale

        obj_ = np.zeros_like(obj)
        inst_list = []
        batch_masks = []
        for inst_id in np.unique(inst):
            inst_mask = inst == inst_id
            # if np.sum(inst_mask) <= 2000: # too small    20  400
            #     continue
            sem_cls = np.unique(obj[inst_mask]) # sem label, only interested obj
            assert sem_cls.shape[0] != 0
            if sem_cls in self.background_cls_list:
                continue
            obj_mask = inst == inst_id
            batch_masks.append(obj_mask)
            inst_list.append(inst_id)
        if len(batch_masks) > 0:
            batch_masks = torch.from_numpy(np.stack(batch_masks))
            cmins, cmaxs, rmins, rmaxs = get_bbox2d_batch(batch_masks)

            for i in range(batch_masks.shape[0]):
                w = rmaxs[i] - rmins[i]
                h = cmaxs[i] - cmins[i]
                if w <= 10 or h <= 10:  # too small   todo
                    continue
                bbox_enlarged = enlarge_bbox([rmins[i], cmins[i], rmaxs[i], cmaxs[i]], scale=bbox_scale, w=obj.shape[1], h=obj.shape[0])
                # inst_list.append(inst_id)
                inst_id = inst_list[i]
                obj_[batch_masks[i]] = 1
                # bbox_dict.update({int(inst_id): torch.from_numpy(np.array(bbox_enlarged).reshape(-1))}) # batch format
                bbox_dict.update({inst_id: torch.from_numpy(np.array([bbox_enlarged[1], bbox_enlarged[3], bbox_enlarged[0], bbox_enlarged[2]]))})   # bbox order

        inst[obj_==0] = 0 # for background
        obj = inst

        # todo iMAP mode
        if self.imap_mode:
            obj = np.zeros_like(obj)

        # bbox_dict.update({0: torch.from_numpy(np.array([int(0), int(0), int(obj.shape[1]), int(obj.shape[0])]).reshape(-1))})
        bbox_dict.update(
            {0: torch.from_numpy(np.array([int(0), int(obj.shape[0]), 0, int(obj.shape[1])]))})  # bbox order

        T = None
        if self.Ts is not None:
            T = self.Ts[idx]
        # todo debug
        T_obj = np.eye(4)
        sample = {"image": image, "depth": depth, "T": T, "T_obj": T_obj,
                  "obj": obj, "bbox_dict": bbox_dict, "frame_id": idx}

        if image is None or depth is None:
            print(rgb_file)
            print(depth_file)
            raise ValueError

        # if self.rgb_transform:
        #     sample["image"] = self.rgb_transform(sample["image"])

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])

        return sample

class ScanNet(Dataset):
    def __init__(self, root_dir,traj_file=None,
                 rgb_transform=None,
                 depth_transform=None,
                 sem_transform=None,
                 gt_sem=False,
                 col_ext=None,
                 intrinsic=None,
                 imap_mode=False):
        self.imap_mode = imap_mode
        self.input_folder = root_dir
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.inst_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'instance-filt', '*.png')), key=lambda x: int(os.path.basename(x)[:-4])) # instance-filt
        self.sem_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'label-filt', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))  # label-filt
        # with open(os.path.join(self.input_folder, '/scene0024_00_vh_clean.aggregation.json')) as f:
        #     self.inst_info = json.load(f)

        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.n_img = len(self.color_paths)
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.sem_transform = sem_transform
        self.gt_sem = gt_sem
        self.edge = 10
        self.use_detector = False
        if self.use_detector:
            det_device = "cuda:1"
            config_detector = "../../detector/configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py"
            checkpoint = "../../detector/checkpoints/mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth"
            self.detector = init_detector(config_detector, checkpoint, device=det_device)

        (self.W,
        self.H,
        self.fx,
        self.fy,
        self.cx,
        self.cy) = intrinsic
        # self.intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
        #             width=self.W,
        #             height=self.H,
        #             fx=self.fx,
        #             fy=self.fy,
        #             cx=self.cx,
        #             cy=self.cy,
        #     )
        self.sem_dict = {}
        self.inst_list = []
        self.min_pixels = 1500
        # from scannetv2-labels.combined.tsv
        #1-wall, 3-floor, 16-window, 41-ceiling, 232-light switch   0-unknown? 21-pillar 161-doorframe, shower walls-128, curtain-21, windowsill-141
        self.background_cls_list = [-1, 0, 1, 3, 16, 41, 232, 21, 161, 128, 21]
        self.bbox_scale = 0.2  # 1 #1.5 0.9== s=1/9, s=0.2

        self.inst_dict = {}

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            # c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        bbox_scale = self.bbox_scale
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        inst_path = self.inst_paths[index]
        sem_path = self.sem_paths[index]
        color_data = cv2.imread(color_path)
        depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32)  # / self.png_depth_scale
        depth_data = np.nan_to_num(depth_data, nan=0.)
        T = None
        T_obj = None
        if self.poses is not None:
            T = self.poses[index]
            if np.any(np.isinf(T)):
                if index + 1 == self.__len__():
                    print("pose inf!")
                    return None
                return self.__getitem__(index + 1)
            # T_obj = self.To[idx]

        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_LINEAR)
        if self.edge:
            edge = self.edge
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        if self.depth_transform:
            depth_data = self.depth_transform(depth_data)

        if self.imap_mode:
            inst_data = np.zeros_like(depth_data).astype(np.uint8)
            bbox_dict = {0: torch.from_numpy(
                np.array([int(0), int(0), int(inst_data.shape[1]), int(inst_data.shape[0])]).reshape(-1))}
        else:
            bbox_dict = {}
            if self.use_detector:
                pass
                # # mmdet
                # results = inference_detector(self.detector, color_data)
                # bboxes, classes, masks = postprocess_mmdet_results(results, score_thr=0.9)
                # T_WC = self.poses[index]
                # # todo T_WO
                # T_CW = np.linalg.inv(T_WC)
                # inst_data = track_instance(masks, classes, depth_data, self.inst_list, self.sem_dict, self.intrinsic_open3d, T_CW, min_pixels=self.min_pixels)
                # print("sem dict ", self.sem_dict)
                # print("inst list ", len(self.inst_list))
                # # viz
                # pred = np.zeros_like(inst_data)
                # for i, mask in enumerate(masks):
                #     pred[mask] = classes[i]
                # # cv2.imshow("rgb", color_data)
                # # cv2.imshow("detect", imgviz.label2rgb(pred))
                # # cv2.imshow("merged", imgviz.label2rgb(inst_data))
                # # cv2.waitKey(1)
                # for obj_id in np.unique(inst_data):
                #     smaller_mask = inst_data == obj_id
                #     bbox2d = get_bbox2d(smaller_mask)
                #     h, w = depth_data.shape
                #     bbox2d = enlarge_bbox(bbox2d, bbox_scale, w=w, h=h)
                #     bbox_dict.update({int(obj_id): torch.from_numpy(np.array(bbox2d).reshape(-1))})  # batch format
            else:
                inst_data = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED)
                inst_data = cv2.resize(inst_data, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
                sem_data = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED)#.astype(np.int32)
                sem_data = cv2.resize(sem_data, (W, H), interpolation=cv2.INTER_NEAREST)
                if self.edge:
                    edge = self.edge
                    inst_data = inst_data[edge:-edge, edge:-edge]
                    sem_data = sem_data[edge:-edge, edge:-edge]
                inst_data += 1  # shift from 0->1 , 0 is for background

                # box filter
                track_start = time.time()
                masks = []
                classes = []
                # convert to list of arrays
                obj_ids = np.unique(inst_data)
                for obj_id in obj_ids:
                    mask = inst_data == obj_id
                    sem_cls = np.unique(sem_data[mask])
                    if sem_cls in self.background_cls_list:
                        # print("merged to bg")
                        inst_data[mask] = 0     # set to background
                        continue
                    masks.append(mask)
                    classes.append(obj_id)
                # if 0 in obj_ids:
                #     masks.append(inst_data == 0)
                #     classes.append(0)
                convert_time = time.time()
                # print("convert time ", convert_time-track_start)
                T_CW = np.linalg.inv(T)
                intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
                    width=self.W,
                    height=self.H,
                    fx=self.fx,
                    fy=self.fy,
                    cx=self.cx,
                    cy=self.cy,
                )
                intrinsic_time = time.time()
                # print("intrinsic time ", intrinsic_time-convert_time)
                inst_data = box_filter(masks, classes, depth_data, self.inst_dict, intrinsic_open3d, T_CW, min_pixels=self.min_pixels)
                filter_time = time.time()
                print("filter time ", filter_time-intrinsic_time)

                # print("self.inst_dict")
                # print(self.inst_dict.keys())
                merged_obj_ids = np.unique(inst_data)
                for obj_id in merged_obj_ids:
                    mask = inst_data == obj_id
                    bbox2d = get_bbox2d(mask, bbox_scale=bbox_scale)
                    # h, w = depth_data.shape
                    # bbox2d = enlarge_bbox(bbox2d, bbox_scale, w=w, h=h)
                    if bbox2d is None:
                        inst_data[mask] = 0 # set to bg
                    else:
                        bbox_dict.update({int(obj_id): torch.from_numpy(np.array(bbox2d).reshape(-1))})  # batch format
                bbox_time = time.time()
                # print("bbox time ", bbox_time - filter_time)
                cv2.imshow("inst", imgviz.label2rgb(inst_data))
                cv2.waitKey(1)
                # # append bg
                # if 0 in merged_obj_ids:
                #     bbox2d = get_bbox2d(inst_data == 0)
                #     h, w = depth_data.shape
                #     bbox2d = enlarge_bbox(bbox2d, bbox_scale, w=w, h=h)
                #     bbox_dict.update({0: torch.from_numpy(np.array(bbox2d).reshape(-1))})  # batch format
                print("frame {} track time {}".format(index, bbox_time-track_start))

                # for obj_id in np.unique(inst_data):
                #     obj_mask = inst_data == obj_id
                #     smaller_mask = cv2.erode(obj_mask.astype(np.uint8), np.ones((5, 5)), iterations=3).astype(bool)
                #     depth_mask = cv2.erode(depth_data, np.ones((5, 5)), iterations=3)
                #     diff_depth = np.abs(depth_mask - depth_data) > 0.20
                #     diff_mask = (obj_mask ^ smaller_mask) & diff_depth# todo mask edge & large depth diff   bbox filter
                #     if np.sum(smaller_mask) <= self.min_pixels:  # too small    20  400 # todo use sem to set background
                #         inst_data[obj_mask] = 0  # set to background
                #         continue
                #     inst_data[diff_mask] = -1 # todo -1 0
                #
                #     # merge background sems
                #     sem_cls = np.unique(sem_data[smaller_mask])
                #     assert sem_cls.shape[0] != 0
                #     if sem_cls in self.background_cls_list:
                #         inst_data[obj_mask] = 0 # set to background
                #         continue
                #
                #     bbox2d = get_bbox2d(smaller_mask)
                #     h, w = depth_data.shape
                #     bbox2d = enlarge_bbox(bbox2d, bbox_scale, w=w, h=h)
                #     bbox_dict.update({int(obj_id): torch.from_numpy(np.array(bbox2d).reshape(-1))})  # batch format
                # # update for background obj_id ==0
                # bbox2d = get_bbox2d(inst_data==0)
                # h, w = depth_data.shape
                # bbox2d = enlarge_bbox(bbox2d, bbox_scale, w=w, h=h)
                # bbox_dict.update({0: torch.from_numpy(np.array(bbox2d).reshape(-1))})  # batch format

        # wrap data to frame dict
        # todo debug
        T_obj = np.identity(4)
        # T_obj = None
        sample = {"image": color_data, "depth": depth_data, "T": T, "T_obj": T_obj}
        # assert not np.any(np.isnan(depth_data))
        # assert not np.any(np.isnan(color_data))
        if color_data is None or depth_data is None:
            print(color_path)
            print(depth_path)
            raise ValueError
        if self.gt_sem:
            if inst_data is None:
                print(inst_path)
                raise ValueError

            # sample.update({"obj": inst_data.astype(np.uint8)})
            sample.update({"obj": inst_data})
            sample.update({"bbox_dict": bbox_dict})

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.sem_transform:
            sample["obj"] = self.sem_transform(sample["obj"])

        return sample


class kinect(Dataset):
    def __init__(self,
                 root_dir,
                 traj_file=None,
                 rgb_transform=None,
                 depth_transform=None,
                 gt_sem = False,
                 sem_transform=None,
                 imap_mode=False,
                 intrinsic=None,
                 col_ext=".jpg"):
        self.imap_mode = imap_mode
        self.input_folder = root_dir
        self.color_paths = sorted(glob.glob(os.path.join(self.input_folder, 'color', '*.jpg')))
        self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, 'depth', '*.png')))
        self.n_img = len(self.color_paths)
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.sem_transform = sem_transform
        self.gt_sem = gt_sem
        self.edge = 0

    def __len__(self):
        # todo debug
        return 3000
        # return self.n_img

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color_data = cv2.imread(color_path)
        depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32)  # / self.png_depth_scale
        depth_data = np.nan_to_num(depth_data, nan=0.)
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_LINEAR)
        if self.edge:
            edge = self.edge
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        if self.depth_transform:
            depth_data = self.depth_transform(depth_data)

        # wrap data to frame dict
        T = None
        T_obj = None

        sample = {"image": color_data, "depth": depth_data, "T": T, "T_obj": T_obj}
        if color_data is None or depth_data is None:
            print(color_path)
            print(depth_path)
            raise ValueError

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.sem_transform:
            sample["obj"] = self.sem_transform(sample["obj"])

        return sample


class TUMDataset(Dataset):
    def __init__(self,
                 root_dir,
                 traj_file=None,
                 rgb_transform=None,
                 depth_transform=None,
                 col_ext=None):

        self.t_poses = None
        if traj_file is not None:
            with open(traj_file) as f:
                lines = (line for line in f if not line.startswith('#'))
                self.t_poses = np.loadtxt(lines, delimiter=' ')

        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

        self.associations_file = root_dir + "associations.txt"
        with open(self.associations_file) as f:
            timestamps, self.rgb_files, self.depth_files = zip(
                *[(float(line.rstrip().split()[0]),
                    line.rstrip().split()[1],
                    line.rstrip().split()[3]) for line in f])

            self.timestamps = np.array(timestamps)

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        depth_file = self.root_dir + self.depth_files[idx]
        rgb_file = self.root_dir + self.rgb_files[idx]

        depth = cv2.imread(depth_file, -1)
        image = cv2.imread(rgb_file)

        T = None
        if self.t_poses is not None:
            rgb_timestamp = self.timestamps[idx]
            timestamp_distance = np.abs(rgb_timestamp - self.t_poses[:, 0])
            gt_idx = timestamp_distance.argmin()
            quat = self.t_poses[gt_idx][4:]
            trans = self.t_poses[gt_idx][1:4]

            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans

        sample = {"image": image, "depth": depth, "T": T}

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])
        print("max depth ", np.max(sample["depth"]))
        return sample


class dyn_replica(Dataset):
    def __init__(self,
                 root_dir,
                 info_file=None,
                 rgb_transform=None,
                 depth_transform=None,
                 gt_sem=True,
                 sem_transform=None,
                 col_ext=".jpg",
                 intrinsic=None,
                 imap_mode=False
                 ):
        self.imap_mode = imap_mode
        # self.Ts = None  # Ts for cam pose
        # self.To = None  # To for obj pose

        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.gt_sem = gt_sem
        self.sem_transform = sem_transform
        self.col_ext = col_ext
        self.file_name_list = os.listdir(f'{self.root_dir}/hdf5_data/')
        # self.file_name_list.sort()
        assert len(self.file_name_list) != 0

        self.bbox_scale = 0.2

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, idx):
        bbox_scale = self.bbox_scale
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = f'{self.root_dir}/hdf5_data/{idx}.hdf5'
        with h5py.File(path, 'r') as data:
            # print(data.keys())
            color = data['colors'][()]
            depth = data['depth'][()]
            # instance = data['instance_segmaps'][()]
            segmaps = data['category_id_segmaps'][()]

        # test camera pose
        cam_pose_file = path.replace(f'hdf5_data/{idx}.hdf5', f'cam_poses/{idx}.npy')
        # print(cam_pose_file)
        # cam_pose_file = '/home/asrl/code/dataset_tool/dyn_sim/output/cam_poses/0.npy'
        if os.path.exists(cam_pose_file) and os.path.isfile(cam_pose_file):
            T_wc = np.load(cam_pose_file)

        # test object pose
        T_obj = np.eye(4)[None, :, :].repeat(100, axis=0)
        # T_obj_prev = T_obj.copy()
        obj_ids = np.unique(segmaps)
        for obj_id in obj_ids:
            obj_pose_file = path.replace(f'hdf5_data/{idx}.hdf5', f'objects_traj/chair_{obj_id}_traj.npy')
            # print(obj_pose_file)
            if os.path.exists(obj_pose_file) and os.path.isfile(obj_pose_file):
                T_wo_traj = np.load(obj_pose_file)
                T_wo = T_wo_traj[idx]
                T_obj[obj_id, :, :] = T_wo
            # print("path ", path)
            # obj_pose_file_prev = path.replace(f'hdf5_data/{idx}.hdf5', f'objects_traj/chair_{obj_id}_traj.npy')
            # print(obj_pose_file_prev)
            # if idx > 0 and os.path.exists(obj_pose_file_prev) and os.path.isfile(obj_pose_file_prev):
            #     T_wo_traj_prev = np.load(obj_pose_file_prev)
            #     T_wo_prev = T_wo_traj_prev[0]
            #     T_obj_prev[obj_id, :, :] = T_wo_prev

        # # convert Tobj to absolute Tobj
        # T_obj = T_obj @ np.linalg.inv(T_obj_prev)


        bbox_dict = {}
        obj = segmaps
        if self.imap_mode:
            obj = np.zeros_like(obj)

        for obj_id in np.unique(obj):
            mask = obj == obj_id
            bbox2d = get_bbox2d(mask, bbox_scale)
            if bbox2d is None:
                obj[mask] = 0  # set to bg
            else:
                bbox_dict.update({int(obj_id): torch.from_numpy(np.array(bbox2d).reshape(-1))})  # batch format
        # bbox_dict.update(
        #     {0: torch.from_numpy(np.array([int(0), int(0), int(obj.shape[1]), int(obj.shape[0])]).reshape(-1))})

        sample = {"image": color, "depth": depth, "T": T_wc, "T_obj": T_obj}

        if self.gt_sem:
            if obj is None:
                print(path)
                raise ValueError
            sample.update({"obj": obj})
            sample.update({"bbox_dict": bbox_dict})

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])

        if self.sem_transform:
            sample["obj"] = self.sem_transform(sample["obj"])

        return sample

def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'Y' not in header['channels'] else channelData['Y']

    return Y


class CoFusion(Dataset):
    def __init__(self,
                 root_dir,
                 info_file=None,
                 rgb_transform=None,
                 depth_transform=None,
                 gt_sem = False,
                 sem_transform=None,
                 col_ext=".jpg",
                 intrinsic=None,
                 imap_mode=False
                 ):
        self.imap_mode = imap_mode
        self.Ts = None  # Ts for cam pose
        self.To = None  # To for obj pose

        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.gt_sem = gt_sem
        self.sem_transform = sem_transform
        self.col_ext = col_ext

        # self.file_name_list = os.listdir(os.path.join(self.root_dir, "rgb"))
        # self.file_name_list.sort()

        self.color_paths = sorted(glob.glob(os.path.join(self.root_dir, 'colour', '*.png')))
        self.depth_paths = sorted(glob.glob(os.path.join(self.root_dir, 'depth_noise', '*.exr')))
        self.inst_paths = sorted(glob.glob(os.path.join(self.root_dir, 'mask_id', '*.png')))
        self.traj_path = os.path.join(self.root_dir, "trajectories")
        self.Ts = np.loadtxt(os.path.join(self.traj_path, "gt-cam-0.txt"))[:, 1:].reshape(-1,7) # T quad
        T_quad = np.zeros_like(self.Ts)
        T_quad[:, :4] = self.Ts[:, 3:]
        T_quad[:, 4:] = self.Ts[:, :3]
        self.Ts = xyzqtoT(T_quad)
        # self.Ts = xyzqtoT(self.Ts)
        print("Ts ", self.Ts)
        self.To_car = np.loadtxt(os.path.join(self.traj_path, "gt-car-2.txt"))[:, 1:].reshape(-1,7)
        self.To_horse = np.loadtxt(os.path.join(self.traj_path, "gt-poses-horse-3.txt"))[:, 1:].reshape(-1, 7)
        self.To_ship = np.loadtxt(os.path.join(self.traj_path, "gt-ship-1.txt"))[:, 1:].reshape(-1, 7)

        self.bbox_scale = 0.2

    def __len__(self):
        # return self.Ts.shape[0]
        return len(self.color_paths)

    def __getitem__(self, idx):
        bbox_scale = self.bbox_scale
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # s = f"{idx:06}"  # int variable
        rgb_file = self.color_paths[idx]
        depth_file = self.depth_paths[idx]
        obj_file = self.inst_paths[idx]

        # rgb_file = os.path.join(self.root_dir, "rgb", self.file_name_list[idx])
        # depth_file = os.path.join(self.root_dir, "depth", self.file_name_list[idx])
        # obj_file = os.path.join(self.root_dir, "mask_RCNN/mask", self.file_name_list[idx][1:-4] + ".npy")
        # id_file = os.path.join(self.root_dir, "mask_RCNN/class_id", self.file_name_list[idx][1:-4] + ".npy")
        # id_raw = np.load(id_file)
        # depth = cv2.imread(depth_file, -1)

        depth = readEXR_onlydepth(depth_file)
        print("depth ", depth.max())
        image = cv2.imread(rgb_file)
        obj = cv2.imread(obj_file, cv2.IMREAD_UNCHANGED).mean(-1).astype(np.int32)
        # obj = np.load(obj_file) # obj_num x w x h
        print("obj ", obj)
        # obj_num = obj_raw.shape[0]
        # obj = np.zeros(depth.shape).astype(np.int32)
        # for k in range(obj_num):
        #     obj[obj_raw[k] == 1] = id_raw[k]
        # mask_invalid_inds = depth == 0
        # obj[mask_invalid_inds] = 0
        T = None
        if self.Ts is not None:
            T = self.Ts[idx]
        else:
            T = np.eye(4)
        T_obj = None
        if self.To is not None:
            T_obj = self.To[idx]  # contain diffrent obj poses for idx frame
        else:
            T_obj = np.eye(4)[None,:,:].repeat(100, axis=0)

        bbox_dict = {}
        if self.imap_mode:
            obj = np.zeros_like(obj)

        for obj_id in np.unique(obj):
            smaller_mask = obj == obj_id
            bbox2d = get_bbox2d(smaller_mask)
            h, w = depth.shape
            bbox2d = enlarge_bbox(bbox2d, bbox_scale, w=w, h=h)
            bbox_dict.update({int(obj_id): torch.from_numpy(np.array(bbox2d).reshape(-1))})  # batch format
        bbox_dict.update(
            {0: torch.from_numpy(np.array([int(0), int(0), int(obj.shape[1]), int(obj.shape[0])]).reshape(-1))})

        sample = {"image": image, "depth": depth, "T": T, "T_obj": T_obj}
        if image is None or depth is None:
            print(rgb_file)
            print(depth_file)
            raise ValueError
        if self.gt_sem:
            if obj is None:
                print(obj_file)
                raise ValueError
            sample.update({"obj": obj})
            sample.update({"bbox_dict": bbox_dict})

        if self.rgb_transform:
            sample["image"] = self.rgb_transform(sample["image"])

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])

        if self.sem_transform:
            sample["obj"] = self.sem_transform(sample["obj"])
        # # todo debug
        # print("sample ", sample)
        # print(sample["depth"].max())
        return sample



