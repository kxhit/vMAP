import imgviz
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import cv2
import os
from utils import enlarge_bbox, get_bbox2d, get_bbox2d_batch, box_filter
import glob
from torchvision import transforms
import image_transforms
import open3d
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

def init_loader(cfg, multi_worker=True):
    if cfg.dataset_format == "Replica":
        dataset = Replica(cfg)
    elif cfg.dataset_format == "ScanNet":
        dataset = ScanNet(cfg)
    else:
        print("Dataset format {} not found".format(cfg.dataset_format))
        exit(-1)

    # init dataloader
    if multi_worker:
        # multi worker loader
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, sampler=None,
                                batch_sampler=None, num_workers=4, collate_fn=None,
                                pin_memory=True, drop_last=False, timeout=0,
                                worker_init_fn=None, generator=None, prefetch_factor=2,
                                persistent_workers=True)
    else:
        # single worker loader
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, sampler=None,
                                     batch_sampler=None, num_workers=0)

    return dataloader

class Replica(Dataset):
    def __init__(self, cfg):
        self.imap_mode = cfg.imap_mode
        self.root_dir = cfg.dataset_dir
        traj_file = os.path.join(self.root_dir, "traj_w_c.txt")
        self.Twc = np.loadtxt(traj_file, delimiter=" ").reshape([-1, 4, 4])
        self.depth_transform = transforms.Compose(
            [image_transforms.DepthScale(cfg.depth_scale),
             image_transforms.DepthFilter(cfg.max_depth)])

        # background semantic classes: undefined--1, undefined-0 beam-5 blinds-12 curtain-30 ceiling-31 floor-40 pillar-60 vent-92 wall-93 wall-plug-95 window-97 rug-98
        self.background_cls_list = [5,12,30,31,40,60,92,93,95,97,98,79]
        # Not sure: door-37 handrail-43 lamp-47 pipe-62 rack-66 shower-stall-73 stair-77 switch-79 wall-cabinet-94 picture-59
        self.bbox_scale = 0.2  # 1 #1.5 0.9== s=1/9, s=0.2

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir, "depth")))

    def __getitem__(self, idx):
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

        if self.imap_mode:
            obj = np.zeros_like(obj)
        else:
            obj_ = np.zeros_like(obj)
            inst_list = []
            batch_masks = []
            for inst_id in np.unique(inst):
                inst_mask = inst == inst_id
                # if np.sum(inst_mask) <= 2000: # too small    20  400
                #     continue
                sem_cls = np.unique(obj[inst_mask])  # sem label, only interested obj
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
                    bbox_enlarged = enlarge_bbox([rmins[i], cmins[i], rmaxs[i], cmaxs[i]], scale=bbox_scale,
                                                 w=obj.shape[1], h=obj.shape[0])
                    # inst_list.append(inst_id)
                    inst_id = inst_list[i]
                    obj_[batch_masks[i]] = 1
                    # bbox_dict.update({int(inst_id): torch.from_numpy(np.array(bbox_enlarged).reshape(-1))}) # batch format
                    bbox_dict.update({inst_id: torch.from_numpy(np.array(
                        [bbox_enlarged[1], bbox_enlarged[3], bbox_enlarged[0], bbox_enlarged[2]]))})  # bbox order

            inst[obj_ == 0] = 0  # for background
            obj = inst

        bbox_dict.update({0: torch.from_numpy(np.array([int(0), int(obj.shape[0]), 0, int(obj.shape[1])]))})  # bbox order

        T = self.Twc[idx]   # could change to ORB-SLAM pose or else
        T_obj = np.eye(4)   # obj pose, if dynamic
        sample = {"image": image, "depth": depth, "T": T, "T_obj": T_obj,
                  "obj": obj, "bbox_dict": bbox_dict, "frame_id": idx}

        if image is None or depth is None:
            print(rgb_file)
            print(depth_file)
            raise ValueError

        if self.depth_transform:
            sample["depth"] = self.depth_transform(sample["depth"])

        return sample

class ScanNet(Dataset):
    def __init__(self, cfg):
        self.imap_mode = cfg.imap_mode
        self.root_dir = cfg.dataset_dir
        self.color_paths = sorted(glob.glob(os.path.join(
            self.root_dir, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.root_dir, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.inst_paths = sorted(glob.glob(os.path.join(
            self.root_dir, 'instance-filt', '*.png')), key=lambda x: int(os.path.basename(x)[:-4])) # instance-filt
        self.sem_paths = sorted(glob.glob(os.path.join(
            self.root_dir, 'label-filt', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))  # label-filt
        self.load_poses(os.path.join(self.root_dir, 'pose'))
        self.n_img = len(self.color_paths)
        self.depth_transform = transforms.Compose(
            [image_transforms.DepthScale(cfg.depth_scale),
             image_transforms.DepthFilter(cfg.max_depth)])
        # self.rgb_transform = rgb_transform
        self.W = cfg.W
        self.H = cfg.H
        self.fx = cfg.fx
        self.fy = cfg.fy
        self.cx = cfg.cx
        self.cy = cfg.cy
        self.edge = cfg.mw
        self.intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
            width=self.W,
            height=self.H,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        )

        self.min_pixels = 1500
        # from scannetv2-labels.combined.tsv
        #1-wall, 3-floor, 16-window, 41-ceiling, 232-light switch   0-unknown? 21-pillar 161-doorframe, shower walls-128, curtain-21, windowsill-141
        self.background_cls_list = [-1, 0, 1, 3, 16, 41, 232, 21, 161, 128, 21]
        self.bbox_scale = 0.2
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
            self.poses.append(c2w)

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        bbox_scale = self.bbox_scale
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        inst_path = self.inst_paths[index]
        sem_path = self.sem_paths[index]
        color_data = cv2.imread(color_path).astype(np.uint8)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_data = np.nan_to_num(depth_data, nan=0.)
        T = None
        if self.poses is not None:
            T = self.poses[index]
            if np.any(np.isinf(T)):
                if index + 1 == self.__len__():
                    print("pose inf!")
                    return None
                return self.__getitem__(index + 1)

        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_LINEAR)
        if self.edge:
            edge = self.edge # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        if self.depth_transform:
            depth_data = self.depth_transform(depth_data)
        bbox_dict = {}
        if self.imap_mode:
            inst_data = np.zeros_like(depth_data).astype(np.int32)
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
                    inst_data[mask] = 0     # set to background
                    continue
                masks.append(mask)
                classes.append(obj_id)
            T_CW = np.linalg.inv(T)
            inst_data = box_filter(masks, classes, depth_data, self.inst_dict, self.intrinsic_open3d, T_CW, min_pixels=self.min_pixels)

            merged_obj_ids = np.unique(inst_data)
            for obj_id in merged_obj_ids:
                mask = inst_data == obj_id
                bbox2d = get_bbox2d(mask, bbox_scale=bbox_scale)
                if bbox2d is None:
                    inst_data[mask] = 0 # set to bg
                else:
                    min_x, min_y, max_x, max_y = bbox2d
                    bbox_dict.update({int(obj_id): torch.from_numpy(np.array([min_x, max_x, min_y, max_y]).reshape(-1))})  # batch format
            bbox_time = time.time()
            # print("bbox time ", bbox_time - filter_time)
            cv2.imshow("inst", imgviz.label2rgb(inst_data))
            cv2.waitKey(1)
            print("frame {} track time {}".format(index, bbox_time-track_start))

        bbox_dict.update({0: torch.from_numpy(np.array([int(0), int(inst_data.shape[1]), 0, int(inst_data.shape[0])]))})  # bbox order
        # wrap data to frame dict
        T_obj = np.identity(4)
        sample = {"image": color_data.transpose(1,0,2), "depth": depth_data.transpose(1,0), "T": T, "T_obj": T_obj}
        if color_data is None or depth_data is None:
            print(color_path)
            print(depth_path)
            raise ValueError

        sample.update({"obj": inst_data.transpose(1,0)})
        sample.update({"bbox_dict": bbox_dict})
        return sample
