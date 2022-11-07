import os
import queue
import time

import cv2
import torch
from time import perf_counter
import numpy as np
from scipy.spatial.transform import Rotation

import rospy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
# from orb_slam3_ros_wrapper.msg import map_frame, keyframes
from orb_slam3_ros_wrapper.msg import Frame, PoseWithIDArray
import copy
import sys
sys.path.append('./detector/detectron2/')
from detectron2.data import MetadataCatalog
from detector.detectron2.demo.predictor import Detector
sys.path.append('../../detector/Detic/')
from detic.modeling.utils import reset_cls_test
# import mmcv
# from mmdet.apis import inference_detector, init_detector
import open3d
import imgviz
from utils import track_instance, get_bbox2d


class Tracking:
    def __init__(self, config, track_to_map_que, track_to_vis_que, kfs_que=None, imap_mode=False, do_thread=False, device="cuda:1",
                 config_detector="../../detector/configs/mask2former/mask2former_r50_lsj_8x2_50e_coco-panoptic.py",
                 checkpoint="../../detector/checkpoints/mask2former_r50_lsj_8x2_50e_coco-panoptic_20220326_224516-11a44721.pth",
                 ) -> None:
        print("track: starting", os.getpid())
        self.data_device = "cuda:0"
        self.imap_mode = imap_mode
        self.config = config
        self.detector = Detector(None)    # detectron2
        vocabulary = 'lvis'
        BUILDIN_CLASSIFIER = {
         'lvis': './detector/Detic/datasets/metadata/lvis_v1_clip_a+cname.npy',
         'objects365': './detector/Detic/datasets/metadata/o365_clip_a+cnamefix.npy',
         'openimages': './detector/Detic/datasets/metadata/oid_clip_a+cname.npy',
         'coco': './detector/Detic/datasets/metadata/coco_clip_a+cname.npy',
        }
        BUILDIN_METADATA_PATH = {
         'lvis': 'lvis_v1_val',
         'objects365': 'objects365_v2_val',
         'openimages': 'oid_val_expanded',
         'coco': 'coco_2017_val',
        }
        metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
        classifier = BUILDIN_CLASSIFIER[vocabulary]
        num_classes = len(metadata.thing_classes)
        print("num_classes ", num_classes)
        reset_cls_test(self.detector.predictor.model, classifier, num_classes)
        self.class_names = metadata.get("thing_classes", None)
        self.clip_features = torch.load("./detector/Detic/lvis_clip_dict.pt")

        self.score_thr = 0.8
        self.map_que = track_to_map_que
        # self.vis_que = track_to_vis_que
        self.kfs_que = kfs_que

        self.sem_dict = {}
        self.inst_list = []

        self.max_depth = self.config["render"]["depth_range"][1]
        self.min_depth = self.config["render"]["depth_range"][0]
        self.mh = self.config["camera"]["mh"]
        self.mw = self.config["camera"]["mw"]
        self.height = self.config["camera"]["h"]
        self.width = self.config["camera"]["w"]
        self.H = self.height - 2 * self.mh
        self.W = self.width - 2 * self.mw
        self.fx = self.config["camera"]["fx"]
        self.fy = self.config["camera"]["fy"]
        self.cx = self.config["camera"]["cx"] - self.mw
        self.cy = self.config["camera"]["cy"] - self.mh

        w = self.config["camera"]["w"]
        h = self.config["camera"]["h"]
        fx = self.config["camera"]["fx"]
        fy = self.config["camera"]["fy"]
        cx = self.config["camera"]["cx"]
        cy = self.config["camera"]["cy"]
        distortion_array = None
        if "distortion" in self.config["camera"]:
            distortion_array = np.array(self.config["camera"]["distortion"])
        elif "k1" in self.config["camera"]:
            k1 = self.config["camera"]["k1"]
            k2 = self.config["camera"]["k2"]
            k3 = self.config["camera"]["k3"]
            k4 = self.config["camera"]["k4"]
            k5 = self.config["camera"]["k5"]
            k6 = self.config["camera"]["k6"]
            p1 = self.config["camera"]["p1"]
            p2 = self.config["camera"]["p2"]
            distortion_array = np.array([k1, k2, p1, p2, k3, k4, k5, k6])
        else:
            print("no distortion")
        if distortion_array is not None:
            K = np.array([[fx, 0., cx],
                   [0., fy, cy],
                   [0., 0., 1.]]).astype(np.float32)
            K_new = np.copy(K)
            self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                                            K,
                                            distortion_array,    # np.array([k1, k2, p1, p2, k3, k4, k5, k6]), np.array([k1, k2, k3, k4, k5, k6]),
                                            np.eye(3),
                                            K_new,
                                            (w, h),
                                            cv2.CV_32FC1)
        else:
            self.map1x = None

        self.intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
                width=self.W,
                height=self.H,
                fx=self.fx,
                fy=self.fy,
                cx=self.cx,
                cy=self.cy,
            )
        self.min_pixels = 200

        self.inst_color_map = imgviz.label_colormap(n_label=2000)
        self.cv_bridge = CvBridge()
        self.bbox_scale = 0.2
        self.prev_kf_id = None

        if do_thread:
            print("do thread")
            rospy.init_node("imap", anonymous=True)
            # self.sub_topic = "latest_frame"
            self.sub_topic = "latest_keyframe"
            rospy.Subscriber(self.sub_topic, Frame, self.one_kf_callback) # latest keyframe
            # rospy.Subscriber(sub_topic, Frame, self.one_kf_callback) # latest frame for vis
            rospy.Subscriber("/keyframe_poses", PoseWithIDArray, self.kf_callback) # only randomly choose one kf from the list
            print("subscribing ros topic")
            rospy.spin()

            self.map_que.put("finish")
            print("Tracking process finished")
        else:   # read saved rosbag
            import rosbag
            bag_file = "/home/xin/kinect_keyframes.bag"    # todo
            print("read from rosbag ", bag_file)
            bag = rosbag.Bag(bag_file)
            self.kfs = []
            for _, keyframe, t in bag.read_messages(topics=["keyframes"]):
                self.kfs.append(keyframe)
            bag.close()
            self.kf_size = len(self.kfs)


    def one_kf_callback(self, msg):
        # if self.map_que.full() and self.vis_que.full():
        #     return
        ros_time = time.time()
        kf_id = msg.id
        if kf_id == self.prev_kf_id:
            return
        else:
            self.prev_kf_id = kf_id
        rgb_np = self.cv_bridge.imgmsg_to_cv2(msg.rgb, "rgb8")   # rgb8
        depth_np = self.cv_bridge.imgmsg_to_cv2(msg.depth, "passthrough")
        depth_np = np.nan_to_num(depth_np, nan=0.0)

        # Formatting the estimated camera pose as a euclidean transformation matrix w.r.t world frame
        quat = msg.pose.orientation
        rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()

        pose = msg.pose.position    # T_wc is from latest_kf, T_cw is latest frame
        trans = np.asarray([[pose.x], [pose.y], [pose.z]])

        camera_transform = np.concatenate((rot, trans), axis=1)
        camera_transform = np.vstack((camera_transform, [0.0, 0.0, 0.0, 1.0]))
        if self.sub_topic == "latest_frame":
            camera_transform = np.linalg.inv(camera_transform)    # no inv for kfs
        camera_transform = camera_transform[None, ...]

        # Crop images to remove the black edges after calibration
        w = rgb_np.shape[1]
        h = rgb_np.shape[0]

        # undistort
        if self.map1x is not None:
            # depth_np = cv2.remap(depth_np, self.map1x, self.map1y, cv2.INTER_NEAREST) # todo only undistort rgb
            rgb_np = cv2.remap(rgb_np, self.map1x, self.map1y, cv2.INTER_LINEAR)

        rgb_np = rgb_np[self.mh: (h - self.mh), self.mw: (w - self.mw)]
        rgb_np = rgb_np[None, ...]

        depth_np = depth_np[self.mh: (h - self.mh), self.mw: (w - self.mw)]
        depth_np = depth_np[None, ...].astype(np.float32)

        if self.imap_mode:
            obj_np = np.zeros_like(depth_np, dtype=np.int)
            bbox_dict = {0: torch.from_numpy(np.array([int(0), int(0), depth_np.shape[1], depth_np.shape[0]]).reshape(1, -1))}
            depth_np[(depth_np > self.max_depth) | (depth_np < self.min_depth)] = 0
        else:
            # init label to background 0
            obj_np = np.zeros(depth_np.shape, dtype=np.int)
            bbox_dict = {}

            frame = rgb_np[0]
            detect_time = time.time()

            # detector
            # print("input shape ", input_frame.shape)
            with torch.cuda.amp.autocast():
                results = self.detector.detect(frame)["instances"].to("cpu")
            classes = results.pred_classes.tolist()
            masks = list(np.asarray(results.pred_masks))
            print("detect time ", time.time() - detect_time)

            depth_np[(depth_np > self.max_depth) | (depth_np < self.min_depth)] = 0

            track_instance_time = time.time()
            T_CW = np.linalg.inv(camera_transform[0])
            inst_data = track_instance(masks, classes, depth_np[0], self.inst_list, self.sem_dict, self.intrinsic_open3d,
                                       T_CW, voxel_size=0.01, min_pixels=self.min_pixels, erode=False,
                                       clip_features=self.clip_features,
                                       class_names=self.class_names)
            print("track instance time ", time.time()-track_instance_time)
            # print("self.sem_dict ", self.sem_dict)
            for obj_id in np.unique(inst_data):
                mask = inst_data == obj_id
                bbox2d = get_bbox2d(mask, bbox_scale=self.bbox_scale)
                if bbox2d is None:
                    inst_data[mask] = 0  # set to bg
                    continue
                # bbox_dict.update({int(obj_id): torch.from_numpy(np.array(bbox2d).reshape(1, -1))})  # batch format
                bbox_dict.update({int(obj_id): torch.from_numpy(np.array([bbox2d[0], bbox2d[2], bbox2d[1], bbox2d[3]]))})   # bbox order
            obj_np[0] = inst_data

            # viz detection
            # frame = mmcv.bgr2rgb(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            alpha = 0.6
            color = None
            if color is None:
                random_colors = np.random.randint(0, 255, (len(masks), 3))
                color = [tuple(c) for c in random_colors]
                color = np.array(color, dtype=np.uint8)
            for i, mask in enumerate(masks):
                color_mask = color[i]
                frame[mask] = frame[mask] * (1 - alpha) + color_mask * alpha
            cv2.imshow("detection", frame)
            cv2.imshow("merged", imgviz.label2rgb(inst_data, colormap=self.inst_color_map))
            cv2.waitKey(1)

        # send data to mapping -------------------------------------------
        rgb = torch.from_numpy(rgb_np[0]).permute(1,0,2)
        depth = torch.from_numpy(depth_np[0]).permute(1,0)
        twc = torch.from_numpy(camera_transform[0])
        inst = torch.from_numpy(obj_np[0]).permute(1,0)
        try:
            self.map_que.put((rgb, depth, twc, inst, bbox_dict.copy(), kf_id), block=False)
            print("`````````````````````````````````````````")
            print("ros kf id ", kf_id)
        except queue.Full:
            print("ros thread queue FULL")
            pass

        # # send pose to vis -----------------------------------------------
        # try:
        #     if self.use_detector:
        #         self.vis_que.put((camera_transform.copy(), depth_np.copy(), rgb_np.copy(), obj_np.copy()), block=False)
        #     else:
        #         self.vis_que.put((camera_transform.copy(), depth_np.copy(), rgb_np.copy()), block=False)
        # except queue.Full:
        #     pass

        del rgb_np
        del depth_np
        del camera_transform
        del obj_np

        print("ros time ", time.time()-ros_time)

    def kf_callback(self, msg):
        kf_update_dict = {}
        kf_ids = []
        poses_np = []
        # print("msg ", msg)
        for i, pose_with_id in enumerate(msg.pose):
            kf_id = pose_with_id.id
            # kf_ids.append(kf_id)
            # Formatting the estimated camera pose as a euclidean transformation matrix w.r.t world frame
            quat = pose_with_id.pose.orientation
            rot = Rotation.from_quat([quat.x, quat.y, quat.z,
                                      quat.w]).as_matrix()

            pose = pose_with_id.pose.position
            trans = np.asarray([[pose.x], [pose.y], [pose.z]])

            Twc = np.concatenate((rot, trans), axis=1)
            Twc = np.vstack((Twc, [0.0, 0.0, 0.0, 1.0]))
            # poses_np.append(Twc)

            kf_update_dict.update({kf_id: torch.from_numpy(Twc).to(self.data_device).float()})

        # kf_ids = np.stack(kf_ids)
        # poses_np = np.stack(poses_np)
        # send data to mapping -------------------------------------------
        try:
            self.kfs_que.put((kf_update_dict.copy(),), block=False) # todo make sure get the newest
        except queue.Full:
            pass



