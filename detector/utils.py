import time

import numpy as np
import open3d
from ScenePriors.train.utils import project_to_image
import cv2
import imgviz
import torch
import mmcv
# from pytorch3d.ops import box3d_overlap
from ScenePriors.train.utils import enlarge_bbox

def get_bbox2d(obj_mask, bbox_scale=1.0):
    contours, hierarchy = cv2.findContours(obj_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
                          -2:]
    # # Find the index of the largest contour
    # areas = [cv2.contourArea(c) for c in contours]
    # max_index = np.argmax(areas)
    # cnt = contours[max_index]
    # Concatenate all contours
    cnt = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(cnt)  # todo if multiple contours, choose the outmost one?
    # x, y, w, h = cv2.boundingRect(contours)
    bbox_enlarged = enlarge_bbox([x, y, x + w, y + h], scale=bbox_scale, w=obj_mask.shape[1], h=obj_mask.shape[0])
    return bbox_enlarged

# #get batched bounds for pytorch
# def get_bbox2d_batch(img):
#     b,h,w = img.shape[:3]
#     rows = np.any(img, axis=2)
#     cols = np.any(img, axis=1)
#     rmins = np.argmax(rows, dim=1)
#     rmaxs = h - np.argmax(rows.flip(dims=[1]), dim=1)
#     cmins = np.argmax(cols, dim=1)
#     cmaxs = w - np.argmax(cols.flip(dims=[1]), dim=1)
#
#     return rmins, rmaxs, cmins, cmaxs

def get_bbox2d_batch(img):
    b,h,w = img.shape[:3]
    rows = torch.any(img, axis=2)
    cols = torch.any(img, axis=1)
    rmins = torch.argmax(rows.float(), dim=1)
    rmaxs = h - torch.argmax(rows.float().flip(dims=[1]), dim=1)
    cmins = torch.argmax(cols.float(), dim=1)
    cmaxs = w - torch.argmax(cols.float().flip(dims=[1]), dim=1)

    return rmins, rmaxs, cmins, cmaxs

def refine_mask(obj_mask, depth, intrinsic):
    obj_depth = np.copy(depth)
    invalid_depth_mask = depth == 0
    obj_depth[~obj_mask] = 0
    # print(obj_depth)
    # voxel_size = 0.01

    pc = open3d.geometry.PointCloud.create_from_depth_image(depth=open3d.geometry.Image(obj_depth),
                                                                   intrinsic=intrinsic,
                                                                    depth_trunc=3.0,
                                                                   depth_scale=1.0)
    # pc.voxel_down_sample(voxel_size)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=500, std_ratio=0.01)  # TODO hack
    # pc, _ = pc.remove_radius_outlier(nb_points=100, radius=0.03)
    # print("pc ", np.array(pc.points))
    P = intrinsic.intrinsic_matrix @ np.identity(4)[:3, :]
    uv, _ = project_to_image(np.array(pc.points), P)    # Nx2
    u = uv[:, 0].reshape(-1, 1)
    v = uv[:, 1].reshape(-1, 1)
    vu = np.concatenate([v, u], axis=-1)
    # print(uv)
    # inter_mask = obj_mask[tuple(vu.T)] == 1 # intersection between obj_mask & depth_mask
    depth_mask = np.zeros_like(obj_mask, dtype=bool)
    depth_mask[tuple(vu.T)] = True # valid depth uv
    depth_mask[invalid_depth_mask & obj_mask] = True # should keep invalid depth inside obj mask for color supervision

    return depth_mask

def grab_cut(obj_mask, depth):
    mask = np.copy(obj_mask).astype(np.uint8)
    mask[mask > 0] = cv2.GC_PR_FGD
    mask[mask == 0] = cv2.GC_BGD
    img = imgviz.depth2rgb(depth, min_value=0.25, max_value=1.6)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    smaller_mask, bgdModel, fgdModel = cv2.grabCut(img, mask, rect=None, bgdModel=bgdModel, fgdModel=fgdModel,
                                                  iterCount=1, mode=cv2.GC_INIT_WITH_MASK)
    refined_obj_mask = smaller_mask == cv2.GC_PR_FGD

    return refined_obj_mask

# def depth_filter(obj_mask, depth, thresh=180):
#     obj_depth = np.copy(depth)
#     obj_depth[~obj_mask] = 0
#     scale = 1
#     delta = 0
#     ddepth = cv2.CV_16S
#     grad_x = cv2.Sobel(obj_depth, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
#     grad_y = cv2.Sobel(obj_depth, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
#     abs_grad_x = cv2.convertScaleAbs(grad_x)
#     abs_grad_y = cv2.convertScaleAbs(grad_y)
#     grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
#     refined_obj_mask = np.copy(obj_mask)
#     refined_obj_mask[grad>thresh] = 0
#     return refined_obj_mask

# def depth_filter(obj_mask, depth, thresh=0.03):
#     obj_depth = np.copy(depth)
#     obj_depth[~obj_mask] = 0
#     inner_mask = cv2.erode(obj_mask.astype(np.uint8), np.ones((15, 15)), iterations=3).astype(bool)
#     intersect_mask = inner_mask ^ obj_mask
#     mean_depth = np.mean(obj_depth[inner_mask])  # todo cmp with nearest depths rather than the whole?
#     noise_mask = obj_depth[intersect_mask] - mean_depth > thresh # further than mean depth + thresh is noise
#     intersect_mask[np.where(intersect_mask)] = noise_mask
#     refined_obj_mask = obj_mask ^ intersect_mask
#
#     return refined_obj_mask

# def reject_outliers(data, m=6.):
#     indices = abs(data - np.mean(data)) < m * np.std(data)
#     return data[indices], indices

# def reject_outliers(data, m = 2.):
#     d = np.abs(data - np.median(data))
#     mdev = np.median(d)
#     s = d/mdev if mdev else 0.
#     indices = s<m
#     return data[indices], indices

def reject_outliers(data, percent=0.50):
    d = np.copy(data).reshape(-1)
    indices = np.argsort(d.reshape(-1))[:int(len(d) * percent)]
    d = np.zeros_like(data)
    d[indices] = 1
    return None, d

def depth_filter(obj_mask, depth):
    obj_depth = depth[obj_mask]
    _, indices = reject_outliers(obj_depth)
    refined_obj_mask = np.copy(obj_mask)
    refined_obj_mask[np.where(obj_mask)] = indices
    return refined_obj_mask

class InstData:
    def __init__(self):
        super(InstData, self).__init__()
        self.bbox3D = None
        self.inst_id = None     # instance
        self.class_id = None    # semantic
        self.pc_sample = None
        self.merge_cnt = 0  # merge times counting
        self.cmp_cnt = 0
        self.tsdf = None
        # self.tsdf = open3d.pipelines.integration.ScalableTSDFVolume(
        #         voxel_length=4. / 256,  # 4.
        #         sdf_trunc=0.04,
        #         color_type=open3d.pipelines.integration.TSDFVolumeColorType.Gray32,   # RGB8
        #     )
    def integrate_tsdf(self, depth_o3d, inst_mask, intrinsic_open3d, T_CW):
        obj_mask = np.zeros_like(inst_mask, dtype=np.float32)
        obj_mask[inst_mask] = 1.
        obj_o3d = open3d.geometry.Image(obj_mask)
        objd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(obj_o3d, depth_o3d,
                                                                           depth_scale=1.0,
                                                                           convert_rgb_to_intensity=False)
        self.tsdf.integrate(image=objd_image,  # rgbd_image,
                              intrinsic=intrinsic_open3d,
                              extrinsic=T_CW)


def postprocess_mmdet_results(result, score_thr=0.8):
    # for panoptic
    # pan_results = result['pan_results']
    # # keep objects ahead
    # ids = np.unique(pan_results)[::-1]
    # legal_indices = ids != self.num_classes  # for VOID label
    # ids = ids[legal_indices]
    # labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
    # segms = (pan_results[None] == ids[:, None, None])

    # print("results ", result)
    bbox_result, segm_result = result
    if isinstance(segm_result, tuple):
        segm_result = segm_result[0]  # ms rcnn

    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)

    # filter by score thresh
    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :-1].astype(np.int)
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]
    # print("bbox ", bboxes.shape)
    # print("segms ", segms.shape)
    # print("labels ", labels.shape)
    # print("bbox ", bboxes)
    # print("seg ", segms)
    # print("labels ", labels)
    return bboxes, labels, segms


def unproject_pointcloud(depth, intrinsic_open3d, T_CW):
    # depth, mask, intrinsic, extrinsic -> point clouds
    pc_sample = open3d.geometry.PointCloud.create_from_depth_image(depth=open3d.geometry.Image(depth),
                                                                   intrinsic=intrinsic_open3d,
                                                                   extrinsic=T_CW,
                                                                   depth_scale=1.0,
                                                                   project_valid_depth_only=True)
    return pc_sample

# def check3DIoU(inst_bbox, candidate_bbox):
#     bbox1 = open3d_bbox3D_to_torch3d(inst_bbox)
#     bbox2 = open3d_bbox3D_to_torch3d(candidate_bbox)
#     # print("-----")
#     # print(bbox1)
#     # print(bbox2)
#     vol, IoU = box3d_overlap(bbox1, bbox2)
#     # print("IoU ", IoU)
#     return IoU

# todo check inside bbox3D ratio instead of time consuming 3D IoU
def check_inside_ratio(pc, bbox3D):
    #  pc, bbox3d -> inside ratio
    indices = bbox3D.get_point_indices_within_bounding_box(pc.points)
    assert len(pc.points) > 0
    ratio = len(indices) / len(pc.points)
    # print("ratio ", ratio)
    return ratio, indices

def open3d_bbox3D_to_torch3d(bbox3D):
    # box_corner = np.array(bbox3D.get_box_points())
    x,y,z = bbox3D.center
    w,h,le = bbox3D.extent
    box_corner = torch.tensor(
        [
            [x - w / 2.0, y - h / 2.0, z - le / 2.0],
            [x + w / 2.0, y - h / 2.0, z - le / 2.0],
            [x + w / 2.0, y + h / 2.0, z - le / 2.0],
            [x - w / 2.0, y + h / 2.0, z - le / 2.0],
            [x - w / 2.0, y - h / 2.0, z + le / 2.0],
            [x + w / 2.0, y - h / 2.0, z + le / 2.0],
            [x + w / 2.0, y + h / 2.0, z + le / 2.0],
            [x - w / 2.0, y + h / 2.0, z + le / 2.0],
        ],
        device="cpu",
        dtype=torch.float32,
    )
    # return torch.from_numpy(box_corner).type(torch.float32).unsqueeze(0)
    return box_corner.unsqueeze(0)

def box_filter(masks, classes, depth, inst_dict, intrinsic_open3d, T_CW, min_pixels=500, voxel_size=0.01):
    bbox3d_scale = 1.0  # 1.05
    inst_data = np.zeros_like(depth, dtype=np.int)
    for i in range(len(masks)):
        diff_mask = None
        inst_mask = masks[i]  # smaller_mask #masks[i]    # todo only
        inst_id = classes[i]
        if inst_id == 0:
            continue
        inst_depth = np.copy(depth)
        inst_depth[~inst_mask] = 0.  # inst_mask
        # proj_time = time.time()
        inst_pc = unproject_pointcloud(inst_depth, intrinsic_open3d, T_CW)
        # print("proj time ", time.time()-proj_time)
        if len(inst_pc.points) <= 10:  # too small    20  400 # todo use sem to set background
            inst_data[inst_mask] = 0  # set to background
            continue
        if inst_id in inst_dict.keys():
            candidate_inst = inst_dict[inst_id]
            # iou_time = time.time()
            IoU, indices = check_inside_ratio(inst_pc, candidate_inst.bbox3D)
            # print("iou time ", time.time()-iou_time)
            # if indices empty
            candidate_inst.cmp_cnt += 1
            if len(indices) >= 1:
                candidate_inst.pc += inst_pc.select_by_index(indices)  # only merge pcs inside scale*bbox
                # todo check indices follow valid depth
                valid_depth_mask = np.zeros_like(inst_depth, dtype=np.bool)
                valid_pc_mask = valid_depth_mask[inst_depth!=0]
                valid_pc_mask[indices] = True
                valid_depth_mask[inst_depth != 0] = valid_pc_mask
                valid_mask = valid_depth_mask
                diff_mask = np.zeros_like(inst_mask)
                # uv_opencv, _ = cv2.projectPoints(np.array(inst_pc.select_by_index(indices).points), T_CW[:3, :3],
                #                                  T_CW[:3, 3], intrinsic_open3d.intrinsic_matrix[:3, :3], None)
                # uv = np.round(uv_opencv).squeeze().astype(int)
                # u = uv[:, 0].reshape(-1, 1)
                # v = uv[:, 1].reshape(-1, 1)
                # vu = np.concatenate([v, u], axis=-1)
                # valid_mask = np.zeros_like(inst_mask)
                # valid_mask[tuple(vu.T)] = True
                # # cv2.imshow("valid", (inst_depth!=0).astype(np.uint8)*255)
                # # cv2.waitKey(1)
                diff_mask[(inst_depth != 0) & (~valid_mask)] = True
                # cv2.imshow("diff_mask", diff_mask.astype(np.uint8) * 255)
                # cv2.waitKey(1)
            else:   # merge all for scannet todo
                # candidate_inst.pc += inst_pc
                print("too few pcs obj ", inst_id)
                inst_data[inst_mask] = -1
                continue
            # downsample_time = time.time()
            # adapt_voxel_size = np.maximum(np.max(candidate_inst.bbox3D.extent)/100, 0.1) # todo adaptive voxel size
            candidate_inst.pc = candidate_inst.pc.voxel_down_sample(voxel_size) # adapt_voxel_size
            # candidate_inst.pc = candidate_inst.pc.farthest_point_down_sample(500)
            # candidate_inst.pc = candidate_inst.pc.random_down_sample(np.minimum(len(candidate_inst.pc.points)/500.,1))
            # print("downsample time ", time.time() - downsample_time)  # 0.03s even
            # bbox_time = time.time()
            try:
                candidate_inst.bbox3D = open3d.geometry.OrientedBoundingBox.create_from_points(candidate_inst.pc.points)
                # print("bbox time ", time.time()-bbox_time)  # 0.03s even
            except RuntimeError:
                # print("bbox time ", time.time() - bbox_time)
                print("too few pcs obj ", inst_id)
                inst_data[inst_mask] = -1
                continue

            # candidate_inst.bbox3D = open3d.geometry.OrientedBoundingBox.create_from_points(candidate_inst.pc.points)
            # enlarge
            candidate_inst.bbox3D.scale(bbox3d_scale, candidate_inst.bbox3D.get_center())
            # inst_id = candidate_inst.inst_id
        else:   # new inst
            # init new inst and new sem
            new_inst = InstData()
            new_inst.inst_id = inst_id
            smaller_mask = cv2.erode(inst_mask.astype(np.uint8), np.ones((5, 5)), iterations=3).astype(bool)
            if np.sum(smaller_mask) < min_pixels:
                print("too few pcs obj ", inst_id)
                inst_data[inst_mask] = 0
                continue
            inst_depth_small = depth.copy()
            inst_depth_small[~smaller_mask] = 0
            inst_pc_small = unproject_pointcloud(inst_depth_small, intrinsic_open3d, T_CW)
            new_inst.pc = inst_pc_small
            new_inst.pc = new_inst.pc.voxel_down_sample(voxel_size)
            try:
                inst_bbox3D = open3d.geometry.OrientedBoundingBox.create_from_points(new_inst.pc.points)
            except RuntimeError:
                print("too few pcs obj ", inst_id)
                inst_data[inst_mask] = 0
                continue
            # inst_bbox3D = open3d.geometry.OrientedBoundingBox.create_from_points(new_inst.pc.points)
            # scale up
            inst_bbox3D.scale(bbox3d_scale, inst_bbox3D.get_center())
            new_inst.bbox3D = inst_bbox3D
            # update inst_dict
            inst_dict.update({inst_id: new_inst})  # init new sem

        # update inst_data
        if inst_id == 39:
            print("bbox ", inst_dict[inst_id].bbox3D.extent)
        inst_data[inst_mask] = inst_id
        if diff_mask is not None:
            inst_data[diff_mask] = -1  # unsure area

    return inst_data

def track_instance(masks, classes, depth, inst_list, sem_dict, intrinsic_open3d, T_CW, IoU_thresh=0.5, voxel_size=0.1,
                   min_pixels=2000, erode=True, clip_features=None, class_names=None):
    inst_data = np.zeros_like(depth, dtype=np.int)
    bbox3d_scale = 1.0
    for i in range(len(masks)):
        # # print("class ", classes[i])
        # if erode:
        #     smaller_mask = cv2.erode(masks[i].astype(np.uint8), np.ones((5, 5)), iterations=3).astype(bool)
        #     # smaller_mask = cv2.erode(masks[i].astype(np.uint8), np.ones((3, 3)), iterations=10).astype(bool)
        # else:
        #     smaller_mask = masks[i]
        # smaller_mask = cv2.erode(masks[i].astype(np.uint8), np.ones((5, 5)), iterations=3).astype(bool)
        smaller_mask = cv2.erode(masks[i].astype(np.uint8), np.ones((5, 5)), iterations=3).astype(bool)
        inst_depth_small = depth.copy()
        inst_depth_small[~smaller_mask] = 0
        inst_pc_small = unproject_pointcloud(inst_depth_small, intrinsic_open3d, T_CW)
        # smaller_mask = masks[i]
        diff_mask = None
        # depth_mask = cv2.dilate(depth, np.ones((5, 5)), iterations=3).astype(bool)
        # diff_depth = np.abs(depth_mask - depth) > 0.20
        # diff_mask = masks[i] ^ smaller_mask & diff_depth  # todo mask edge & large depth diff
        # cv2.imshow("diff mask ", diff_mask.astype(np.uint8)*255)
        # cv2.waitKey(1)
        if np.sum(smaller_mask) <= min_pixels:  # too small    20  400 # todo use sem to set background
            inst_data[masks[i]] = 0  # set to background
            continue
        inst_pc_voxel = inst_pc_small.voxel_down_sample(voxel_size)
        if len(inst_pc_voxel.points) <= 10:  # too small    20  400 # todo use sem to set background
            inst_data[masks[i]] = 0  # set to background
            continue
        is_merged = False
        inst_id = None
        inst_mask = masks[i] #smaller_mask #masks[i]    # todo only
        inst_class = classes[i]
        inst_depth = np.copy(depth)
        inst_depth[~masks[i]] = 0.  # inst_mask
        # valid_depth_mask = inst_depth !=0
        # p = time.time()
        inst_pc = unproject_pointcloud(inst_depth, intrinsic_open3d, T_CW)
        # inst_pc_voxel = inst_pc.voxel_down_sample(voxel_size)
        # if len(inst_pc_voxel.points) <= 10:  # too small    20  400 # todo use sem to set background
        #     inst_data[masks[i]] = 0  # set to background
        #     continue

        # print("init time ", time.time() - p)
        sem_inst_list = []
        if clip_features is not None: # check similar sems based on clip feature distance
            sem_thr = 260.
            for sem_exist in sem_dict.keys():
                if torch.abs(clip_features[class_names[inst_class]] - clip_features[class_names[sem_exist]]).sum() < sem_thr:
                    sem_inst_list.extend(sem_dict[sem_exist])
        else:   # no clip features, only do strictly sem check
            if inst_class in sem_dict.keys():
                s = time.time()
                # delete_list = []
                sem_inst_list.extend(sem_dict[inst_class])

        for candidate_inst in sem_inst_list:
    # if True:  # only consider 3D bbox, merge them if they are spatial together
    #     for candidate_inst in inst_list:
            # IoU = check3DIoU(inst_bbox3D, candidate_inst.bbox3D)
            IoU, indices = check_inside_ratio(inst_pc, candidate_inst.bbox3D)
            # print("IoU ", IoU)
            candidate_inst.cmp_cnt += 1
            # print("cmp time ", candidate_inst.cmp_cnt)
            # print("merge time ", candidate_inst.merge_cnt)
            if IoU > IoU_thresh:
                # merge inst to candidate
                is_merged = True
                candidate_inst.merge_cnt += 1
                candidate_inst.pc += inst_pc.select_by_index(indices)   # only merge pcs inside scale*bbox
                diff_mask = np.zeros_like(inst_mask)
                # assert tmp.shape == len(i)
                # print("tmp ", tmp.shape)
                # print("indices ", np.array(inst_pc.points).shape)
                # T_WC = np.linalg.inv(T_CW)
                uv_opencv, _ = cv2.projectPoints(np.array(inst_pc.select_by_index(indices).points), T_CW[:3, :3],
                                                 T_CW[:3, 3], intrinsic_open3d.intrinsic_matrix[:3,:3], None)
                uv = np.round(uv_opencv).squeeze().astype(int)
                # intrinsic = intrinsic_open3d.intrinsic_matrix[:3,:3]
                # P = intrinsic @ T_CW[:3, :]  # 3x4
                # uv_, _ = project_to_image(np.array(inst_pc.select_by_index(indices).points), P)  # Nx2
                # print("uv ", uv_)
                # print("uv_opencv raw ", uv_opencv)
                # print("uv_opencv ", uv)
                # UV = np.round(uv).astype(np.uint8)
                # print("UV ", UV)
                u = uv[:, 0].reshape(-1, 1)
                v = uv[:, 1].reshape(-1, 1)
                vu = np.concatenate([v, u], axis=-1)
                # vu = np.concatenate([u, v], axis=-1)
                # print("tuple(vu.T) ", tuple(vu.T))
                valid_mask = np.zeros_like(inst_mask)
                valid_mask[tuple(vu.T)] = True
                # cv2.imshow("valid", (inst_depth!=0).astype(np.uint8)*255)
                # cv2.waitKey(1)

                # diff_mask[inst_mask & ~valid_mask] = True
                diff_mask[(inst_depth!=0) & (~valid_mask)] = True
                # cv2.imshow("diff_mask", diff_mask.astype(np.uint8) * 255)
                # cv2.waitKey(1)
                candidate_inst.pc = candidate_inst.pc.voxel_down_sample(voxel_size)
                # candidate_inst.bbox3D = candidate_inst.pc.get_oriented_bounding_box()
                # candidate_inst.pc.random_down_sample(np.minimum(500//len(candidate_inst.pc.points),1))
                candidate_inst.bbox3D = open3d.geometry.OrientedBoundingBox.create_from_points(candidate_inst.pc.points)
                # enlarge
                candidate_inst.bbox3D.scale(bbox3d_scale, candidate_inst.bbox3D.get_center())
                inst_id = candidate_inst.inst_id
                # print("merged!!!!")
                break
            if candidate_inst.cmp_cnt >= 20 and candidate_inst.merge_cnt <= 5:
                sem_inst_list.remove(candidate_inst)
                # print("delete")
                # del candidate_inst
        # del sem_dict[inst_class][delete_list]
        # print("merge time ", time.time() - s)
        if not is_merged:
            # init new inst and new sem
            new_inst = InstData()
            new_inst.inst_id = len(inst_list) + 1
            new_inst.class_id = inst_class

            new_inst.pc = inst_pc_small
            new_inst.pc = new_inst.pc.voxel_down_sample(voxel_size)
            inst_bbox3D = open3d.geometry.OrientedBoundingBox.create_from_points(new_inst.pc.points)
            # scale up
            inst_bbox3D.scale(bbox3d_scale, inst_bbox3D.get_center())
            new_inst.bbox3D = inst_bbox3D
            inst_list.append(new_inst)
            inst_id = new_inst.inst_id
            # update sem_dict
            if inst_class in sem_dict.keys():
                sem_dict[inst_class].append(new_inst)   # append new inst to exist sem
            else:
                sem_dict.update({inst_class: [new_inst]})   # init new sem
        # update inst_data
        inst_data[inst_mask] = inst_id
        if diff_mask is not None:
            inst_data[diff_mask] = -1   # unsure area

        # inst_data[masks[i]] = inst_id

    return inst_data

# def track_instance(masks, classes, depth, inst_list, sem_dict, intrinsic_open3d, T_CW, IoU_thresh=0.5, voxel_size=0.1,
#                    min_pixels=2000, erode=True):
#     fuse_thr = 0.1
#     inst_data = np.zeros_like(depth, dtype=np.uint8)
#     depth_o3d = open3d.geometry.Image(depth.astype(np.float32))
#     for i in range(len(masks)):
#         # if np.sum(masks[i]) <= 1500:  # too small    20  400 # todo use sem to set background
#         #     inst_data[masks[i]] = 0  # set to background
#         #     continue
#         if erode:
#             smaller_mask = cv2.erode(masks[i].astype(np.uint8), np.ones((5, 5)), iterations=3).astype(bool)
#             # smaller_mask = cv2.erode(masks[i].astype(np.uint8), np.ones((3, 3)), iterations=10).astype(bool)
#
#         else:
#             smaller_mask = masks[i]
#         if np.sum(smaller_mask) <= min_pixels:  # too small    20  400 # todo use sem to set background
#             inst_data[masks[i]] = 0  # set to background
#             continue
#         is_merged = False
#         inst_id = None
#         inst_mask = smaller_mask #masks[i]
#         inst_class = classes[i]
#         inst_depth = np.copy(depth)
#         inst_depth[~inst_mask] = 0.
#         # inst_depth_o3d = open3d.geometry.Image(inst_depth.astype(np.float32))
#         p = time.time()
#         inst_pc = unproject_pointcloud(inst_depth, intrinsic_open3d, T_CW)
#         # inst_pc = inst_pc.voxel_down_sample(voxel_size)
#         if len(inst_pc.voxel_down_sample(voxel_size).points) <= 10:  # too small    20  400 # todo use sem to set background
#             inst_data[masks[i]] = 0  # set to background
#             continue
#
#         # print("init time ", time.time() - p)
#
#         if inst_class in sem_dict.keys():
#             s = time.time()
#             # delete_list = []
#             sem_inst_list = sem_dict[inst_class]
#             for candidate_inst in sem_inst_list:
#         # if True:  # only consider 3D bbox, merge them if they are spatial together
#         #     for candidate_inst in inst_list:
#                 # IoU = check3DIoU(inst_bbox3D, candidate_inst.bbox3D)
#                 IoU, _ = check_inside_ratio(inst_pc, candidate_inst.bbox3D)
#                 candidate_inst.cmp_cnt += 1
#                 # print("cmp time ", candidate_inst.cmp_cnt)
#                 # print("merge time ", candidate_inst.merge_cnt)
#                 # print("len ori ", len(inst_pc.points))
#                 if IoU > IoU_thresh:
#                     # merge inst to candidate
#                     is_merged = True
#                     candidate_inst.merge_cnt += 1
#                     # todo use tsdf bbox & pc, instead of raw pcs
#                     # integrate tsdf
#                     # todo not binary tsdf, but global tsdf label
#                     # obj_mask = np.zeros_like(depth, dtype=np.float32)
#                     # obj_mask[inst_mask] = 1.
#                     # obj_o3d = open3d.geometry.Image(obj_mask)
#                     # objd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(obj_o3d, depth_o3d,
#                     #                                                                 depth_scale=1.0,
#                     #                                                                 convert_rgb_to_intensity=False)
#                     # candidate_inst.tsdf.integrate(image=objd_image,   #rgbd_image,
#                     #                               intrinsic=intrinsic_open3d,
#                     #                               extrinsic=T_CW)
#                     candidate_inst.integrate_tsdf(depth_o3d, inst_mask, intrinsic_open3d, T_CW)
#                     all_pc = candidate_inst.tsdf.extract_point_cloud()
#                     # candidate_inst.pc = all_pc.points[np.array(all_pc.colors) > 0.5]
#                     # print("c ", np.unique(np.array(all_pc.colors)))
#                     # print("color ", np.where(np.array(all_pc.colors) > fuse_thr)[0])
#                     candidate_inst.pc = all_pc.select_by_index(np.where(np.array(all_pc.colors) > fuse_thr)[0])
#                     if len(candidate_inst.pc.points) < 10:  # too few
#                         is_merged = False
#                         continue
#                     # candidate_inst.pc += inst_pc[pc_indices]    # todo inst_pc
#                     # candidate_inst.pc = candidate_inst.pc.voxel_down_sample(voxel_size)
#                     # candidate_inst.pc.random_down_sample(np.minimum(500//len(candidate_inst.pc.points),1))
#                     candidate_inst.bbox3D = open3d.geometry.OrientedBoundingBox.create_from_points(candidate_inst.pc.points)
#                     # check with 2D mask
#                     _, pc_indices = check_inside_ratio(inst_pc, candidate_inst.bbox3D)
#                     # print("len filter ", len(pc_indices))
#                     # print("indices ", pc_indices)
#                     # open3d.visualization.draw_geometries([inst_pc.select_by_index(pc_indices)])
#
#                     filtered_mask = np.zeros_like(inst_mask)
#                     tmp = filtered_mask[inst_mask]
#                     tmp[pc_indices] = True
#                     filtered_mask[inst_mask] = tmp
#                     diff = inst_mask.astype(np.bool) ^ filtered_mask
#                     # print("len f mask ", np.sum(filtered_mask))
#                     # cv2.imshow("ori", inst_mask.astype(np.uint8) * 255)
#                     # cv2.imshow("filter", filtered_mask.astype(np.uint8) * 255)
#                     # cv2.imshow("diff", diff.astype(np.uint8)*255)
#                     # cv2.waitKey(0)
#                     inst_mask = filtered_mask.copy()    # filter 2D mask by TSDF info
#                     # enlarge
#                     candidate_inst.bbox3D.scale(1.2, candidate_inst.bbox3D .get_center())
#                     inst_id = candidate_inst.inst_id
#                     # print("merged!!!!")
#                     break
#                 if candidate_inst.cmp_cnt >= 20 and candidate_inst.merge_cnt <=5:
#                     sem_inst_list.remove(candidate_inst)
#                     print("delete")
#                     # del candidate_inst
#             # del sem_dict[inst_class][delete_list]
#             # print("merge time ", time.time() - s)
#         if not is_merged:
#             # init new inst and new sem
#             new_inst = InstData()
#             new_inst.inst_id = len(inst_list) + 1
#             new_inst.class_id = inst_class
#             # todo integrate tsdf
#             new_inst.integrate_tsdf(depth_o3d, inst_mask, intrinsic_open3d, T_CW)
#             all_pc = new_inst.tsdf.extract_point_cloud()
#             # new_inst.pc = all_pc.points.set[np.array(all_pc.colors) > 0.5]
#             new_inst.pc = all_pc.select_by_index(np.where(np.array(all_pc.colors) > fuse_thr)[0])
#
#             # new_inst.pc = inst_pc
#             # new_inst.pc.voxel_down_sample(voxel_size)
#             inst_bbox3D = open3d.geometry.OrientedBoundingBox.create_from_points(inst_pc.points)
#             # scale up
#             inst_bbox3D.scale(1.2, inst_bbox3D.get_center())
#             new_inst.bbox3D = inst_bbox3D
#             inst_list.append(new_inst)
#             inst_id = new_inst.inst_id
#             # update sem_dict
#             if inst_class in sem_dict.keys():
#                 sem_dict[inst_class].append(new_inst)   # append new inst to exist sem
#             else:
#                 sem_dict.update({inst_class: [new_inst]})   # init new sem
#         # update inst_data
#         inst_data[inst_mask] = inst_id
#         # inst_data[masks[i]] = inst_id
#
#     return inst_data