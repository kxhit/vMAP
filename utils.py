import cv2
import imgviz
import numpy as np
import torch
from functorch import combine_state_for_ensemble
import open3d
import queue
import copy
import torch.utils.dlpack

class BoundingBox():
    def __init__(self):
        super(BoundingBox, self).__init__()
        self.extent = None
        self.R = None
        self.center = None
        self.points3d = None    # (8,3)

def bbox_open3d2bbox(bbox_o3d):
    bbox = BoundingBox()
    bbox.extent = bbox_o3d.extent
    bbox.R = bbox_o3d.R
    bbox.center = bbox_o3d.center
    return bbox

def bbox_bbox2open3d(bbox):
    bbox_o3d = open3d.geometry.OrientedBoundingBox(bbox.center, bbox.R, bbox.extent)
    return bbox_o3d

def update_vmap(models, optimiser):
    fmodel, params, buffers = combine_state_for_ensemble(models)
    [p.requires_grad_() for p in params]
    optimiser.add_param_group({"params": params})  # imap b l
    return (fmodel, params, buffers)

def enlarge_bbox(bbox, scale, w, h):
    assert scale >= 0
    # print(bbox)
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale * (max_x - min_x))
    margin_y = int(0.5 * scale * (max_y - min_y))
    if margin_y == 0 or margin_x == 0:
        return None
    # assert margin_x != 0
    # assert margin_y != 0
    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    min_x = np.clip(min_x, 0, w-1)
    min_y = np.clip(min_y, 0, h-1)
    max_x = np.clip(max_x, 0, w-1)
    max_y = np.clip(max_y, 0, h-1)

    bbox_enlarged = [int(min_x), int(min_y), int(max_x), int(max_y)]
    return bbox_enlarged

def get_bbox2d(obj_mask, bbox_scale=1.0):
    contours, hierarchy = cv2.findContours(obj_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
                          -2:]
    # # Find the index of the largest contour
    # areas = [cv2.contourArea(c) for c in contours]
    # max_index = np.argmax(areas)
    # cnt = contours[max_index]
    # Concatenate all contours
    if len(contours) == 0:
        return None
    cnt = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(cnt)  # todo if multiple contours, choose the outmost one?
    # x, y, w, h = cv2.boundingRect(contours)
    bbox_enlarged = enlarge_bbox([x, y, x + w, y + h], scale=bbox_scale, w=obj_mask.shape[1], h=obj_mask.shape[0])
    return bbox_enlarged

def get_bbox2d_batch(img):
    b,h,w = img.shape[:3]
    rows = torch.any(img, axis=2)
    cols = torch.any(img, axis=1)
    rmins = torch.argmax(rows.float(), dim=1)
    rmaxs = h - torch.argmax(rows.float().flip(dims=[1]), dim=1)
    cmins = torch.argmax(cols.float(), dim=1)
    cmaxs = w - torch.argmax(cols.float().flip(dims=[1]), dim=1)

    return rmins, rmaxs, cmins, cmaxs

def get_latest_queue(q):
    message = None
    while(True):
        try:
            message_latest = q.get(block=False)
            if message is not None:
                del message
            message = message_latest

        except queue.Empty:
            break

    return message

# for association/tracking
class InstData:
    def __init__(self):
        super(InstData, self).__init__()
        self.bbox3D = None
        self.inst_id = None     # instance
        self.class_id = None    # semantic
        self.pc_sample = None
        self.merge_cnt = 0  # merge times counting
        self.cmp_cnt = 0


def box_filter(masks, classes, depth, inst_dict, intrinsic_open3d, T_CW, min_pixels=500, voxel_size=0.01):
    bbox3d_scale = 1.0  # 1.05
    inst_data = np.zeros_like(depth, dtype=np.int)
    for i in range(len(masks)):
        diff_mask = None
        inst_mask = masks[i]
        inst_id = classes[i]
        if inst_id == 0:
            continue
        inst_depth = np.copy(depth)
        inst_depth[~inst_mask] = 0.  # inst_mask
        # proj_time = time.time()
        inst_pc = unproject_pointcloud(inst_depth, intrinsic_open3d, T_CW)
        # print("proj time ", time.time()-proj_time)
        if len(inst_pc.points) <= 10:  # too small
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
            else:   # merge all for scannet
                # print("too few pcs obj ", inst_id)
                inst_data[inst_mask] = -1
                continue
            # downsample_time = time.time()
            # adapt_voxel_size = np.maximum(np.max(candidate_inst.bbox3D.extent)/100, 0.1)
            candidate_inst.pc = candidate_inst.pc.voxel_down_sample(voxel_size) # adapt_voxel_size
            # candidate_inst.pc = candidate_inst.pc.farthest_point_down_sample(500)
            # candidate_inst.pc = candidate_inst.pc.random_down_sample(np.minimum(len(candidate_inst.pc.points)/500.,1))
            # print("downsample time ", time.time() - downsample_time)  # 0.03s even
            # bbox_time = time.time()
            try:
                candidate_inst.bbox3D = open3d.geometry.OrientedBoundingBox.create_from_points(candidate_inst.pc.points)
            except RuntimeError:
                # print("too few pcs obj ", inst_id)
                inst_data[inst_mask] = -1
                continue
            # enlarge
            candidate_inst.bbox3D.scale(bbox3d_scale, candidate_inst.bbox3D.get_center())
        else:   # new inst
            # init new inst and new sem
            new_inst = InstData()
            new_inst.inst_id = inst_id
            smaller_mask = cv2.erode(inst_mask.astype(np.uint8), np.ones((5, 5)), iterations=3).astype(bool)
            if np.sum(smaller_mask) < min_pixels:
                # print("too few pcs obj ", inst_id)
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
                # print("too few pcs obj ", inst_id)
                inst_data[inst_mask] = 0
                continue
            # scale up
            inst_bbox3D.scale(bbox3d_scale, inst_bbox3D.get_center())
            new_inst.bbox3D = inst_bbox3D
            # update inst_dict
            inst_dict.update({inst_id: new_inst})  # init new sem

        # update inst_data
        inst_data[inst_mask] = inst_id
        if diff_mask is not None:
            inst_data[diff_mask] = -1  # unsure area

    return inst_data

def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)

def check_mask_order(obj_masks, depth_np, obj_ids):
    print(len(obj_masks))
    print(len(obj_ids))

    assert len(obj_masks) == len(obj_ids)
    depth = torch.from_numpy(depth_np)
    obj_masked_modified = copy.deepcopy(obj_masks[:])
    for i in range(len(obj_masks) - 1):

        mask1 = obj_masks[i].float()
        mask1_ = obj_masked_modified[i].float()
        for j in range(i + 1, len(obj_masks)):
            mask2 = obj_masks[j].float()
            mask2_ = obj_masked_modified[j].float()
            # case 1: if they don't intersect we don't touch them
            if ((mask1 + mask2) == 2).sum() == 0:
                continue
            # case 2: the entire object 1 is inside of object 2, we say object 1 is in front of object 2:
            elif (((mask1 + mask2) == 2).float() - mask1).sum() == 0:
                mask2_ -= mask1_
            # case 3: the entire object 2 is inside of object 1, we say object 2 is in front of object 1:
            elif (((mask1 + mask2) == 2).float() - mask2).sum() == 0:
                mask1_ -= mask2_
            # case 4: use depth to check object order:
            else:
                # object 1 is closer
                if (depth * mask1).sum() / mask1.sum() > (depth * mask2).sum() / mask2.sum():
                    mask2_ -= ((mask1 + mask2) == 2).float()
                # object 2 is closer
                if (depth * mask1).sum() / mask1.sum() < (depth * mask2).sum() / mask2.sum():
                    mask1_ -= ((mask1 + mask2) == 2).float()

    final_mask = torch.zeros_like(depth, dtype=torch.int)
    # instance_labels = {}
    for i in range(len(obj_masked_modified)):
        final_mask = final_mask.masked_fill(obj_masked_modified[i] > 0, obj_ids[i])
        # instance_labels[i] = obj_ids[i].item()
    return final_mask.cpu().numpy()


def unproject_pointcloud(depth, intrinsic_open3d, T_CW):
    # depth, mask, intrinsic, extrinsic -> point clouds
    pc_sample = open3d.geometry.PointCloud.create_from_depth_image(depth=open3d.geometry.Image(depth),
                                                                   intrinsic=intrinsic_open3d,
                                                                   extrinsic=T_CW,
                                                                   depth_scale=1.0,
                                                                   project_valid_depth_only=True)
    return pc_sample

def check_inside_ratio(pc, bbox3D):
    #  pc, bbox3d -> inside ratio
    indices = bbox3D.get_point_indices_within_bounding_box(pc.points)
    assert len(pc.points) > 0
    ratio = len(indices) / len(pc.points)
    # print("ratio ", ratio)
    return ratio, indices

def track_instance(masks, classes, depth, inst_list, sem_dict, intrinsic_open3d, T_CW, IoU_thresh=0.5, voxel_size=0.1,
                   min_pixels=2000, erode=True, clip_features=None, class_names=None):
    device = masks.device
    inst_data_dict = {}
    inst_data_dict.update({0: torch.zeros(depth.shape, dtype=torch.int, device=device)})
    inst_ids = []
    bbox3d_scale = 1.0  # todo 1.0
    min_extent = 0.05
    depth = torch.from_numpy(depth).to(device)
    for i in range(len(masks)):
        inst_data = torch.zeros(depth.shape, dtype=torch.int, device=device)
        smaller_mask = cv2.erode(masks[i].detach().cpu().numpy().astype(np.uint8), np.ones((5, 5)), iterations=3).astype(bool)
        inst_depth_small = depth.detach().cpu().numpy()
        inst_depth_small[~smaller_mask] = 0
        inst_pc_small = unproject_pointcloud(inst_depth_small, intrinsic_open3d, T_CW)
        diff_mask = None
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
        inst_depth = depth.detach().cpu().numpy()
        inst_depth[~masks[i].detach().cpu().numpy()] = 0.  # inst_mask
        inst_pc = unproject_pointcloud(inst_depth, intrinsic_open3d, T_CW)
        sem_inst_list = []
        if clip_features is not None: # check similar sems based on clip feature distance
            sem_thr = 200 #300. for table #320.  # 260.
            for sem_exist in sem_dict.keys():
                if torch.abs(clip_features[class_names[inst_class]] - clip_features[class_names[sem_exist]]).sum() < sem_thr:
                    sem_inst_list.extend(sem_dict[sem_exist])
        else:   # no clip features, only do strictly sem check
            if inst_class in sem_dict.keys():
                sem_inst_list.extend(sem_dict[inst_class])

        for candidate_inst in sem_inst_list:
    # if True:  # only consider 3D bbox, merge them if they are spatial together
            IoU, indices = check_inside_ratio(inst_pc, candidate_inst.bbox3D)
            candidate_inst.cmp_cnt += 1
            if IoU > IoU_thresh:
                # merge inst to candidate
                is_merged = True
                candidate_inst.merge_cnt += 1
                candidate_inst.pc += inst_pc.select_by_index(indices)
                # inst_uv = inst_pc.select_by_index(indices).project_to_depth_image(masks[i].shape[1], masks[i].shape[0], intrinsic_open3d, T_CW, depth_scale=1.0, depth_max=10.0)
                # # inst_uv = torch.utils.dlpack.from_dlpack(uv_opencv.as_tensor().to_dlpack())
                # valid_mask = inst_uv.squeeze() > 0.  # shape --> H, W
                # diff_mask = (inst_depth > 0.) & (~valid_mask)
                diff_mask = torch.zeros_like(inst_mask)
                uv_opencv, _ = cv2.projectPoints(np.array(inst_pc.select_by_index(indices).points), T_CW[:3, :3],
                                                 T_CW[:3, 3], intrinsic_open3d.intrinsic_matrix[:3,:3], None)
                uv = np.round(uv_opencv).squeeze().astype(int)
                u = uv[:, 0].reshape(-1, 1)
                v = uv[:, 1].reshape(-1, 1)
                vu = np.concatenate([v, u], axis=-1)
                valid_mask = np.zeros(inst_mask.shape, dtype=np.bool)
                valid_mask[tuple(vu.T)] = True
                diff_mask[(inst_depth!=0) & (~valid_mask)] = True
                # downsample pcs
                candidate_inst.pc = candidate_inst.pc.voxel_down_sample(voxel_size)
                # candidate_inst.pc.random_down_sample(np.minimum(500//len(candidate_inst.pc.points),1))
                candidate_inst.bbox3D = open3d.geometry.OrientedBoundingBox.create_from_points(candidate_inst.pc.points)
                # enlarge
                candidate_inst.bbox3D.scale(bbox3d_scale, candidate_inst.bbox3D.get_center())
                candidate_inst.bbox3D.extent = np.maximum(candidate_inst.bbox3D.extent, min_extent) # at least bigger than min_extent
                inst_id = candidate_inst.inst_id
                break
            # if candidate_inst.cmp_cnt >= 20 and candidate_inst.merge_cnt <= 5:
            #     sem_inst_list.remove(candidate_inst)

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
            inst_bbox3D.extent = np.maximum(inst_bbox3D.extent, min_extent)
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
        if inst_id not in inst_ids:
            inst_data_dict.update({inst_id: inst_data})
        else:
            continue
            # idx = inst_ids.index(inst_id)
            # inst_data_list[idx] = inst_data_list[idx] & torch.from_numpy(inst_data) # merge them? todo
    # return inst_data
    mask_bg = torch.stack(list(inst_data_dict.values())).sum(0) != 0
    inst_data_dict.update({0: mask_bg.int()})
    return inst_data_dict
