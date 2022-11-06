import cv2
import numpy as np
import torch
from functorch import combine_state_for_ensemble

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
    cnt = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(cnt)  # todo if multiple contours, choose the outmost one?
    # x, y, w, h = cv2.boundingRect(contours)
    bbox_enlarged = enlarge_bbox([x, y, x + w, y + h], scale=bbox_scale, w=obj_mask.shape[1], h=obj_mask.shape[0])
    return bbox_enlarged

def get_bbox2d_batch(img):
    #
    b,h,w = img.shape[:3]
    rows = torch.any(img, axis=2)
    cols = torch.any(img, axis=1)
    rmins = torch.argmax(rows.float(), dim=1)
    rmaxs = h - torch.argmax(rows.float().flip(dims=[1]), dim=1)
    cmins = torch.argmax(cols.float(), dim=1)
    cmaxs = w - torch.argmax(cols.float().flip(dims=[1]), dim=1)

    return rmins, rmaxs, cmins, cmaxs