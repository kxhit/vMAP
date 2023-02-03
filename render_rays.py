import torch
import numpy as np

def occupancy_activation(alpha, distances=None):
    # occ = 1.0 - torch.exp(-alpha * distances)
    occ = torch.sigmoid(alpha)    # unisurf

    return occ

def alpha_to_occupancy(depths, dirs, alpha, add_last=False):
    interval_distances = depths[..., 1:] - depths[..., :-1]
    if add_last:
        last_distance = torch.empty(
            (depths.shape[0], 1),
            device=depths.device,
            dtype=depths.dtype).fill_(0.1)
        interval_distances = torch.cat(
            [interval_distances, last_distance], dim=-1)

    dirs_norm = torch.norm(dirs, dim=-1)
    interval_distances = interval_distances * dirs_norm[:, None]
    occ = occupancy_activation(alpha, interval_distances)

    return occ

def occupancy_to_termination(occupancy, is_batch=False):
    if is_batch:
        first = torch.ones(list(occupancy.shape[:2]) + [1], device=occupancy.device)
        free_probs = (1. - occupancy + 1e-10)[:, :, :-1]
    else:
        first = torch.ones([occupancy.shape[0], 1], device=occupancy.device)
        free_probs = (1. - occupancy + 1e-10)[:, :-1]
    free_probs = torch.cat([first, free_probs], dim=-1)
    term_probs = occupancy * torch.cumprod(free_probs, dim=-1)

    # using escape probability
    # occupancy = occupancy[:, :-1]
    # first = torch.ones([occupancy.shape[0], 1], device=occupancy.device)
    # free_probs = (1. - occupancy + 1e-10)
    # free_probs = torch.cat([first, free_probs], dim=-1)
    # last = torch.ones([occupancy.shape[0], 1], device=occupancy.device)
    # occupancy = torch.cat([occupancy, last], dim=-1)
    # term_probs = occupancy * torch.cumprod(free_probs, dim=-1)

    return term_probs

def render(termination, vals, dim=-1):
    weighted_vals = termination * vals
    render = weighted_vals.sum(dim=dim)

    return render

def render_loss(render, gt, loss="L1", normalise=False):
    residual = render - gt
    if loss == "L2":
        loss_mat = residual ** 2
    elif loss == "L1":
        loss_mat = torch.abs(residual)
    else:
        print("loss type {} not implemented!".format(loss))

    if normalise:
        loss_mat = loss_mat / gt

    return loss_mat

def reduce_batch_loss(loss_mat, var=None, avg=True, mask=None, loss_type="L1"):
    mask_num = torch.sum(mask, dim=-1)
    if (mask_num == 0).any():   # no valid sample, return 0 loss
        loss = torch.zeros_like(loss_mat)
        if avg:
            loss = torch.mean(loss, dim=-1)
        return loss
    if var is not None:
        eps = 1e-4
        if loss_type == "L2":
            information = 1.0 / (var + eps)
        elif loss_type == "L1":
            information = 1.0 / (torch.sqrt(var) + eps)

        loss_weighted = loss_mat * information
    else:
        loss_weighted = loss_mat

    if avg:
        if mask is not None:
            loss = (torch.sum(loss_weighted, dim=-1)/(torch.sum(mask, dim=-1)+1e-10))
            if (loss > 100000).any():
                print("loss explode")
                exit(-1)
        else:
            loss = torch.mean(loss_weighted, dim=-1).sum()
    else:
        loss = loss_weighted

    return loss

def make_3D_grid(occ_range=[-1., 1.], dim=256, device="cuda:0", transform=None, scale=None):
    t = torch.linspace(occ_range[0], occ_range[1], steps=dim, device=device)
    grid = torch.meshgrid(t, t, t)
    grid_3d = torch.cat(
        (grid[0][..., None],
         grid[1][..., None],
         grid[2][..., None]), dim=3
    )

    if scale is not None:
        grid_3d = grid_3d * scale
    if transform is not None:
        R1 = transform[None, None, None, 0, :3]
        R2 = transform[None, None, None, 1, :3]
        R3 = transform[None, None, None, 2, :3]

        grid1 = (R1 * grid_3d).sum(-1, keepdim=True)
        grid2 = (R2 * grid_3d).sum(-1, keepdim=True)
        grid3 = (R3 * grid_3d).sum(-1, keepdim=True)
        grid_3d = torch.cat([grid1, grid2, grid3], dim=-1)

        trans = transform[None, None, None, :3, 3]
        grid_3d = grid_3d + trans

    return grid_3d
