import torch
import numpy as np

def ray_dirs_C_sparse(B, indices_h, indices_w, fx, fy, cx, cy, device, depth_type='z'):
    z = torch.ones(indices_w.shape[0], device=device)
    x = (indices_w - cx) / fx
    y = (indices_h - cy) / fy

    dirs = torch.stack((x, y, z), dim=-1)
    # print("dirs shape ", dirs.shape)
    if depth_type == 'euclidean':
        norm = torch.norm(dirs, dim=-1)
        dirs = dirs * (1. / norm)[:, :, :, None]

    return dirs

def ray_dirs_C(B, H, W, fx, fy, cx, cy, device, depth_type='z'):
    c, r = torch.meshgrid(torch.arange(W, device=device),
                          torch.arange(H, device=device))
    c, r = c.t().float(), r.t().float()
    size = [B, H, W]

    C = torch.empty(size, device=device)
    R = torch.empty(size, device=device)
    C[:, :, :] = c[None, :, :]
    R[:, :, :] = r[None, :, :]

    z = torch.ones(size, device=device)
    x = (C - cx) / fx
    y = (R - cy) / fy

    dirs = torch.stack((x, y, z), dim=3)
    # print("dirs shape ", dirs.shape)    # [1, 680, 1200, 3] [1, 113, 200, 3]
    if depth_type == 'euclidean':
        norm = torch.norm(dirs, dim=3)
        dirs = dirs * (1. / norm)[:, :, :, None]

    return dirs

def origin_dirs_W(T_WC, dirs_C):
    R_WC = T_WC[:, :3, :3]
    dirs_W = (R_WC * dirs_C[..., None, :]).sum(dim=-1)
    origins = T_WC[:, :3, -1]

    return origins, dirs_W

def origin_dirs_W_batch(T_WC, dirs_C):
    R_WC = T_WC[:, :, :3, :3]
    dirs_W = (R_WC * dirs_C[..., None, :]).sum(dim=-1)
    origins = T_WC[:, :, :3, -1]

    return origins, dirs_W

# def normal_bins_sampling(depth,n_bins, n_rays, device, delta=0.1): # niceslam
#     t_vals_surface = torch.linspace(
#         0., 1., steps=n_bins).to(device)
#     z_vals = 0.95 * depth * \
#                      (1. - t_vals_surface) + 1.05 * \
#                      depth * (t_vals_surface)
#     return z_vals

def normal_bins_sampling(depth,n_bins, n_rays, device, delta=0.1):
    bins = torch.normal(0, delta/3., [n_rays, n_bins], device=device).sort().values
    bins = torch.clip(bins, -delta, delta)
    z_vals = depth + bins
    return z_vals

def stratified_bins(min_depth,
                    max_depth,
                    n_bins,
                    n_rays,
                    device="cpu"):
    if torch.is_tensor(max_depth):  # batch of depths from gt_depth
        bin_limits_scale = torch.linspace(
            0,
            1,
            n_bins + 1,
            device=device,
        )
        # print("n_rays ", n_rays)
        # print("bin_limits_scale ", bin_limits_scale.shape)
        # print("depth shape ", (max_depth - min_depth).reshape(-1,1).shape)
        # bins_limits_scale = bin_limits_scale * (max_depth - min_depth).reshape(-1, 1)
        # bins_limits_scale = bins_limits_scale + min_depth
        bins_limits_scale = bin_limits_scale * (max_depth - min_depth).reshape(-1,1) + min_depth
        lower_limits_scale = bins_limits_scale[:, :-1]
        bin_length_scale = (max_depth - min_depth) / (n_bins)
        increments_scale = torch.rand(n_rays, n_bins, device=device) * bin_length_scale.reshape(-1,1)
        # increments_scale = 0.5 * torch.ones(n_rays, n_bins, device=device) * bin_length_scale.reshape(-1,1) # (n_rays,n_bins)
        # print("increments_scale ", increments_scale.shape)  # [N, n_bins]
        z_vals_scale = lower_limits_scale + increments_scale
        # print("z_vals_scale ", z_vals_scale.shape)  # [N, n_bins]
        return z_vals_scale

    bin_limits = torch.linspace(
        min_depth,
        max_depth,
        n_bins + 1,
        device=device,
    )
    # print("bin limits ", bin_limits.shape)  # [n_bins+1]
    lower_limits = bin_limits[:-1]
    bin_length = (max_depth - min_depth) / (n_bins)
    increments = torch.rand(n_rays, n_bins, device=device) * bin_length
    # increments = 0.5 * torch.ones(n_rays, n_bins, device=device) * bin_length
    # print("increments ", increments.shape)  # [N, n_bins]
    z_vals = lower_limits[None, :] + increments
    # print("z_vals ", z_vals.shape)  # [N, n_bins]
    # print("== ", torch.sum(z_vals-z_vals_scale))
    return z_vals


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


def render_loss(render, gt_depth, loss="L1", normalise=False):
    residual = render - gt_depth
    if loss == "L2":
        loss_mat = residual ** 2
    elif loss == "L1":
        loss_mat = torch.abs(residual)

    if normalise:
        loss_mat = loss_mat / gt_depth

    return loss_mat

def occupancy_loss(occ_with_depth, gt_depth, e=0.1):
    loss = 0
    occ_depth = occ_with_depth[1,:,:]
    occ = occ_with_depth[0,:,:]
    gt_surface_depth = gt_depth.reshape(-1,1).repeat(1, occ.shape[-1])
    # print("occ ", occ.shape)
    # print("gt ", gt_surface_depth.shape)
    assert occ.shape == occ_depth.shape
    assert occ.shape == gt_surface_depth.shape
    pred_occ_empty = occ[occ_depth < torch.clip(gt_surface_depth - e, 0)]  # e controls how near the depth surface
    # pred_occ_empty = occ    # occ out side of mask all be 0
    # pred_occ_dist = occ[occ_depth > gt_surface_depth + e]   # occupancy from depth surface +e to boundary
    # print("occ ", pred_occ_empty.shape)
    # occ between cam and depth surface should be 0
    # loss = torch.sum(pred_occ_empty)
    loss = torch.mean(pred_occ_empty)    # todo do we need avg?
    # loss_empty = torch.mean(pred_occ_empty)
    # loss_dist = torch.mean(pred_occ_dist)
    # loss = loss_empty + loss_dist
    return loss

# @torch.jit.script
# def reduce_l1_loss_info_fused(loss, var):
#     n = loss.shape[0]
#     eps = 1e-4
#     loss_red = (loss * (1.0 / (torch.sqrt(var) + eps))).sum() / n
#
#     return loss_red


def reduce_loss(loss_mat, var=None, avg=True, loss_type="L1"):
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
        # print("loss weighted ", loss_weighted.shape)
        loss = torch.mean(loss_weighted, dim=0).sum()
    else:
        loss = loss_weighted

    return loss

def reduce_batch_loss(loss_mat, var=None, avg=True, mask=None, loss_type="L1"):
    mask_num = torch.sum(mask, dim=-1)
    if (mask_num == 0).any():
        print("notice 0 mask num ", mask_num)
        loss = torch.zeros_like(loss_mat)
        if avg:
            if mask is not None:
                loss = torch.mean(loss, dim=-1)
            else:
                loss = torch.mean(loss, dim=-1).sum()
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
            # loss = torch.masked_select(loss_weighted, mask).mean(dim=-1)
            # print("loss ", loss)
            # if (loss > 100000).any():
            #     print("loss explode")
            #     exit(-1)
            # loss = loss.sum()
        else:
            loss = torch.mean(loss_weighted, dim=-1).sum()
    else:
        loss = loss_weighted

    return loss

def mix_zvals_alpha(alpha,
                    alpha_fine,
                    z_vals,
                    z_vals_fine,
                    color=None,
                    color_fine=None,
                    sem=None,
                    sem_fine=None):

    z_vals_mix, arg_inds = torch.sort(
        torch.cat((z_vals, z_vals_fine), dim=-1), dim=-1)
    alpha_mix = torch.cat((alpha, alpha_fine), dim=-1)

    inds_1 = torch.arange(
        arg_inds.shape[0]).repeat_interleave(arg_inds.shape[1])
    inds_2 = arg_inds.view(-1)

    alpha_mix = alpha_mix[inds_1, inds_2].view(z_vals_mix.shape)

    color_mix = None
    if color is not None:
        color_mix = torch.cat((color, color_fine), dim=-2)
        color_shape = color_mix.shape
        color_mix = color_mix[inds_1, inds_2, :].view(color_shape)

    sem_mix = None
    if sem is not None:
        sem_mix = torch.cat((sem, sem_fine), dim=-2)
        sem_shape = sem_mix.shape
        sem_mix = sem_mix[inds_1, inds_2, :].view(sem_shape)

    return z_vals_mix, alpha_mix, color_mix, sem_mix

def sample_fine(weights,
                bin_limits,
                n_bins_fine,
                origins,
                dirs_W,
                occ_map,
                B_layer=None,
                noise_std=None,
                do_color=True,
                do_sem=False,
                obj_c=None
                ):
    z_vals_fine = sample_pdf(bin_limits, weights, n_bins_fine)
    pc_fine = origins[:, None, :] + \
        (dirs_W[:, None, :] * z_vals_fine[:, :, None])
    if obj_c is not None:
        pc_fine -= obj_c
    points_embedding_fine = B_layer(pc_fine)
    alpha_fine, color_fine, sem_fine = occ_map(points_embedding_fine)
    alpha_fine = alpha_fine.squeeze(dim=-1)

    if color_fine is not None:
        color_fine = color_fine.squeeze(dim=-1)
    if sem_fine is not None:
        sem_fine = sem_fine.squeeze(dim=-1)

    return z_vals_fine, alpha_fine, color_fine, sem_fine, pc_fine

def render_images_chunks(T_WC,
                         min_depth, max_depth,
                         n_embed_funcs, n_bins,
                         occ_map,
                         B_layer=None,
                         H=None, W=None, fx=None, fy=None, cx=None, cy=None,
                         grad=False, dirs_C=None,
                         do_fine=False,
                         do_var=True,
                         do_color=False,
                         do_sem=False,
                         n_bins_fine=None,
                         noise_std=None,
                         chunk_size=10000):

    if dirs_C is None:
        B_cam = T_WC.shape[0]
        dirs_C = ray_dirs_C(
            B_cam, H, W, fx, fy, cx, cy, T_WC.device, depth_type='z')
        dirs_C = dirs_C.view(B_cam, -1, 3)

    n_pts = dirs_C.shape[1]
    n_chunks = int(np.ceil(n_pts / chunk_size))
    depths = []
    vars = []
    cols = []
    sems = []
    for n in range(n_chunks):
        start = n * chunk_size
        end = start + chunk_size
        chunk = dirs_C[:, start:end, :]

        depth, var, col, sem = render_images(
            T_WC,
            min_depth, max_depth,
            n_embed_funcs, n_bins,
            occ_map,
            B_layer=B_layer,
            H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy,
            grad=grad,
            dirs_C=chunk,
            do_fine=do_fine,
            do_var=do_var,
            do_color=do_color,
            do_sem=do_sem,
            n_bins_fine=n_bins_fine,
            noise_std=noise_std
        )
        depths.append(depth.detach())
        if do_var:
            vars.append(var.detach())
        if do_sem:
            sems.append(sem.detach())
        if do_color:
            cols.append(col.detach())

    depths = torch.cat(depths, dim=0)
    if do_var:
        vars = torch.cat(vars, dim=0)
    if do_sem:
        sems = torch.cat(sems, dim=0)
    if do_color:
        cols = torch.cat(cols, dim=0)

    return depths, vars, cols, sems

def render_normals(T_WC,
                   render_depth,
                   occ_map,
                   dirs_C,
                   n_embed_funcs,
                   origins_dirs,
                   B_layer=None,
                   noise_std=None,
                   do_mip=False,
                   radius=None,
                   obj_c=None
                   ):
    # origins, dirs_W = origin_dirs_W(T_WC, dirs_C)
    origins, dirs_W = origins_dirs
    origins = origins.view(-1, 3)
    dirs_W = dirs_W.view(-1, 3)

    pc = origins + (dirs_W * (render_depth[:, None]))
    if obj_c is not None:
        pc -= obj_c.to(pc.device)
    pc.requires_grad_(True)
    points_embedding = embedding.positional_encoding(
        pc, B_layer, num_encoding_functions=n_embed_funcs)

    alpha, _, _ = occ_map(
        points_embedding, noise_std=noise_std,
        do_color=False, do_sem=False
    )

    d_points = torch.ones_like(
        alpha, requires_grad=False, device=alpha.device)
    points_grad = torch.autograd.grad(
        outputs=alpha,
        inputs=pc,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if do_mip:
        points_grad = points_grad.squeeze(1)

    surface_normals_W = -points_grad / \
        (points_grad.norm(dim=1, keepdim=True) + 1e-4)
    R_CW = T_WC[:, :3, :3].inverse()
    surface_normals_C = (R_CW * surface_normals_W[..., None, :]).sum(dim=-1)

    L = dirs_C.squeeze(0)
    L = -L / L.norm(dim=1, keepdim=True)
    diffuse = (L * surface_normals_C).sum(1)

    return surface_normals_W, diffuse



def render_images(T_WC_sample,
                  min_depth, max_depth,
                  n_embed_funcs, n_bins,
                  occ_map,
                  B_layer=None,
                  H=None, W=None, fx=None, fy=None, cx=None, cy=None,
                  grad=False, dirs_C=None,
                  do_fine=False,
                  do_var=True,
                  do_color=False,
                  do_sem=False,
                  n_bins_fine=None,
                  noise_std=None,
                  z_vals_limits=None,
                  obj_batch=None,
                  obj_mask=None,
                  T_WO=None,
                  depth_sample=None,
                  obj_c=None,
                  pe=None,
                  occlusion_batch=None,
                  obj_code=None
                  ):
    gt_depth = depth_sample.clone()
    # gt_depth =None
    T_WC = T_WC_sample.clone()
    if T_WO is not None:    # todo which coordinate is used for optim? how to re-render?
        # print("T_WO ", T_WO.shape)
        T_OW = torch.inverse(T_WO).to(T_WC.device)
        T_WC = T_OW @ T_WC # @ T_OW # todo deepcopy
        # todo try torch.linalg.solve(A, B) == A.inv() @ B in pytorch 1.10
        # T_WC = torch.linalg.solve(T_WO, T_WC)

    B_cam = T_WC.shape[0]
    with torch.set_grad_enabled(grad):
        if dirs_C is None:
            dirs_C = ray_dirs_C(
                B_cam, H, W, fx, fy, cx, cy, T_WC.device, depth_type='z')
            dirs_C = dirs_C.view(B_cam, -1, 3)
        # print("dirs_C ", dirs_C.shape)  # (b, 120x68, 3)    H/reduce_factor x W/reduce_factor

        # rays in world coordinate
        origins, dirs_W = origin_dirs_W(T_WC, dirs_C)

        origins = origins.view(-1, 3)
        dirs_W = dirs_W.view(-1, 3)
        n_rays = dirs_W.shape[0]

        # use gt_depth to guide z_vals sampling
        band_eps = 0.1
        if gt_depth is not None:
            if z_vals_limits is None:
                if obj_batch is not None and obj_mask is not None:   # has obj mask
                    invalid_depth_mask = gt_depth == 0
                    gt_depth[invalid_depth_mask] = max_depth - band_eps
                    # make samplings in space from cam to depth surface
                    min_bound = torch.full_like(gt_depth, min_depth) # all bound
                    max_bound = torch.clamp_min(gt_depth + band_eps, min_depth + band_eps)  # todo
                    z_vals_cam2surface = stratified_bins(
                        min_bound.reshape(-1, 1), max_bound.reshape(-1, 1),
                        5, n_rays, T_WC.device)

                    # sampling for invalid depth rays and mask out
                    invalid_mask = invalid_depth_mask | ~obj_mask    # todo which strategy?
                    invalid_n_rays = torch.sum(invalid_mask)
                    z_vals_invalid = None
                    if invalid_n_rays > 0:
                        z_vals_invalid = stratified_bins(
                            min_bound[invalid_mask].reshape(-1, 1), max_bound[invalid_mask].reshape(-1, 1),
                            n_bins, invalid_n_rays, T_WC.device)
                    # # sampling for invalid depth rays
                    # invalid_n_rays = torch.sum(invalid_depth_mask)
                    # z_vals_invalid_depth = None
                    # if invalid_n_rays > 0:
                    #     z_vals_invalid_depth = stratified_bins(
                    #         min_bound[invalid_depth_mask].reshape(-1, 1), max_bound[invalid_depth_mask].reshape(-1, 1),
                    #         n_bins, invalid_n_rays, T_WC.device)
                    # sampling around valid depth surface
                    sampling_method = "normal"  # stratified or normal
                    if sampling_method == "stratified":
                        # min_bound = torch.full_like(gt_depth, min_depth)
                        # min_bound[obj_mask] = torch.clamp_min(gt_depth[obj_mask] - 0.2, min_depth)  # only obj mask bound
                        min_bound = torch.clamp_min(gt_depth - band_eps, min_depth)    # all bound
                        max_bound = torch.clamp_min(gt_depth + band_eps, min_depth + band_eps)  # todo
                        z_vals_limits = stratified_bins(
                            min_bound.reshape(-1,1), max_bound.reshape(-1,1),
                            n_bins, n_rays, T_WC.device)
                    elif sampling_method == "normal":
                        z_vals_limits = normal_bins_sampling(gt_depth.reshape(-1,1), n_bins, n_rays, T_WC.device, delta=band_eps)
                        # print("z_vals_limit ", z_vals_limits)
                    else:
                        print("sampling method not implemented ", sampling_method)
                        exit(-1)
                    if z_vals_invalid is not None:    # replace samplings for invalid depth rays
                        z_vals_limits[invalid_mask] = z_vals_invalid
                    z_vals_limits = torch.cat([z_vals_cam2surface, z_vals_limits], dim=-1).sort().values
                    # print("zvals_sort ", z_vals_limits)
                else:   # obj_id None
                    gt_depth[gt_depth == 0] = max_depth
                    # back project samples
                    z_vals_limits = stratified_bins(  # todo guide by particles z_vals
                        min_depth, gt_depth + 0.2,
                        n_bins, n_rays, T_WC.device)
        else:
            if z_vals_limits is None:
                # back project samples
                z_vals_limits = stratified_bins(
                    min_depth, max_depth,
                    n_bins, n_rays, T_WC.device)

        z_vals = 0.5 * (z_vals_limits[..., 1:] + z_vals_limits[..., :-1])
        # print("z_vals ", z_vals)
        pc = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])
        if obj_c is not None:
            pc -= obj_c.to(pc.device)
        if B_layer:
            points_embedding = embedding.positional_encoding(
                pc, B_layer, num_encoding_functions=n_embed_funcs)
        else:
            points_embedding = pe(pc)
        alpha, color, sem = occ_map(
            points_embedding, noise_std=noise_std,
            do_color=do_color, do_sem=do_sem, view_dirs=dirs_W, xyz=pc, obj_code=obj_code
        )

        alpha = alpha.squeeze(dim=-1)
        if do_color:
            color = color.squeeze(dim=-1)
        else:
            color = None
        if do_sem:
            sem = sem.squeeze(dim=-1)
        else:
            sem = None
        occupancy = alpha_to_occupancy(z_vals_limits, dirs_W, alpha)
        termination = occupancy_to_termination(occupancy)   # shape [num_ray, points_per_ray]
        occ_with_depth = torch.cat((occupancy.unsqueeze(0),
                                    z_vals_limits[:,:-1].unsqueeze(0)), dim=0)

        if do_fine:
            # print("do fine ", do_fine)
            z_vals_fine, alpha_fine, color_fine, sem_fine, _ = sample_fine(
                termination,
                z_vals_limits,
                min_depth,
                max_depth,  # todo
                n_bins,
                n_bins_fine,
                origins,
                dirs_W,
                n_embed_funcs,
                occ_map,
                B_layer,
                noise_std=noise_std,
                do_color=do_color,
                do_sem=do_sem
            )

            z_vals_mix, alpha_mix, color_mix, sem_mix = mix_zvals_alpha(
                alpha,
                alpha_fine,
                z_vals,
                z_vals_fine,
                color,
                color_fine,
                sem,
                sem_fine
            )

            occupancy_mix = alpha_to_occupancy(z_vals_mix, dirs_W, alpha_mix, add_last=True)
            termination_mix = occupancy_to_termination(occupancy_mix)

            occ_with_depth = torch.cat((occupancy_mix.unsqueeze(0),
                                        z_vals_mix.unsqueeze(0)), dim=0)
            # print("alpha ", alpha_mix.shape)
            # print("------------------------------------------")
            # print("occ_with_depth ", occ_with_depth.shape)
            # print(occ_with_depth)
            # print("termination ", termination_mix.shape)
            # print(termination_mix)
        else:
            termination_mix = termination
            z_vals_mix = z_vals
            sem_mix = sem
            color_mix = color
        render_depth = render(termination_mix, z_vals_mix)
        # print("z_vals_mix ", z_vals_mix)
        # print("render_depth ", render_depth)
        # print("gt_depth ", gt_depth)
        var = None
        if do_var:
            diff_sq = (z_vals_mix - render_depth[:, None]) ** 2
            var = render(termination_mix, diff_sq)
        # print("var ", var)
        render_color = None
        if do_color:
            render_color = render(
                termination_mix[..., None], color_mix, dim=-2)
        render_sem = None
        if do_sem:
            render_sem = render(
                termination_mix[..., None], sem_mix, dim=-2)

    if grad == True:
        return render_depth, var, render_color, render_sem, occ_with_depth, alpha, color, sem, z_vals_mix, pc   # todo zvals zvals limit
    else:
        return render_depth, var, render_color, render_sem, occ_with_depth


def entropy(dist):
    return -(dist * torch.log(dist)).sum(2)


def map_color(sem, colormap):
    return (colormap[..., :, :] * sem[..., None]).sum(-2)


def sample_pdf(bin_limits, weights, num_samples, det=False):
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
    )

    u = torch.rand(
        list(cdf.shape[:-1]) + [num_samples],
        dtype=weights.dtype,
        device=weights.device,
    )

    # Invert CDF
    u = u.contiguous()
    cdf = cdf.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack((below, above), dim=-1)  # (batchsize, num_samples, 2)

    matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])

    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bin_limits.unsqueeze(
        1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def loss_approx(full_loss, binary_masks, W, H, factor=8):
    w_block = W // factor
    h_block = H // factor
    loss_approx = full_loss.view(-1, factor, h_block, factor, w_block)
    loss_approx = loss_approx.sum(dim=(2, 4))
    actives = binary_masks.view(-1, factor, h_block, factor, w_block)
    actives = actives.sum(dim=(2, 4))
    actives[actives == 0] = 1.0
    loss_approx = loss_approx / actives

    return loss_approx


# merge same kf to speed up
def sample_data(win_idxs, n_per_frame, kf_info_list, obj_id, depth_gt, obj_gt, color_gt, dirs_C_gt, T_WC_gt,
                  fx, fy, cx, cy, depth_eps, max_d, min_d, n_bins, n_bins_cam2surface, obj_center, do_track_obj=False, device="cpu"):
    # n_total = self.iters_per_frame*self.n_per_frame # 20*1200//5=4800
    n_total = n_per_frame
    win_idxs = np.array(win_idxs)
    win_kf_ids = np.unique(win_idxs)
    for _, idx in enumerate(win_kf_ids):    # try to reduce loop nums
        ii = np.where(idx == win_idxs)[0]
        n = ii.shape[0]
    # for i, idx in enumerate(self.win_idxs):
        kf_select = kf_info_list[idx]
        # if kf_select.T_WO_track is None and do_track_obj:  # init pose
        #     kf_select.T_WO_track = torch.eye(4).unsqueeze(0)

        # if obj_id == 0:
        #     w_min, h_min, w_max, h_max = 0, 0, self.W, self.H
        # else:
        # w_min, h_min, w_max, h_max = kf_select.bbox_dict[int(obj_id)][0]
        # todo debug
        w_min, h_min, w_max, h_max = 0, 0, 1200, 680

        # sample from uv
        indices_h = torch.randint(h_min, h_max, (n_total, n))
        indices_w = torch.randint(w_min, w_max, (n_total, n))
        # # todo random scale and round
        # random_uv = torch.rand((n_total, 2))
        # indices_h = torch.round(random_uv[:,0]*(h_max-h_min)+h_min).long()
        # indices_w = torch.round(random_uv[:,1]*(w_max-w_min)+w_min).long()
        # print("kf_select.depth_batch[0, indices_h, indices_w] ", kf_select.depth_batch[0, indices_h, indices_w].shape)
        depth_gt.view(-1, n_total)[ii] = kf_select.depth_batch[0, indices_h, indices_w].view(n_total, n).T
        # print(self.depth_gt.shape)
        color_gt.view(-1, n_total, 3)[ii] = kf_select.im_batch[0, indices_h, indices_w].view(n_total, n, 3).permute(1,0,2)
        obj_gt.view(-1, n_total)[ii] = kf_select.obj_batch[0, indices_h, indices_w].view(n_total, n).T
        dirs_C_gt.view(-1, n_total, 3)[ii] = ray_dirs_C_sparse(0, indices_h.view(-1), indices_w.view(-1), fx, fy, cx, cy,
                                                      indices_h.device).view(n_total, n, 3).permute(1,0,2)

        T_WO = None
        if kf_select.T_WO_track is not None and do_track_obj:
            # T_WO = self.init_joint_poses(kf_select.T_WO_track, kf_select.trans_delta_obj, kf_select.w_obj, obj=True)[0]
            T_WO = kf_select.T_WO_track[0]
        elif kf_select.T_WO_batch is not None:
            T_WO = kf_select.T_WO_batch[0, obj_id]

        if T_WO is not None:
            T_OW = torch.inverse(T_WO)
            T_WC = kf_select.T_WC_batch[0]
            T_WC = T_OW @ T_WC
            T_WC_gt.view(-1, n_total,4,4)[ii] = T_WC.unsqueeze(0).repeat_interleave(n_total*n, dim=0).view(n_total, n,4,4).permute(1,0,2,3)
        else:
            T_WC_gt.view(-1, n_total, 4,4)[ii] = kf_select.T_WC_batch.repeat_interleave(n_total*n, dim=0).view(n_total, n,4,4).permute(1,0,2,3)

    depth_mask = depth_gt != 0
    obj_mask = obj_gt == obj_id

# def sample_points(T_WC_gt, dirs_C_gt, depth_eps, depth_gt, max_d, min_d, n_bins, n_bins_cam2surface, obj_mask, obj_center, device="cpu", vis_mode=False):

    # rays in world coordinate
    origins, dirs_W = origin_dirs_W(T_WC_gt, dirs_C_gt)
    origins = origins.view(-1, 3)
    dirs_W = dirs_W.view(-1, 3)
    n_rays = dirs_W.shape[0]
    eps = depth_eps
    gt_depth = depth_gt.clone()
    max_depth = min(max_d, torch.max(gt_depth))
    min_depth = min_d

    # use gt_depth to guide z_vals sampling
    invalid_depth_mask = ~depth_mask
    gt_depth[invalid_depth_mask] = max_depth
    # make samplings in space from cam to depth surface
    min_bound = torch.full_like(gt_depth, min_depth)  # all bound
    max_bound = torch.clamp_min(gt_depth + eps, min_depth + eps)  # todo   too close might have prob?
    # min -> d-eps
    z_vals_cam2surface = stratified_bins(
        min_bound[depth_mask].reshape(-1, 1), max_bound[depth_mask].reshape(-1, 1),
        n_bins_cam2surface, torch.sum(depth_mask), device)

    # sampling for invalid depth rays and mask out area
    # invalid_mask = invalid_depth_mask  # | ~obj_mask    # todo which strategy?
    # todo depth guidance for invalid depth part?
    invalid_depth_n_rays = torch.sum(invalid_depth_mask)
    z_vals_invalid_depth = None
    if invalid_depth_n_rays > 0:
        # min -> max
        z_vals_invalid_depth = stratified_bins(
            min_depth, max_depth, n_bins + n_bins_cam2surface, invalid_depth_n_rays, device)
    # todo sample for mask out area with early stopped ray
    z_vals_invalid_obj = None
    invalid_obj_mask = ~obj_mask & depth_mask
    invalid_obj_n_rays = torch.sum(invalid_obj_mask)
    if invalid_obj_n_rays > 0:
        min_bound = torch.clamp_min(gt_depth - eps, min_depth)  # all bound
        max_bound = torch.clamp_min(gt_depth + 0.02, min_depth + eps)  # todo
        # d-eps -> d+0.02       early stop for opacity loss
        z_vals_invalid_obj = stratified_bins(
            min_bound[invalid_obj_mask].reshape(-1, 1), max_bound[invalid_obj_mask].reshape(-1, 1),
            n_bins, invalid_obj_n_rays, device)

    # sampling around valid depth surface
    z_vals_surface = None
    surface_mask = obj_mask & depth_mask
    surface_n_rays = torch.sum(surface_mask)
    # d-eps -> d+eps
    sampling_method = "normal"  # stratified or normal
    if sampling_method == "stratified":
        min_bound = torch.clamp_min(gt_depth - eps, min_depth)  # all bound
        max_bound = torch.clamp_min(gt_depth + eps, min_depth + eps)  # todo
        z_vals_surface = stratified_bins(
            min_bound[surface_mask].reshape(-1, 1), max_bound[surface_mask].reshape(-1, 1),
            n_bins, surface_n_rays, device)
    elif sampling_method == "normal":
        z_vals_surface = normal_bins_sampling(gt_depth[surface_mask].reshape(-1, 1), n_bins,
                                                          surface_n_rays, device,
                                                          delta=eps)
    else:
        print("sampling method not implemented ", sampling_method)
        exit(-1)
    # if vis_mode:
    z_vals_cat = torch.zeros([n_rays, n_bins_cam2surface + n_bins], dtype=torch.float)
    if z_vals_invalid_depth is not None:
        z_vals_cat[invalid_depth_mask, :] = z_vals_invalid_depth
    if z_vals_cam2surface is not None:
        z_vals_cat[depth_mask, :n_bins_cam2surface] = z_vals_cam2surface
    if z_vals_invalid_obj is not None:
        z_vals_cat[invalid_obj_mask, n_bins_cam2surface:] = z_vals_invalid_obj
    if z_vals_surface is not None:
        z_vals_cat[surface_mask, n_bins_cam2surface:] = z_vals_surface

    z_vals = 0.5 * (z_vals_cat[..., 1:] + z_vals_cat[..., :-1])
    input_pcs = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])
    # # vis
    # pc = open3d.geometry.PointCloud()
    # pc.points = open3d.utility.Vector3dVector(self.input_pcs.numpy().reshape(-1, 3))
    # open3d.visualization.draw_geometries([pc])  # pc should fill the bbox   layered n rays
    if obj_center is not None:
        input_pcs -= obj_center

    return obj_id, depth_gt, color_gt, obj_gt, depth_mask, obj_mask, input_pcs, z_vals, dirs_W

if __name__ == "__main__":
    # depths = [[0.32, 0.38, 0.68, 0.82, 0.87, 1.99],
    #           [0.17, 0.29, 0.71, 0.76, 1.2, 1.25]]
    # depths_arr = torch.tensor(depths)
    # term_probs = torch.ones(depths_arr.shape) * 0.5
    # depth_range = torch.tensor([0.1, 2.])
    # n_bins = 19
    # bin_term_probs = torch.ones([depths_arr.shape[0], n_bins]) * 0.5
    # # must be all 0.5
    # updated_bin_probs = update_bin_probs(
    #     bin_term_probs, depths_arr, term_probs, depth_range)

    #-------------------------------------------------------------
    bins_limits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    weights = torch.tensor([[0.2, 0.3, 0.1, 0.1, 0.5],
                            [0.2, 0.3, 0.1, 0.1, 0.5]])
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)
    n_samples = 20

    sample_pdf(bins_limits, weights, n_samples)

    import ipdb
    ipdb.set_trace()
