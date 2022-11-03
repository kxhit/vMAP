import torch
import render_rays

ce_loss = torch.nn.CrossEntropyLoss(reduce=False)
mse_loss = torch.nn.MSELoss(reduction="none")

def step_batch_loss(alpha, color, gt_depth, gt_color, sem_labels, mask_depth, z_vals,
                    color_scaling=5.0, opacity_scaling=10.0):
    mask_obj = sem_labels == 1
    # invalid_obj_mask = sem == -1    # no decided obj mask todo should we supervise mask=-1 col loss?
    # sem_scaling = 8.0
    # color_scaling = self.color_scaling #5.0
    # opacity_scaling = self.opacity_scaling  # 10.0
    alpha = alpha.squeeze(dim=-1)
    color = color.squeeze(dim=-1)
    # sem = sem.squeeze(dim=-1)
    occupancy = render_rays.occupancy_activation(alpha)
    termination = render_rays.occupancy_to_termination(occupancy, is_batch=True)  # shape [num_batch, num_ray, points_per_ray]
    print("termination ", termination.shape)
    print("z_vals ", z_vals.shape)
    # from obj-nerf opacity loss    todo render_rays.render(termination, occupancy)
    render_opacity = torch.sum(termination, dim=-1)
    # render_opacity = render_rays.render(termination, occupancy)
    # print("render_opacity ", render_opacity)
    render_depth = render_rays.render(termination, z_vals)
    # print("render depth ", render_depth[mask_depth & mask_obj])
    # print("gt depth ", gt_depth[mask_depth & mask_obj])
    # print("render_time ", time.time()-render_time)
    diff_sq = (z_vals - render_depth[..., None]) ** 2
    # print("square time ", time.time() - render_time)
    var = render_rays.render(termination, diff_sq)
    var = var.detach()  # todo bug!!!

    render_color = render_rays.render(termination[..., None], color, dim=-2)
    # render_sem = render_rays.render(termination[..., None], sem, dim=-2)

    # 2D depth loss: only on valid depth & mask
    # var_masked = var
    # [mask_depth & mask_obj]
    loss_all = torch.zeros_like(render_depth)
    loss_depth_raw = render_rays.render_loss(render_depth, gt_depth, normalise=False)
    # print("loss_depth raw ", loss_depth_raw.shape)  # Mxn_per_frame
    loss_depth = torch.mul(loss_depth_raw, mask_depth & mask_obj)   # keep dim but set invalid element be zero
    loss_all += loss_depth
    loss_depth = render_rays.reduce_batch_loss(loss_depth, var=var, avg=True, mask=mask_depth & mask_obj)
    # print("loss_depth ", loss_depth)

    # 2D color loss: only on obj mask   [mask_obj]
    loss_col_raw = render_rays.render_loss(render_color, gt_color, normalise=False)  # only masked area do col loss
    # print("loss_col raw ", loss_col_raw.shape)
    loss_col = torch.mul(loss_col_raw.sum(-1), mask_obj)
    loss_all += loss_col / 3. * color_scaling
    loss_col = render_rays.reduce_batch_loss(loss_col, var=None, avg=True, mask=mask_obj)
    # print("loss_col ", loss_col)

    # all samples do sem loss
    # print("sem_labels ", sem_labels.dtype)
    # print("if in ", -1 in sem_labels)
    mask_sem = sem_labels != 2 # todo for background?
    mask_sem = mask_sem.detach()# todo
    #celoss assumes feature dim is always the second dimension
    # loss_sem_raw = ce_loss(render_sem.permute(0,2,1), sem_labels)  # avoid mask edge, label==-1, low confident
    # # print("loss_sem_raw ", loss_sem_raw.shape)
    # loss_sem = torch.mul(loss_sem_raw, mask_sem)
    # loss_all += loss_sem * sem_scaling
    # loss_sem = render_rays.reduce_batch_loss(loss_sem, var=None, avg=True, mask=mask_sem)

    # do weak supervision in mask out area
    # only do extra supervision on background mask out area == gt_occlusion
    do_sup_maskout = False
    if do_sup_maskout:
        scale_mask_out = 1.0
        gt_occlusion = True # todo debug
        loss_depth_bg = torch.mul(loss_depth_raw, mask_depth & ~mask_obj & gt_occlusion)
        loss_all += loss_depth_bg * scale_mask_out
        loss_depth_bg = render_rays.reduce_batch_loss(loss_depth_bg, var=var, avg=True, mask=mask_depth & ~mask_obj & gt_occlusion)

        loss_depth += loss_depth_bg * scale_mask_out

    do_sup_opacity = True  # True
    if do_sup_opacity:
        loss_opacity = mse_loss(torch.clamp(render_opacity, 0, 1), mask_obj.float().detach()) # todo here for bg
        # loss_opacity = mse_loss(torch.clamp(render_opacity, 0, 1), mask_sem.long()) # todo here for bg

        # print("render_opacity ", render_opacity)

        loss_opacity = torch.mul(loss_opacity, mask_sem)  # but ignore -1 edges  # todo bug!!!
        loss_all += loss_opacity * opacity_scaling
        loss_opacity = render_rays.reduce_batch_loss(loss_opacity, var=None, avg=True, mask=mask_sem)   # todo var
        # print("loss_opacity ", loss_opacity)
        loss_depth += loss_opacity * opacity_scaling

    # print("loss_sem ", loss_sem)
    # print("loss_depth ", loss_depth)
    # print("loss_col ", loss_col)
    # print("sem_loss ", sem_loss)
    # print("sem time ", time.time() - render_time)
    # loss for bp
    l_batch = loss_depth + loss_col * color_scaling #+ loss_sem * sem_scaling  # todo param
    # print("loss batch ", l_batch)
    l_random = l_batch.sum()
    # print("fn loss time ", time.time()-render_time)

    return l_random, loss_all.detach()