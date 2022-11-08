import torch
import cv2
import numpy as np
import os

import loss
from sampling_manager import *
import utils
import open3d
import dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import image_transforms
import trainer

# vmap
from functorch import vmap

from time  import perf_counter
import json

"""
n_sample_per_step:  120                                          
batch_input_pcs: torch.Size([20, 120, 10, 3]), torch.float32                                                                       
batch_gt_depth: torch.Size([20, 120]), torch.float32
batch_gt_rgb: torch.Size([20, 120, 3]), torch.float32                                                                              
batch_depth_mask: torch.Size([20, 120]), torch.bool
batch_obj_mask: torch.Size([20, 120]), torch.uint8                                                                                 
batch_sampled_z: torch.Size([20, 120, 10]), torch.float32
"""

# todo verify on replica
if __name__ == "__main__":
    ###################################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    # init todo arg parser class
    # hyper param for trainer
    log_dir = "logs/room0_vmap_kf10_bg32"
    training_device = "cuda:0"
    # data_device = "cpu"
    data_device ="cuda:0"
    # vis_device = "cuda:1"
    imap_mode = False #False
    training_strategy = "forloop" # "forloop" "vmap"
    # training_strategy = "vmap" # "forloop" "vmap"
    win_size = 5
    n_iter_per_frame = 20
    n_samples_per_frame = 120 // 5 #120 // 5
    n_sample_per_step = n_samples_per_frame * win_size
    min_depth = 0.
    max_depth = 10.
    depth_scale = 1/1000.

    # param for dataset
    bbox_scale = 0.2

    # init obj_dict
    obj_dict = {}

    # init for training
    learning_rate = 0.001
    weight_decay = 0.013
    AMP = False
    if AMP:
        scaler = torch.cuda.amp.GradScaler()    # amp https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/

    results = {"forloop": [], "vmap":[], "n_objs":[]}

    n_to_eval = np.linspace(1, 200, num=21, dtype=np.uint8)
    # n_to_eval = np.linspace(1, 100, num=11, dtype=np.uint8)
    results["n_objs"] = n_to_eval.tolist()

    for training_strategy in ["forloop", "vmap"]:
        for num_objs in n_to_eval:
            print(f"Evaluating {num_objs} objects training using {training_strategy}.")

            trainers = []
            fc_models, pe_models = [], []
            optimiser = torch.optim.AdamW([torch.autograd.Variable(torch.tensor(0))], lr=learning_rate, weight_decay=weight_decay)

            for i in range(num_objs):
                temp_trainer = trainer.Trainer()
                trainers.append(temp_trainer)

                optimiser.add_param_group({"params": temp_trainer.fc_occ_map.parameters(), "lr": learning_rate, "weight_decay": weight_decay})
                optimiser.add_param_group({"params": temp_trainer.pe.parameters(), "lr": learning_rate, "weight_decay": weight_decay})

                if training_strategy == "vmap":
                    fc_models.append(temp_trainer.fc_occ_map)
                    pe_models.append(temp_trainer.pe)

            if training_strategy == "vmap":
                fc_model, fc_param, fc_buffer = utils.update_vmap(fc_models, optimiser)
                pe_model, pe_param, pe_buffer = utils.update_vmap(pe_models, optimiser)

            batch_input_pcs = torch.rand(num_objs, n_sample_per_step, 10, 3,device=training_device)
            batch_gt_depth = torch.rand(num_objs, n_sample_per_step, device=training_device)
            batch_gt_rgb = torch.rand(num_objs, n_sample_per_step, 3, device=training_device)
            batch_sampled_z = torch.rand(num_objs, n_sample_per_step, 10, dtype=torch.float32, device=training_device)
            batch_depth_mask = torch.cuda.FloatTensor(num_objs, n_sample_per_step, device=training_device).uniform_() > 0.8
            batch_obj_mask = torch.randint(high=2, size=(num_objs, n_sample_per_step), dtype=torch.uint8, device=training_device).to(torch.uint8)

            # warm up
            if num_objs == n_to_eval[0]:
                print("warming up!")
                for _ in range(5):
                    _ = trainers[0].pe(batch_input_pcs[0])

            start_time = perf_counter()
            for iter_step in range(n_iter_per_frame):
                if training_strategy == "forloop":
                    # for loop training
                    batch_alpha = []
                    batch_color = []
                    for idx, trainer_ in enumerate(trainers):
                        embedding_k = trainer_.pe(batch_input_pcs[idx])
                        alpha_k, color_k = trainer_.fc_occ_map(embedding_k)
                        batch_alpha.append(alpha_k)
                        batch_color.append(color_k)

                    batch_alpha = torch.stack(batch_alpha)
                    batch_color = torch.stack(batch_color)

                elif training_strategy == "vmap":
                    # batched training
                    # set_trace()
                    batch_embedding = vmap(pe_model)(pe_param, pe_buffer, batch_input_pcs)
                    batch_alpha, batch_color = vmap(fc_model)(fc_param, fc_buffer, batch_embedding)

                else:
                    print("training strategy {} is not implemented ".format(training_strategy))
                    exit(-1)

                # step loss
                batch_loss, _ = loss.step_batch_loss(batch_alpha, batch_color,
                                        batch_gt_depth.detach(), batch_gt_rgb.detach(),
                                        batch_obj_mask.detach(), batch_depth_mask.detach(),
                                        batch_sampled_z.detach())

                if AMP:
                    # Scales the loss, and calls backward()
                    # to create scaled gradients
                    scaler.scale(batch_loss).backward()
                    # Unscales gradients and calls
                    # or skips optimizer.step()
                    scaler.step(optimiser)
                    # Updates the scale for next iteration
                    scaler.update()
                else:
                    batch_loss.backward()
                    optimiser.step()

                optimiser.zero_grad(set_to_none=True)

            end_time = perf_counter()
            step_time = (end_time - start_time)  / n_iter_per_frame
            print(f"{step_time}")

            results[training_strategy].append(step_time)
    # print(results)
    # print(json.dumps(results, indent=4))

    results_np = np.stack((n_to_eval, results["forloop"], results["vmap"]), axis=0)
    np.save("n_objs_eval.npy", results_np)
