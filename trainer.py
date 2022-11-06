import torch
import model
import embedding
import render_rays
import numpy as np
import vis
from tqdm import tqdm

class Trainer:
    def __init__(self):
        # todo set param
        self.device = "cuda:0"
        self.hidden_feature_size = 32 #32   # 256 for iMAP, 128 for seperate bg
        self.obj_scale = 3. # 10 for bg and iMAP
        self.n_unidir_funcs = 5
        self.emb_size1 = 21*(3+1)+3
        self.emb_size2 = 21*(5+1)+3 - self.emb_size1
        self.learning_rate = 0.001
        self.weight_decay = 0.013

        self.load_network()
        self.optimiser = torch.optim.AdamW(
            self.fc_occ_map.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay)
        self.optimiser.add_param_group({"params": self.pe.parameters(),
                                        "lr": self.learning_rate,
                                        "weight_decay": self.weight_decay})


    def load_network(self):
        self.fc_occ_map = model.OccupancyMap(
            self.emb_size1,
            self.emb_size2,
            hidden_size=self.hidden_feature_size
        )
        self.fc_occ_map.apply(model.init_weights).to(self.device)
        self.pe = embedding.UniDirsEmbed(max_deg=self.n_unidir_funcs, scale=self.obj_scale).to(self.device)

    def meshing(self, bound, grid_dim=256):
        occ_range = [-1., 1.]
        range_dist = occ_range[1] - occ_range[0]
        scene_scale_np = bound.extent / (range_dist * 0.9)
        scene_scale = torch.from_numpy(scene_scale_np).float().to(self.device)
        transform_np = np.eye(4, dtype=np.float32)
        transform_np[:3, 3] = bound.center
        transform_np[:3, :3] = bound.R
        # transform_np = np.linalg.inv(transform_np)  #
        transform = torch.from_numpy(transform_np).to(self.device)
        grid_pc = render_rays.make_3D_grid(occ_range=occ_range, dim=grid_dim, device=self.device,
                                           scale=scene_scale, transform=transform).view(-1, 3)
        ret = self.eval_points(grid_pc)
        if ret is None:
            return None

        occ, _ = ret
        mesh = vis.marching_cubes(occ.view(grid_dim, grid_dim, grid_dim).cpu().numpy())
        if mesh is None:
            print("marching cube failed")
            return None

        # Transform to [-1, 1] range
        mesh.apply_translation([-0.5, -0.5, -0.5])
        mesh.apply_scale(2)

        # Transform to scene coordinates
        mesh.apply_scale(scene_scale_np)
        mesh.apply_transform(transform_np)

        vertices_pts = torch.from_numpy(np.array(mesh.vertices)).float().to(self.device)
        ret = self.eval_points(vertices_pts)
        _, color = ret
        mesh_color = color * 255
        vertex_colors = mesh_color.detach().squeeze(0).cpu().numpy().astype(np.uint8)
        mesh.visual.vertex_colors = vertex_colors

        return mesh


    def eval_points(self, points, chunk_size=100000):
        # 256^3 = 16777216
        alpha, color = [], []
        n_chunks = int(np.ceil(points.shape[0] / chunk_size))
        with torch.no_grad():
            for k in tqdm(range(n_chunks)): # 2s/it 1000000 pts
                chunk_idx = slice(k * chunk_size, (k + 1) * chunk_size)
                embedding_k = self.pe(points[chunk_idx, ...])
                alpha_k, color_k = self.fc_occ_map(embedding_k)
                alpha.extend(alpha_k.detach().squeeze())
                color.extend(color_k.detach().squeeze())
        alpha = torch.stack(alpha)
        color = torch.stack(color)

        occ = render_rays.occupancy_activation(alpha).detach()
        if occ.max() == 0:
            print("no occ")
            return None
        return (occ, color)











