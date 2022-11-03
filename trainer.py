import torch
import model
import embedding

class Trainer:
    def __init__(self):
        # todo set param
        self.device = "cuda:0"
        self.hidden_feature_size = 32   # 256 for iMAP, 128 for seperate bg
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

