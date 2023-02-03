import torch
import torch.nn as nn

def init_weights(m, init_fn=torch.nn.init.xavier_normal_):
    if type(m) == torch.nn.Linear:
        init_fn(m.weight)


def fc_block(in_f, out_f):
    return torch.nn.Sequential(
        torch.nn.Linear(in_f, out_f),
        torch.nn.ReLU(out_f)
    )


class OccupancyMap(torch.nn.Module):
    def __init__(
        self,
        emb_size1,
        emb_size2,
        hidden_size=256,
        do_color=True,
        hidden_layers_block=1
    ):
        super(OccupancyMap, self).__init__()
        self.do_color = do_color
        self.embedding_size1 = emb_size1
        self.in_layer = fc_block(self.embedding_size1, hidden_size)

        hidden1 = [fc_block(hidden_size, hidden_size)
                   for _ in range(hidden_layers_block)]
        self.mid1 = torch.nn.Sequential(*hidden1)
        # self.embedding_size2 = 21*(5+1)+3 - self.embedding_size # 129-66=63 32
        self.embedding_size2 = emb_size2
        self.cat_layer = fc_block(
            hidden_size + self.embedding_size1, hidden_size)

        # self.cat_layer = fc_block(
        #     hidden_size , hidden_size)

        hidden2 = [fc_block(hidden_size, hidden_size)
                   for _ in range(hidden_layers_block)]
        self.mid2 = torch.nn.Sequential(*hidden2)

        self.out_alpha = torch.nn.Linear(hidden_size, 1)

        if self.do_color:
            self.color_linear = fc_block(self.embedding_size2 + hidden_size, hidden_size)
            self.out_color = torch.nn.Linear(hidden_size, 3)

        # self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x,
                noise_std=None,
                do_alpha=True,
                do_color=True,
                do_cat=True):
        fc1 = self.in_layer(x[...,:self.embedding_size1])
        fc2 = self.mid1(fc1)
        # fc3 = self.cat_layer(fc2)
        if do_cat:
            fc2_x = torch.cat((fc2, x[...,:self.embedding_size1]), dim=-1)
            fc3 = self.cat_layer(fc2_x)
        else:
            fc3 = fc2
        fc4 = self.mid2(fc3)

        alpha = None
        if do_alpha:
            raw = self.out_alpha(fc4)   # todo ignore noise
            if noise_std is not None:
                noise = torch.randn(raw.shape, device=x.device) * noise_std
                raw = raw + noise

            # alpha = self.relu(raw) * scale    # nerf
            alpha = raw * 10. #self.scale     # unisurf

        color = None
        if self.do_color and do_color:
            fc4_cat = self.color_linear(torch.cat((fc4, x[..., self.embedding_size1:]), dim=-1))
            raw_color = self.out_color(fc4_cat)
            color = self.sigmoid(raw_color)

        return alpha, color



