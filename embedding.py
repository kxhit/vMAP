import torch
import numpy as np

def positional_encoding(
    tensor,
    B_layer=None,
    num_encoding_functions=6,
    scale=10.
):
    if B_layer is not None:
        embedding_gauss = B_layer(tensor / scale)
        embedding_gauss = torch.sin(embedding_gauss)
        embedding = embedding_gauss
    else:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        n_repeat = num_encoding_functions * 2 + 1
        embedding = tensor[..., None, :].repeat(1, 1, n_repeat, 1) / scale
        even_idx = np.arange(1, num_encoding_functions + 1) * 2
        odd_idx = even_idx - 1

        frequency_bands = frequency_bands[None, None, :, None]

        embedding[:, :, even_idx, :] = torch.cos(
            frequency_bands * embedding[:, :, even_idx, :])
        embedding[:, :, odd_idx, :] = torch.sin(
            frequency_bands * embedding[:, :, odd_idx, :])

        n_dim = tensor.shape[-1]
        embedding = embedding.view(
            embedding.shape[0], embedding.shape[1], n_repeat * n_dim)
        # print("embedding ", embedding.shape)
        embedding = embedding.squeeze(0)

    return embedding

class UniDirsEmbed(torch.nn.Module):
    def __init__(self, min_deg=0, max_deg=2, scale=2.):
        super(UniDirsEmbed, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.n_freqs = max_deg - min_deg + 1
        self.tensor_scale = torch.tensor(scale, requires_grad=False)

        dirs = torch.tensor([
        0.8506508, 0, 0.5257311,
        0.809017, 0.5, 0.309017,
        0.5257311, 0.8506508, 0,
        1, 0, 0,
        0.809017, 0.5, -0.309017,
        0.8506508, 0, -0.5257311,
        0.309017, 0.809017, -0.5,
        0, 0.5257311, -0.8506508,
        0.5, 0.309017, -0.809017,
        0, 1, 0,
        -0.5257311, 0.8506508, 0,
        -0.309017, 0.809017, -0.5,
        0, 0.5257311, 0.8506508,
        -0.309017, 0.809017, 0.5,
        0.309017, 0.809017, 0.5,
        0.5, 0.309017, 0.809017,
        0.5, -0.309017, 0.809017,
        0, 0, 1,
        -0.5, 0.309017, 0.809017,
        -0.809017, 0.5, 0.309017,
        -0.809017, 0.5, -0.309017
        ]).reshape(-1, 3)

        self.B_layer = torch.nn.Linear(3, 21, bias=False)
        self.B_layer.weight.data = dirs

        frequency_bands = 2.0 ** torch.linspace(self.min_deg, self.max_deg, self.n_freqs)
        self.register_buffer("frequency_bands", frequency_bands, persistent=False)
        self.register_buffer("scale", self.tensor_scale, persistent=True)

    def forward(self, x):
        tensor = x / self.scale   # functorch needs buffer, otherwise changed
        proj = self.B_layer(tensor)
        proj_bands = proj[..., None, :] * self.frequency_bands[None, None, :, None]
        xb = proj_bands.view(list(proj.shape[:-1]) + [-1])
        # embedding = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
        embedding = torch.sin(xb * np.pi)
        embedding = torch.cat([tensor] + [embedding], dim=-1)
        # print("emb size ", embedding.shape)
        return embedding