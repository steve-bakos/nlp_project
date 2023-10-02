import logging
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F


class MultipleMappings(torch.nn.Module):
    """
    Torch module for using orthogonal mappings to map several language
    representation to a pivot one
    """

    def __init__(
        self,
        nb_langs,
        dim,
        map_beta=0.001,
        reset_with_identity=True,
        bias=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.nb_langs = nb_langs
        self.dim = dim
        self.reset_with_identity = reset_with_identity
        self.map_beta = map_beta

        self.mapping = Parameter(torch.empty((nb_langs, dim, dim), device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(torch.empty((nb_langs, dim), device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.nb_langs):
            if self.reset_with_identity:
                self.mapping.data[i] = torch.eye(self.dim)
            else:
                init.kaiming_uniform_(self.mapping.data[i], a=math.sqrt(5))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, right_emb, pair_id):
        """
        Apply an orthogonal mapping to each element of right_emb according
        to the pair_id (language id) provided
        """
        res = torch.zeros_like(right_emb)
        for i in range(right_emb.shape[0]):
            if int(pair_id[i][0]) == -1:
                res[i] = right_emb[i]
                continue
            res[i] = F.linear(
                right_emb[i],
                self.mapping[pair_id[i][0]],
                bias=self.bias[pair_id[i][0]] if self.bias is not None else None,
            )
        return res

    def orthogonalize(self):
        """
        perform an update step to orthogonalize the mapping matrices
        """
        W = self.mapping.data
        for i in range(self.nb_langs):
            W[i].copy_(
                (1 + self.map_beta) * W[i] - self.map_beta * W[i].mm(W[i].transpose(0, 1).mm(W[i]))
            )
