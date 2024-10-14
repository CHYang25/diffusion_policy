from typing import Type, Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import math

logger = logging.getLogger(__name__)

def init_layer(module, gain=math.sqrt(2)):
    with torch.no_grad():
        nn.init.orthogonal_(module.weight.data, gain=gain)
        nn.init.constant_(module.bias.data, 0)
    return module

def make_module(input_dim, output_dim, hidden, activation: Type[nn.Module] = nn.ReLU):
    n_in = input_dim
    l_hidden = []
    for h in hidden:
        l_hidden.append(init_layer(torch.nn.Linear(n_in, h)))
        l_hidden.append(activation())
        n_in = h
    l_hidden.append(init_layer(torch.nn.Linear(n_in, output_dim), gain=0.1))
    return torch.nn.Sequential(*l_hidden)


def make_module_list(input_dim, output_dim, hidden, n_net, activation: Type[nn.Module] = nn.ReLU):
    return nn.ModuleList([make_module(input_dim, output_dim, hidden, activation) for _ in range(n_net)])


class OptionNetwork(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 option_dim: int,
                 is_shared: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.option_dim = option_dim
        self.opt_hidden_dim = (64, 64)
        self.pi_hidden_dim = (64, 64)
        self.log_clamp = (-20., 0.)

        self.option_policy = make_module_list(
            input_dim=self.input_dim, 
            output_dim=self.option_dim,
            hidden=self.opt_hidden_dim,
            n_net=self.option_dim+1,
        )

        self.a_log_std = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(1, self.output_dim, dtype=torch.float32).fill_(0.)) for _ in range(self.option_dim)
        ])

        self.policy = make_module_list(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden=self.pi_hidden_dim,
            n_net=self.option_dim,
        )

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
    
    def a_mean_logstd(self, st, ot=None):
        # ot: None or long(N x 1)
        # ot: None for all c, return (N x opt_dim x output_dim); else return (N x output_dim)
        # s: N x obs_dim, c: N x 1, c should always < opt_dim
        mean = torch.stack([m(st) for m in self.policy], dim=-2)
        logstd = torch.stack([m.expand_as(mean[:, 0, :]) for m in self.a_log_std], dim=-2)
        if ot is not None:
            ind = ot.view(-1, 1, 1).expand(-1, 1, self.output_dim)
            mean = mean.gather(dim=-2, index=ind).squeeze(dim=-2)
            logstd = logstd.gather(dim=-2, index=ind).squeeze(dim=-2)
        return mean.clamp(-10, 10), logstd.clamp(self.log_clamp[0], self.log_clamp[1])

    def switcher(self, s):
        return torch.stack([m(s) for m in self.option_policy], dim=-2)
