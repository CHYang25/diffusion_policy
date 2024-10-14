from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.option.option_model import OptionNetwork
import math

class OptionBCLowdimPolicy(BaseLowdimPolicy):
    def __init__(self,
                 horizon,
                 obs_dim,
                 option_dim,
                 action_dim,
                 deterministic: bool = False):
        super().__init__()

        self.model = OptionNetwork(
            input_dim=obs_dim,
            output_dim=action_dim,
            option_dim=option_dim,        
            is_shared=False
        )
        self.normalizer = LinearNormalizer()
        self.obs_dim = obs_dim
        self.opt_dim = option_dim
        self.action_dim = action_dim
        self.deterministic = deterministic
        self.ot_1 = None

    # auxiliary functions
    def log_trans(self, st, ot_1=None):
        # ot_1: long(N x 1) or None
        # ot_1: None: direct output p(ot|st, ot_1): a (N x ot_1 x ot) array where ot is log-normalized
        unnormed_pcs = self.model.switcher(st)
        log_pcs = unnormed_pcs.log_softmax(dim=-1)
        if ot_1 is None:
            return log_pcs
        else:
            return log_pcs.gather(dim=-2, index=ot_1.view(-1, 1, 1).expand(-1, 1, self.opt_dim)).squeeze(dim=-2)

    def log_prob_action(self, st, ot, at):
        # if c is None, return (N x opt_dim x 1), else return (N x 1)
        mean, logstd = self.model.a_mean_logstd(st, ot)
        if ot is None:
            at = at.view(-1, 1, self.action_dim)
        return (-((at - mean).square()) / (2 * (logstd * 2).exp()) - logstd - math.log(math.sqrt(2 * math.pi))).sum(dim=-1, keepdim=True)

    def log_prob_option(self, st, ot_1, ot):
        log_opt = self.log_trans(st, ot_1)
        return log_opt.gather(dim=-1, index=ot)


    # interface functions
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        
        st = obs_dict['obs']
        st = st.squeeze(dim=1)
        ot_1 = self.ot_1

        # sample option
        # log of the probability of options
        log_opt = self.log_trans(st, ot_1)
        if ot_1 is None:
            log_opt = log_opt[:, -1, :]
        if self.deterministic:
            ot = log_opt.argmax(dim=-1, keepdim=True)
        else:
            ot = F.gumbel_softmax(log_opt, hard=False).multinomial(1).long()
        
        # sample action
        action_mean, action_log_std = self.model.a_mean_logstd(st, ot)
        if self.deterministic:
            at = action_mean
        else:
            eps = torch.empty_like(action_mean).normal_()
            at = action_mean + action_log_std.exp() * eps

        self.ot_1 = ot

        result = {
            'action': at.unsqueeze(dim=1),
            'option': ot.unsqueeze(dim=1)
        }        

        return result


    def compute_loss(self, batch, lambda_entropy: float = 1.0):
        # TODO: assuming that we need normalization
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']
        obs = obs.squeeze(dim=1)
        action = action.squeeze(dim=1)

        log_acts = self.log_prob_action(obs, None, action).view(-1, self.opt_dim)
        log_opts = self.log_trans(obs, None)
        
        log_opt0 = log_opts[0, -1] # the last option #
        log_opts = log_opts[1:, :-1]
        log_alpha = [log_opt0 + log_acts[0]]
        for log_opt, log_act in zip(log_opts, log_acts[1:]):
            log_alpha_t = (log_alpha[-1].unsqueeze(dim=-1) + log_opt).logsumexp(dim=0) + log_act
            log_alpha.append(log_alpha_t)

        log_alpha = torch.stack(log_alpha)
        entropy = -(log_opts * log_opts.exp()).sum(dim=-1).mean()
        log_p = (log_alpha.softmax(dim=-1).detach() * log_alpha).sum()
        # the second term is for regularization
        loss = -log_p - lambda_entropy * entropy
        return loss
