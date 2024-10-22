from typing import Type, Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import math

from diffusion_policy.model.spirl.modules.subnetworks import Predictor, BaseProcessingLSTM
from diffusion_policy.model.spirl.modules.mdn import MDN, GMM
from diffusion_policy.model.spirl.modules.flow_models import ConditionedFlowModel
from diffusion_policy.model.spirl.modules.variational_inference import get_fixed_prior, ProbabilisticModel, MultivariateGaussian, Gaussian
from diffusion_policy.model.spirl.modules.losses import NLL, KLDivLoss
from diffusion_policy.model.spirl.modules.layers import LayerBuilderParams

from diffusion_policy.model.spirl.utils.general_utils import AttrDict, batch_apply, get_clipped_optimizer
from diffusion_policy.model.spirl.utils.pytorch_utils import get_constant_parameter, TensorModule, RAdam

logger = logging.getLogger(__name__)

class SpirlPriorNetwork(nn.Module, ProbabilisticModel):
    """Skill embedding AutoEncoder + Skill prior model for SPIRL (closed loop version)"""
    def __init__(self, hp, 
            n_action_steps, 
            n_obs_steps):
        
        nn.Module.__init__(self)
        ProbabilisticModel.__init__(self)

        # set class properties
        self._hp = hp
        self.state_dim = hp.state_dim
        self.action_dim = hp.action_dim
        self.latent_dim = hp.nz_vae
        self.prior_input_size = hp.state_dim
        self.n_rollout_steps = hp.n_rollout_steps
        self.enc_size = hp.state_dim

        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

        # spirl needs builder
        self._hp.builder = LayerBuilderParams(self._hp.use_convs, self._hp.normalization)

        # build networks
        self.q = torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=hp.action_dim+hp.state_dim, out_dim=hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2)
        )
        self.decoder = Predictor(self._hp,
                                 input_size=self.enc_size + self._hp.nz_vae,
                                 output_size=self._hp.action_dim,
                                 mid_size=self._hp.nz_mid_prior)
        self.p = nn.ModuleList([self._build_prior_net() for _ in range(self._hp.n_prior_nets)])
        self.log_sigma = get_constant_parameter(0., learnable=False) # constant Parameter

        # optionally: optimize beta with dual gradient descent
        if self._hp.target_kl is not None:
            self._log_beta = TensorModule(torch.zeros(1, requires_grad=True, device=self._hp.device))
            self._beta_opt = self._get_beta_opt()

    # property functions
    @property
    def beta(self):
        return self._log_beta().exp()[0].detach() if self._hp.target_kl is not None else self._hp.kl_div_weight

    # auxiliary functions
    def _build_prior_net(self):
        """Supports building Gaussian, GMM and Flow prior networks. Default is Gaussian skill prior."""
        if self._hp.learned_prior_type == 'gmm':
            return torch.nn.Sequential(
                Predictor(self._hp, input_size=self.prior_input_size, output_size=self._hp.nz_mid,
                          num_layers=self._hp.num_prior_net_layers, mid_size=self._hp.nz_mid_prior),
                MDN(input_size=self._hp.nz_mid, output_size=self._hp.nz_vae,
                    num_gaussians=self._hp.n_gmm_prior_components)
            )
        elif self._hp.learned_prior_type == 'flow':
            return ConditionedFlowModel(self._hp, input_dim=self.prior_input_size, output_dim=self._hp.nz_vae,
                                        n_flow_layers=self._hp.num_prior_net_layers)
        else:
            return Predictor(self._hp, input_size=self.prior_input_size, output_size=self._hp.nz_vae * 2,
                             num_layers=self._hp.num_prior_net_layers, mid_size=self._hp.nz_mid_prior)

    def _get_beta_opt(self):
        return get_clipped_optimizer(filter(lambda p: p.requires_grad, self._log_beta.parameters()),
                                     lr=3e-4, optimizer_type=RAdam, betas=(0.9, 0.999), gradient_clip=None)

    def _learned_prior_input(self, inputs):
        return inputs.obs[:, 0]

    def _run_inference(self, inputs):
        # run inference with state sequence conditioning
        inf_input = torch.cat((inputs.action, self._get_seq_enc(inputs)), dim=-1)
        return MultivariateGaussian(self.q(inf_input)[:, -1])
    
    def _get_seq_enc(self, inputs):
        return inputs.obs[:, :-1].repeat(1, 10, 1)

    def _compute_learned_prior(self, prior_mdl, inputs):
        if self._hp.learned_prior_type == 'gmm':
            return GMM(*prior_mdl(inputs))
        elif self._hp.learned_prior_type == 'flow':
            return prior_mdl(inputs)
        else:
            return MultivariateGaussian(prior_mdl(inputs))

    def compute_learned_prior(self, inputs, first_only=False):
        """Splits batch into separate batches for prior ensemble, optionally runs first or avg prior on whole batch.
           (first_only, avg == True is only used for RL)."""
        if first_only:
            return self._compute_learned_prior(self.p[0], inputs)

        assert inputs.shape[0] % self._hp.n_prior_nets == 0
        per_prior_inputs = torch.chunk(inputs, self._hp.n_prior_nets)
        prior_results = [self._compute_learned_prior(prior, input_batch)
                         for prior, input_batch in zip(self.p, per_prior_inputs)]

        return type(prior_results[0]).cat(*prior_results, dim=0)
    
    def _compute_learned_prior_loss(self, model_output):
        if self._hp.nll_prior_train:
            loss = NLL(breakdown=0)(model_output.q_hat, model_output.z_q.detach())
        else:
            loss = KLDivLoss(breakdown=0)(model_output.q.detach(), model_output.q_hat)
        # aggregate loss breakdown for each of the priors in the ensemble
        loss.breakdown = torch.stack([chunk.mean() for chunk in torch.chunk(loss.breakdown, self._hp.n_prior_nets)])
        return loss
    
    def _update_beta(self, kl_div):
        """Updates beta with dual gradient descent."""
        assert self._hp.target_kl is not None
        beta_loss = self._log_beta().exp() * (self._hp.target_kl - kl_div).detach().mean()
        self._beta_opt.zero_grad()
        beta_loss.backward()
        self._beta_opt.step()

    @staticmethod
    def _compute_total_loss(losses):
        total_loss = torch.stack([loss[1].value * loss[1].weight for loss in
                                  filter(lambda x: x[1].weight > 0, losses.items())]).sum()
        return AttrDict(value=total_loss)

    # Major functions
    def decode(self, z, cond_inputs, steps, inputs=None):
        assert inputs is not None       # need additional state sequence input for full decode
        seq_enc = self._get_seq_enc(inputs)
        decode_inputs = torch.cat((seq_enc[:, :steps], z[:, None].repeat(1, steps, 1)), dim=-1)
        return batch_apply(decode_inputs, self.decoder)

    def forward(self, inputs, use_learned_prior=False):
        """Forward pass of the SPIRL model.
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        :arg use_learned_prior: if True, decodes samples from learned prior instead of posterior, used for RL
        """
        output = AttrDict()
        inputs.action = inputs.action[:, :self.n_action_steps]
        inputs.obs = inputs.obs[:, :self.n_obs_steps]

        inputs.observations = inputs.action

        # run inference
        output.q = self._run_inference(inputs)

        # compute (fixed) prior
        output.p = get_fixed_prior(output.q)

        # infer learned skill prior
        output.q_hat = self.compute_learned_prior(self._learned_prior_input(inputs))
        if use_learned_prior:
            output.p = output.q_hat     # use output of learned skill prior for sampling

        # sample latent variable
        output.z = output.p.sample() if self._sample_prior else output.q.sample()
        output.z_q = output.z.clone() if not self._sample_prior else output.q.sample()   # for loss computation

        # decode
        assert inputs.action.shape[1] == self._hp.n_rollout_steps
        output.reconstruction = self.decode(output.z,
                                            cond_inputs=self._learned_prior_input(inputs),
                                            steps=self._hp.n_rollout_steps,
                                            inputs=inputs)
        return output

    def compute_loss(self, inputs):
        """Loss computation of the SPIRL model.
        :arg model_output: output of SPIRL model forward pass
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        """
        model_output = self.forward(inputs)
        losses = AttrDict()

        # reconstruction loss, assume unit variance model output Gaussian
        losses.rec_mse = NLL(self._hp.reconstruction_mse_weight) \
            (Gaussian(model_output.reconstruction, torch.zeros_like(model_output.reconstruction)),
             inputs.action)

        # KL loss
        losses.kl_loss = KLDivLoss(self.beta)(model_output.q, model_output.p)

        # learned skill prior net loss
        losses.q_hat_loss = self._compute_learned_prior_loss(model_output)

        # Optionally update beta
        if self.training and self._hp.target_kl is not None:
            self._update_beta(losses.kl_loss.value)

        losses.total = self._compute_total_loss(losses)
        return losses