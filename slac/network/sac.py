import torch
from torch import nn

from slac.network.initializer import initialize_weight
from slac.utils import build_mlp, reparameterize


class GaussianPolicy(torch.jit.ScriptModule):
    """
    Policy parameterized as diagonal gaussian distribution.
    """

    def __init__(self, action_shape, num_sequences, img_feature_dim, hidden_units=(256, 256)):
        super(GaussianPolicy, self).__init__()

        # NOTE: Conv layers are shared with the latent model.
        self.net = build_mlp(
            input_dim=num_sequences * img_feature_dim + (num_sequences - 1) * action_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=nn.ReLU(inplace=True),
        ).apply(initialize_weight)

    @torch.jit.script_method
    def forward(self, feature_action):
        means = torch.chunk(self.net(feature_action), 2, dim=-1)[0]
        return torch.tanh(means)

    @torch.jit.script_method
    def sample(self, feature_action):
        mean, log_std = torch.chunk(self.net(feature_action), 2, dim=-1)
        log_std = torch.clip(log_std, -20, 2)
        action, log_pi = reparameterize(mean, log_std)
        return action, log_pi


class TwinnedQNetwork(torch.jit.ScriptModule):
    """
    Twinned Q networks.
    """

    def __init__(
        self,
        action_shape,
        z_dim,
        hidden_units=(256, 256),
    ):
        super(TwinnedQNetwork, self).__init__()

        self.net1 = build_mlp(
            input_dim=2072,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=nn.ReLU(inplace=True),
        ).apply(initialize_weight)
        self.net2 = build_mlp(
            input_dim=2072,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=nn.ReLU(inplace=True),
        ).apply(initialize_weight)

    @torch.jit.script_method
    def forward(self, feature, action):
        x = torch.cat([feature, action], dim=1)
        return self.net1(x), self.net2(x)
