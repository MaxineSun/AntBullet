import gym
from stable_baselines3 import PPO

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

import pybullet
import pybullet_envs

from network.baseline_network import baseline_network
# from network.graph_network import graph_network_noaction, graph_network_action


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
            device = 'cpu',
            args=None,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = graph_network_noaction(11,
                                             output_size=1,
                                             network_shape=args["network_shape"],
                                             define_std=False,
                                             device=device,
                                             args=args)

        # Value network
        self.value_net = baseline_network(11, None, args["network_shape"], False, device, args["transform"])

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net.forward(features), self.value_net.forward(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            # para : str = {},
            *args,
            **kwargs,
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        # self.para = para
        self.args = kwargs["args"]
        self.action_net = graph_network_action(11,
                                             output_size=1,
                                             network_shape=self.args["network_shape"],
                                             device=kwargs["device"],
                                             args=self.args)
        self.value_net = nn.Linear(self.features_dim, 1).to(kwargs["device"])
        self.device0 = kwargs["device"]

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim,
                                           self.features_dim,
                                           self.features_dim,
                                           self.device0,
                                           self.args
                                           )


def obsTransform(obs28):
    # input size: 1 * 28
    obs = obs28[0]
    print(obs)

    device = obs.device
    obs_list = []
    for i in range(9):
        obs_i = [obs[:8]]

        if i != 8:
            obs_ii = []
            obs_ii.append(obs[8 + 2 * i])
            obs_ii.append(obs[8 + 2 * i + 1])
            obs_ii.append(obs[24 + i // 2])
            obs_ii = th.stack(obs_ii)
            obs_i.append(obs_ii)
            obs_i = th.cat(obs_i)

            obs_list.append(obs_i)

        else:
            obs_i.append(th.zeros(3).to(device))
            obs_i = th.cat(obs_i)
            obs_list.append(obs_i)

    obs_new = th.stack(obs_list, dim=0)
    print(obs_new.shape)
    return obs_new


class flow_pass(nn.Module):
    def __init__(self):
        super(flow_pass, self).__init__()

    def forward(self, x):
        return x


if __name__ == "__main__":

    env = gym.make("AntBulletEnv-v0")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    model = PPO("MlpPolicy", env, verbose=1, device=device, tensorboard_log="experiments")
    model.learn(total_timesteps=1000000)
    # model.learn(total_timesteps=4097)
    model.save("models/baseline")
    # model = PPO.load("models", env, device)




    """
    obs = env.reset()
    for i in range(1000):
        if (i + 1) % 100 == 0:
            print(str(i + 1))
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(reward)
        env.render()
        if done:
            obs = env.reset()

    env.close()
    """


    """
    node_type = [1, 2, 1, 2, 1, 2, 1, 2, 0]
    edge_attr = th.LongTensor(
        [[9, 9, 9, 9, 0, 2, 4, 8, 0, 2, 4, 8, 1, 3, 5, 7],
         [0, 2, 4, 8, 1, 3, 5, 7, 9, 9, 9, 9, 0, 2, 4, 8]]).to(device)

    args = {
        "input_size": 15,
        "output_size": 1,
        "hidden_size": 64,
        "network_shape": [64, 64],
        "output_type": "unified",
        "node_type": node_type,  ## need to specify
        "state_update_func": "lstm",
        "state_update_type": "shared",
        "edge_update_type": "unified",
        "node_agg_fn_type": "sum",
        "propagation_step": 4,
        "edge_attr": edge_attr,
        "transform": obsTransform()
    }

    # model = PPO("MlpPolicy", env, verbose=1)


    # custom_policy = CustomActorCriticPolicy(env, para=para)
    model = PPO(CustomActorCriticPolicy, env, verbose=1, args = args, device = device)


    model.learn(total_timesteps=10000)

    obs = env.reset()
    for i in range(1000):
        if (i + 1) % 100 == 0:
            print(str(i + 1))
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()
    """