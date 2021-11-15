import os
import pickle

import numpy as np
import torch
from torch.optim import Adam

from slac.buffer import ReplayBuffer
from slac.network import GaussianPolicy, LatentModel, TwinnedQNetwork
from slac.utils import create_feature_actions, grad_false, soft_update


def save_pickle(data, myfile):
    with open(myfile, "wb") as f:
        pickle.dump(data, f)


class SlacAlgorithm:
    """
    Stochactic Latent Actor-Critic(SLAC).

    Paper: https://arxiv.org/abs/1907.00953
    """

    def __init__(
        self,
        state_shape = (3,224,224),
        action_shape = (3,),
        tactile_shape = (6,),
        action_repeat = 1,
        device = 'cuda',
        seed = 1,
        gamma=0.99,
        batch_size_sac=256,
        batch_size_latent=32,
        buffer_size=10 ** 7,
        num_sequences=8,
        lr_sac=3e-4,
        img_feature_dim=256,
        hidden_units=(256, 256),
        tau=5e-3,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Replay buffer.
        self.buffer = ReplayBuffer(buffer_size, num_sequences, state_shape, tactile_shape, action_shape, device)

        # Networks.
        self.actor = GaussianPolicy(action_shape, num_sequences, img_feature_dim, hidden_units).to(device)
        self.critic = TwinnedQNetwork(action_shape, img_feature_dim, hidden_units).to(device)
        self.critic_target = TwinnedQNetwork(action_shape, img_feature_dim, hidden_units).to(device)
        self.latent = LatentModel().to(device)
        soft_update(self.critic_target, self.critic, 1.0)
        grad_false(self.critic_target)

        # Target entropy is -|A|.
        self.target_entropy = -float(action_shape[0])
        # We optimize log(alpha) because alpha is always bigger than 0.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()

        # Optimizers.
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_sac)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_sac)
        self.optim_alpha = Adam([self.log_alpha], lr=lr_sac)

        self.learning_steps_sac = 0
        self.learning_steps_latent = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_repeat = action_repeat
        self.device = device
        self.gamma = gamma
        self.batch_size_sac = batch_size_sac
        self.batch_size_latent = batch_size_latent
        self.num_sequences = num_sequences
        self.tau = tau
        self.episodes = 0
        # JIT compile to speed up.
        fake_feature = torch.empty(1, num_sequences + 1, img_feature_dim, device=device)
        fake_action = torch.empty(1, num_sequences, action_shape[0], device=device)
        self.create_feature_actions = torch.jit.trace(create_feature_actions, (fake_feature, fake_action))
        self.end_rewards = []
        self.total_step = 0
        self.steps_record = []

    def preprocess(self, ob):
        state = torch.tensor(ob.state, dtype=torch.uint8, device=self.device).float().div_(255.0)
        with torch.no_grad():
            # dimension unknown
            feature = self.latent.DINO_encoder(state).view(1, -1)
        action = torch.tensor(ob.action, dtype=torch.float, device=self.device)
        # propagate & concatenate feature
        feature_action = torch.cat([feature, action], dim=1)
        return feature_action

    def explore(self, ob):
        feature_action = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor.sample(feature_action)[0]
        return action.cpu().numpy()[0]

    def exploit(self, ob):
        feature_action = self.preprocess(ob)
        with torch.no_grad():
            action = self.actor(feature_action)
        return action.cpu().numpy()[0]

    def step(self, env, ob, t, is_random):
        t += 1

        if is_random:
            action = env.action_space.sample()
            action[3] = -0.3
        else:

            action = self.explore(ob)

        action = np.append(action, -0.3)
        state, reward, done, _ = env.step(action)

        mask = False if t == env._max_episode_steps else done
        img = state[0][0]
        tactile = state[0][1]
        ob.append(img, tactile, action[0:3])
        self.buffer.append(action[0:3], reward, mask, img, tactile, done)
        self.total_step += 1

        if done:
            self.episodes += 1
            t = 0
            self.end_rewards.append(reward)
            self.steps_record.append(self.total_step)
            state = env.reset()
            img = state[0][0]
            tactile = state[0][1]
            ob.reset_episode(img, tactile)
            self.buffer.reset_episode(img, tactile)
            save_pickle(self.end_rewards, "end_rewards.pkl")
            save_pickle(self.steps_record, "steps_record.pkl")
        return t, self.episodes


    def update_latent(self, writer):
        self.learning_steps_latent += 1
        state_, tactile_, action_, reward_, done_ = self.buffer.sample_latent(self.batch_size_latent)
        self.latent.DINO_Latent_update(state_)


    def update_sac(self, writer):
        self.learning_steps_sac += 1
        state_, tactile_, action_, reward, done = self.buffer.sample_sac(self.batch_size_sac)
        action, feature_action, next_feature_action = self.prepare_batch(state_, tactile_, action_)
        self.update_critic(action, feature_action, next_feature_action, reward, done, writer)
        self.update_actor(feature_action)
        soft_update(self.critic_target, self.critic, self.tau)


    def prepare_batch(self, state_, tactile_, action_):
        with torch.no_grad():
            # f(1:t+1)
            feature_ = self.latent.DINO_encoder(state_)
        action = action_[:, -1]
        feature_action, next_feature_action = self.create_feature_actions(feature_, action_)
        return action, feature_action, next_feature_action

    # update of using SAC
    def update_critic(self, action, feature_action, next_feature_action, reward, done, writer):
        curr_q1, curr_q2 = self.critic(feature_action, action)
        with torch.no_grad():
            next_action, log_pi = self.actor.sample(next_feature_action)
            next_q1, next_q2 = self.critic_target(next_feature_action, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * log_pi
        target_q = reward + (1.0 - done) * self.gamma * next_q
        loss_critic = (curr_q1 - target_q).pow_(2).mean() + (curr_q2 - target_q).pow_(2).mean()
        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        self.optim_critic.step()

        if self.learning_steps_sac % 1000 == 0:
            writer.add_scalar("loss/critic", loss_critic.item(), self.learning_steps_sac)

    def update_actor(self, feature_action):
        action, log_pi = self.actor.sample(feature_action)
        q1, q2 = self.critic(feature_action, action)
        loss_actor = -torch.mean(torch.min(q1, q2) - self.alpha * log_pi)

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

        with torch.no_grad():
            entropy = -log_pi.detach().mean()
        loss_alpha = -self.log_alpha * (self.target_entropy - entropy)

        self.optim_alpha.zero_grad()
        loss_alpha.backward(retain_graph=False)
        self.optim_alpha.step()
        with torch.no_grad():
            self.alpha = self.log_alpha.exp()


    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # We don't save target network to reduce workloads.
        torch.save(self.latent.encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
        torch.save(self.latent.state_dict(), os.path.join(save_dir, "latent.pth"))
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))
