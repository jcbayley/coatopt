import os
import random
import numpy as np
import torch
import gym
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import matplotlib.pyplot as plt

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class Base_Agent(object):
    """
    Define a basic reinforcement learning agent
    """

    NAME = "Abstract Agent"

    def __init__(self):
        #self.set_random_seeds(self.seed) 
        pass

    def set_random_seeds(self, random_seed):
        """
        Sets all possible random seeds to results can be reproduces.

        :param random_seed:
        :return:
        """
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)

    def pick_action(self, state):
        """
        Determines which action to take when given state

        :param state:
        :return:
        """
        raise NotImplementedError

    def start_episode(self):
        """
        Perform any initialisation for the start of an episode.
        :return:
        """
        raise NotImplementedError

    def end_episode(self):
        """
        Performs any cleanup before the next episode.
        :return:
        """
        raise NotImplementedError

    def __str__(self):
        desc = self.NAME
        return desc
    
class PDQNAgent(Base_Agent):
    """
    A soft actor-critic agent for hybrid action spaces

    """

    NAME = 'P-DQN Agent'

    def __init__(
            self, 
            state_dim, 
            action_dim,
            epsilon_initial=1.0,
            epsilon_final=0.01,
            epsilon_decay=1000,
            batch_size=64,
            gamma=0.99,
            lr_critic=0.001,
            lr_actor=0.001,
            lr_alpha=0.001,
            tau_critic=0.005,
            tau_actor=0.005,
            critic_hidden_layers=(256, 128, 64),
            actor_hidden_layers=(256, 128, 64),
            min_thickness=0.0,
            max_thickness=1.0,
            device="cpu"):
        Base_Agent.__init__(self)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device=device

        self.hyperparameters = {
            'epsilon_initial': epsilon_initial,
            'epsilon_final': epsilon_final,
            'epsilon_decay': epsilon_decay,
            'batch_size': batch_size,
            'gamma': gamma,
            'lr_critic': lr_critic,
            'lr_actor': lr_actor,
            'lr_alpha': lr_alpha,
            'tau_critic': tau_critic,
            'tau_actor': tau_actor,
            'critic_hidden_layers': critic_hidden_layers,
            'actor_hidden_layers': actor_hidden_layers,
        }

        self.epsilon = self.hyperparameters['epsilon_initial']
        self.epsilon_initial = self.hyperparameters['epsilon_initial']
        self.epsilon_final = self.hyperparameters['epsilon_final']
        self.epsilon_decay = self.hyperparameters['epsilon_decay']
        self.batch_size = self.hyperparameters['batch_size']
        self.gamma = self.hyperparameters['gamma']

        self.lr_critic = self.hyperparameters['lr_critic']
        self.lr_actor = self.hyperparameters['lr_actor']
        self.lr_alpha = self.hyperparameters['lr_alpha']
        self.tau_critic = self.hyperparameters['tau_critic']
        self.tau_actor = self.hyperparameters['tau_actor']
        self.critic_hidden_layers = self.hyperparameters['critic_hidden_layers']
        self.actor_hidden_layers = self.hyperparameters['actor_hidden_layers']

        action_par_space = {
            "high":max_thickness,
            "low":min_thickness
        }

        self.counts = 0
        self.alpha = 0.2

        # ----  Initialization  ----
        self.critic = DuelingDQN(self.state_dim, self.action_dim, self.critic_hidden_layers,
                                 ).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.critic_target = DuelingDQN(self.state_dim, self.action_dim, self.critic_hidden_layers,
                                        ).to(self.device)
        hard_update(source=self.critic, target=self.critic_target)

        self.actor = GaussianPolicy(
            self.state_dim, self.action_dim, self.actor_hidden_layers, action_par_space).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        self.target_entropy = -torch.Tensor([self.action_dim]).to(self.device).item()
        self.log_alpha = torch.tensor(-np.log(self.action_dim), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_critic)  # todo

    def select_action(self, state, train=True):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        self.epsilon = self.epsilon_final + (self.epsilon_initial - self.epsilon_final) * math.exp(-1. * self.counts / self.epsilon_decay)
        self.counts += 1
        if train:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action_params, _, _ = self.actor.sample(state)

                if random.random() < self.epsilon:
                    action = np.random.randint(self.action_dim)

                else:
                    Q_a, _ = self.critic(state, action_params)
                    Q_a = Q_a.detach().cpu().numpy()
                    action = int(np.argmax(Q_a))
                action_params = action_params.detach().cpu().numpy()
        else:
            with torch.no_grad():
                _, _, action_params = self.actor.sample(state)
                Q_a = self.critic.forward(state, action_params)
                Q_a = Q_a.detach().cpu().numpy()
                action = int(np.argmax(Q_a))
                action_params = action_params.detach().cpu().numpy()

        if np.any(np.isnan(action)):
            print(action)
            sys.exit()
        if np.any(np.isnan(action_params)):
            print(action_params)
            sys.exit()
        return action, action_params[0]

    def update(self, memory):
        state_batch, action_batch, action_params_batch, reward_batch, next_state_batch, done_batch, obs_batch, next_obs_batch, layer_num, next_layer_num = memory.sample(
            self.batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        action_batch = torch.IntTensor(action_batch).to(self.device).long().unsqueeze(1)
        action_params_batch = torch.FloatTensor(action_params_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
        layer_num = torch.FloatTensor(layer_num).to(self.device).unsqueeze(1)
        next_layer_num = torch.FloatTensor(next_layer_num).to(self.device).unsqueeze(1)

        # ------------------------------------ update critic -----------------------------------------------
        with torch.no_grad():
            next_state_action_params, next_state_log_pi, _ = self.actor.sample(next_obs_batch)
            q1_next_target, q2_next_target = self.critic_target(next_obs_batch, next_state_action_params)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi
            q_next = reward_batch + (1 - done_batch) * self.gamma * min_q_next_target
        q1, q2 = self.critic(obs_batch, action_params_batch)
        q_loss = F.mse_loss(q1, q_next) + F.mse_loss(q2, q_next)

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()
        soft_update(self.critic_target, self.critic, self.tau_critic)

        # ------------------------------------ update actor -----------------------------------------------
        pi, log_pi, _ = self.actor.sample(obs_batch)
        q1_pi, q2_pi = self.critic(obs_batch, pi)
        # min_q_pi = torch.min(q1_pi.gather(1, action_batch), q2_pi.gather(1, action_batch))
        min_q_pi = torch.min(q1_pi.mean(), q2_pi.mean())

        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------------------------ update alpha -----------------------------------------------
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.detach().exp()

        return q_loss.item(), actor_loss.item(), alpha_loss.item()

    def save_models(self, actor_path, actor_param_path):
        torch.save(self.critic.state_dict(), actor_path)
        torch.save(self.actor.state_dict(), actor_param_path)
        print('Models saved successfully')

    def load_models(self, actor_path, actor_param_path):
        # also try load on CPU if no GPU available?
        self.critic.load_state_dict(torch.load(actor_path, actor_param_path))
        self.actor.load_state_dict(torch.load(actor_path, actor_param_path))
        print('Models loaded successfully')

    def start_episode(self):
        pass

    def end_episode(self):
        pass



def init_(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


epsilon = 1e-6


class DuelingDQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_layers=(256, 128, 64),
                 ):
        """

        :param state_dim:
        :param action_dim:
        :param hidden_layers:
        """
        super().__init__()

        # initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim + action_dim, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))

        self.adv_layers_1 = nn.Linear(hidden_layers[-1], action_dim)
        self.val_layers_1 = nn.Linear(hidden_layers[-1], 1)

        self.adv_layers_2 = nn.Linear(hidden_layers[-1], action_dim)
        self.val_layers_2 = nn.Linear(hidden_layers[-1], 1)

        self.apply(init_)

    def forward(self, state, action_params):
        temp = torch.cat((state, action_params), dim=1)

        x1 = temp
        for i in range(len(self.layers)):
            x1 = F.relu(self.layers[i](x1))
        adv1 = self.adv_layers_1(x1)
        val1 = self.val_layers_1(x1)
        q_duel1 = val1 + adv1 - adv1.mean(dim=1, keepdim=True)

        x2 = temp
        for i in range(len(self.layers)):
            x2 = F.relu(self.layers[i](x2))
        adv2 = self.adv_layers_1(x2)
        val2 = self.val_layers_1(x2)
        q_duel2 = val2 + adv2 - adv2.mean(dim=1, keepdim=True)

        return q_duel1, q_duel2


class GaussianPolicy(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_layers=(256, 128, 64), action_space=None,
                 ):
        """

        :param state_dim:
        :param action_dim:
        :param hidden_layers:
        """
        super().__init__()

        # initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        self.mean_layers = nn.Linear(hidden_layers[-1], action_dim)
        self.log_std_layers = nn.Linear(hidden_layers[-1], action_dim)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                [(action_space["high"] - action_space["low"]) / 2.])
            self.action_bias = torch.FloatTensor(
                [(action_space["high"] + action_space["low"]) / 2.])

        self.apply(init_)

    def forward(self, state):
        x = state

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))
        mean = torch.sigmoid(self.mean_layers(x))
        log_std = torch.sigmoid(self.log_std_layers(x))*(10) - 10
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        #print('std:', std)
        #print('mean:', mean)
        # normal = Normal(mean, std)
        # x_t = normal.rsample()
        # y_t = torch.tanh(x_t)
        # action = y_t * self.action_scale + self.action_bias
        noise = torch.randn_like(mean, requires_grad=True)
        action = (mean + std * noise).tanh()

        # log_prob = normal.log_prob(x_t)
        # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        # log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        log_prob = log_std + np.log(np.sqrt(2 * np.pi)) + noise.pow(2).__mul__(0.5)
        log_prob += (-action.pow(2) + 1.00000001).log()
        return action, log_prob.sum(1, keepdim=True), mean
    
class ReplayBuffer:
    def __init__(self, capacity=1e5):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, action_param, reward, next_state, done, obs, next_obs, step, next_step = map(np.stack, zip(*batch))
        return state, action, action_param, reward, next_state, done, obs, next_obs, step, next_step

    def push(self, state, action, action_param, reward, next_state, done, obs, next_obs, step, next_step):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, action_param, reward, next_state, done, obs, next_obs, step, next_step)
        self.position = int((self.position + 1) % self.capacity)

    def __len__(self):
        return len(self.buffer)


class PDQNTrainer():

    def __init__(self, agent, environment, episodes, max_steps, exploration_decay=0.999, useobs=True, root_dir=".", batch_size=256):
        self.agent = agent
        self.env = environment
        self.episodes = episodes
        self.max_steps = max_steps
        self.useobs=useobs
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.exploration_decay = exploration_decay
        self.updates_per_step = 1

        self.memory = ReplayBuffer(1e4)

    def train(self,):

        q_losses = []
        a_losses = []
        e_losses = []
        state_values = []
        rewards = []
        max_state = None
        max_value = -np.inf
        max_reward = -np.inf

        for i_episode in range(self.episodes):

            episode_score = []
            episode_steps = 0
            done = 0
            state = self.env.reset()  # n_steps
            total_reward = 0
            tq_losses = []
            ta_losses = []
            te_losses = []

            while not done:
                obs = self.env.get_observation_from_state(state).flatten()
                if len(self.memory) > self.batch_size:
                    action, action_params = self.agent.select_action(obs, True)

                    action_for_env = [action, action_params[action]*(self.env.max_thickness - self.env.min_thickness) + self.env.min_thickness]

                    for i in range(self.updates_per_step):
                        qloss, aloss, eloss = self.agent.update(self.memory)
                        tq_losses.append(qloss)
                        ta_losses.append(aloss)
                        te_losses.append(eloss)
                else:
                    action_params = np.random.uniform(low=self.env.min_thickness, high=self.env.max_thickness, size=self.env.n_materials)
                    action = np.random.randint(self.env.n_materials, size=1)[0]
                    action_for_env = [action, action_params[action]]

                next_state, reward, terminated, finished, _, _ = self.env.step(action_for_env)
                next_obs = self.env.get_observation_from_state(next_state).flatten()

                done = terminated or finished
                total_reward += reward

                episode_steps += 1

                self.memory.push(state, action, action_params, reward, next_state, done, obs, next_obs, episode_steps, episode_steps+1)

                state = next_state

            rewards.append(total_reward)
            q_losses.append(np.mean(tq_losses))
            a_losses.append(np.mean(ta_losses))
            e_losses.append(np.mean(te_losses))

            print(f"Episode {i_episode + 1}, Total Reward: {total_reward}")

            if total_reward > max_reward:
                max_reward = total_reward
                max_state = state
                max_value = self.env.compute_state_value(state)
                fig, ax = self.env.plot_stack(state)
                ax.set_title(f"Reward: {max_reward}, value: {max_value}")
                fig.savefig(os.path.join(self.root_dir, f"best_state.png"))


            if i_episode % 100 == 0:
                self.loss_plot(q_losses, a_losses, e_losses, rewards, savefig=True)
                


    def loss_plot(self, q_losses, a_losses, e_losses, rewards, savefig=False):
        
        fig, ax = plt.subplots(nrows=4)
        ax[0].plot(q_losses, label="Critic Loss")
        ax[1].plot(a_losses, label="Actor Loss")
        ax[2].plot(e_losses, label="Alpha Loss")
        ax[3].plot(rewards, label="Rewards")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[3].legend()
    

        if savefig:
            fig.savefig(os.path.join(self.root_dir, "losses.png"))