import numpy as np
import torch 
from torch.nn import functional as F
from collections import deque
from coatopt.networks.truncated_normal import TruncatedNormalDist
from coatopt.algorithms.policy_nets import DiscretePolicy, ContinuousPolicy, Value
from coatopt.algorithms.pre_networks import PreNetworkLinear, PreNetworkLSTM, PreNetworkAttention
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

class ReplayBuffer:
    def __init__(self):
        self.discrete_actions = []
        self.continuous_actions = []
        self.states = []
        self.logprobs_discrete = []
        self.logprobs_continuous = []
        self.rewards = []
        self.state_values = []
        self.dones = []
        self.entropy_discrete = []
        self.entropy_continuous = []
        self.returns = []
        self.layer_number = []
        self.hidden_state = []

        self.t_discrete_actions = []
        self.t_continuous_actions = []
        self.t_states = []
        self.t_logprobs = []
        self.t_rewards = []
        self.t_state_values = []
        self.t_dones = []
        self.t_entropy = []
    
    def clear(self):
        del self.discrete_actions[:]
        del self.continuous_actions[:]
        del self.states[:]
        del self.logprobs_discrete[:]
        del self.logprobs_continuous[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.dones[:]
        del self.entropy_discrete[:]
        del self.entropy_continuous[:]
        del self.returns[:]
        del self.layer_number[:]
        del self.hidden_state[:]

    def update(
            self, 
            discrete_action, 
            continuous_action, 
            state, 
            logprob_discrete,
            logprob_continuous, 
            reward, 
            state_value, 
            done,
            entropy_discrete,
            entropy_continuous,
            layer_number=0,
            hidden_state=None):
        self.discrete_actions.append(discrete_action)
        self.continuous_actions.append(continuous_action)
        self.states.append(state)
        self.logprobs_discrete.append(logprob_discrete)
        self.logprobs_continuous.append(logprob_continuous)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.dones.append(done)
        self.entropy_discrete.append(entropy_discrete)
        self.entropy_continuous.append(entropy_continuous)
        self.layer_number.append(layer_number)
        self.hidden_state.append(hidden_state)

    def update_returns(self, returns):
        self.returns.extend(returns)

    def pad_states(self):
        self.states = pad_sequence(self.states, batch_first=True)



class HPPO(object):

    def __init__(
            self, 
            state_dim, 
            num_discrete, 
            num_cont, 
            hidden_size, 
            disc_lr_policy=1e-4, 
            cont_lr_policy=1e-4, 
            lr_value=2e-4, 
            lr_step=10,
            lr_min=1e-6,
            T_mult=1,
            lower_bound=0.1, 
            upper_bound=1.0, 
            n_updates=1, 
            beta=0.1,
            clip_ratio=0.5,
            gamma=0.99,
            include_layer_number=False,
            pre_type="linear",
            n_heads=2,
            n_pre_layers=2,
            optimiser="adam",
            n_discrete_layers=2,
            n_continuous_layers=2,
            n_value_layers=2,
            value_hidden_size=32,
            discrete_hidden_size=32,
            continuous_hidden_size=32,
            activation_function="relu",
            include_material_in_policy=False,
            substrate_material_index=0,
            ignore_air_option=False):

        print("sd", state_dim)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.include_layer_number = include_layer_number
        self.substrate_material_index = substrate_material_index
        self.ignore_air_option = ignore_air_option

        self.pre_output_dim = hidden_size
        self.pre_type = pre_type
        #self.pre_output_dim = pre_output_dim 

        if self.pre_type == "attn":
            self.pre_network = PreNetworkAttention(
            state_dim[-1],
            self.pre_output_dim,
            hidden_size,
            num_heads=n_heads,
            num_layers=n_pre_layers
        )
        elif pre_type == "lstm":
            self.pre_network = PreNetworkLSTM(
                    state_dim[-1],
                    self.pre_output_dim,
                    hidden_size,
                    include_layer_number=include_layer_number,
                    n_layers = n_pre_layers
                )
        elif pre_type == "linear":
            self.pre_network = PreNetworkLinear(
                np.prod(state_dim),
                self.pre_output_dim,
                hidden_size,
                n_layers=n_pre_layers,
                include_layer_number=include_layer_number
            )
        else:
            raise Exception(f"No type: {pre_type}")
    

        for i in range(2):
            appended_str = "" if i == 0 else "_old"
            setattr(self, f"policy_discrete{appended_str}",DiscretePolicy(
                self.pre_output_dim, 
                num_discrete, 
                discrete_hidden_size,
                n_layers=n_discrete_layers,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                include_layer_number=include_layer_number,
                activation=activation_function))
            
            setattr(self, f"policy_continuous{appended_str}",  ContinuousPolicy(
                self.pre_output_dim, 
                num_cont, 
                continuous_hidden_size,
                n_layers=n_continuous_layers,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                include_layer_number=include_layer_number,
                include_material=include_material_in_policy,
                activation=activation_function))
            
            setattr(self, f"value{appended_str}", Value(
                self.pre_output_dim, 
                value_hidden_size,
                n_layers=n_value_layers,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                include_layer_number=include_layer_number,
                activation=activation_function))
        
        
        self.policy_discrete_old.load_state_dict(self.policy_discrete.state_dict())
        self.policy_continuous_old.load_state_dict(self.policy_continuous.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())
        
        self.lr_value = lr_value
        self.disc_lr_policy = disc_lr_policy
        self.cont_lr_policy = cont_lr_policy
        if optimiser == "adam":
            self.optimiser_discrete = torch.optim.Adam(self.policy_discrete.parameters(), lr=self.disc_lr_policy)
            self.optimiser_continuous = torch.optim.Adam(self.policy_continuous.parameters(), lr=self.cont_lr_policy)
            self.optimiser_value = torch.optim.Adam(self.value.parameters(), lr=self.lr_value)
        elif optimiser == "sgd":
            self.optimiser_discrete = torch.optim.SGD(self.policy_discrete.parameters(), lr=self.disc_lr_policy)
            self.optimiser_continuous = torch.optim.SGD(self.policy_continuous.parameters(), lr=self.cont_lr_policy)
            self.optimiser_value = torch.optim.SGD(self.value.parameters(), lr=self.lr_value)
        else:
            raise Exception(f"Optimiser not supported: {optimiser}")
        
        if type(lr_step) in [list, tuple]:
            lr_step_discrete, lr_step_continuous, lr_step_value = lr_step
        else:
            lr_step_discrete = lr_step
            lr_step_continuous = lr_step
            lr_step_value = lr_step

        if type(T_mult) in [list, tuple]:
            T_mult_discrete, T_mult_continuous, T_mult_value = T_mult
        else:
            T_mult_discrete = T_mult
            T_mult_continuous = T_mult
            T_mult_value = T_mult

        self.scheduler_discrete = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimiser_discrete, T_0=lr_step_discrete, T_mult=T_mult_discrete, eta_min=lr_min)
        self.scheduler_continuous = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimiser_continuous, T_0=lr_step_continuous, T_mult=T_mult_continuous, eta_min=lr_min)
        self.scheduler_value = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimiser_value, T_0=lr_step_value, T_mult=T_mult_value, eta_min=lr_min)

        self.mse_loss = torch.nn.MSELoss()
        self.replay_buffer = ReplayBuffer()

        self.beta = beta
        self.clip_ratio = clip_ratio
        self.gamma=gamma

        self.n_updates = n_updates

    def get_returns(self, rewards):

        temp_r = deque()
        R=0
        for r in rewards[::-1]:
            R = r + self.gamma * R
            temp_r.appendleft(R)
        
        return np.array(temp_r)

    def scheduler_step(self, make_step=True):
        if make_step:
            self.scheduler_discrete.step()
            self.scheduler_continuous.step()
            self.scheduler_value.step()
        
        return self.scheduler_discrete.get_last_lr(), self.scheduler_continuous.get_last_lr(), self.scheduler_value.get_last_lr()

    def update(self, update_policy=True, update_value=True):

        R = 0
        policy_loss = []

        eps = 1e-8

        returns = torch.from_numpy(np.array(self.replay_buffer.returns))
        returns = (returns - returns.mean()) / (returns.std() + eps)

        state_vals = torch.cat(self.replay_buffer.state_values).to(torch.float32)
        old_lprobs_discrete = torch.cat(self.replay_buffer.logprobs_discrete).to(torch.float32).detach()
        old_lprobs_continuous = torch.cat(self.replay_buffer.logprobs_continuous).to(torch.float32).detach()
        #advantage = returns.detach() - state_vals.detach()

        #for log_prob, R in zip(self.replay_buffer.logprobs, returns):
        #    policy_loss.append(-log_prob * R)
        #policy_loss = torch.cat(policy_loss).mean()
        #print(torch.cat(self.replay_buffer.logprobs).size(), returns.size())
        #print(self.replay_buffer.continuous_actions)
        actionsc = torch.cat(self.replay_buffer.continuous_actions, dim=0).detach()
        actionsd = torch.cat(self.replay_buffer.discrete_actions, dim=-1).detach()
        #print(actionsd.size(), actionsc.size())
        #torch.autograd.set_detect_anomaly(True)
        for _ in range(self.n_updates):
            # compute probs values and advantages

            states = torch.tensor(self.replay_buffer.states).to(torch.float32)
            

            if self.include_layer_number:
                layer_numbers = torch.tensor(self.replay_buffer.layer_number).to(torch.float32)
            else:
                layer_numbers = False

            action, actiond, actionc, log_prob_discrete, log_prob_continuous, d_probs, c_means, c_std, state_value, entropy_discrete, entropy_continuous = self.select_action(
                states, 
                layer_numbers, 
                actionsc, 
                actionsd, 
                packed=True)
            advantage = returns.detach() - state_value.detach()

            # compute discrete PPO clipped objective
            ratios_discrete = torch.exp(log_prob_discrete - old_lprobs_discrete)
            #policy_loss_discrete = (-ratios_discrete.squeeze() * advantage.squeeze() - self.beta*entropy_discrete.squeeze()).mean()
            d_surr1 = ratios_discrete.squeeze() * advantage.squeeze()
            d_surr2 = torch.clamp(ratios_discrete, 1-self.clip_ratio, 1+self.clip_ratio) * advantage.squeeze()
            policy_loss_discrete = -(torch.min(d_surr1, d_surr2) + self.beta*entropy_discrete.squeeze()).mean()
 
            # compute continuous PPO clipped objective
            ratios_continuous = torch.exp(log_prob_continuous - old_lprobs_continuous)
            c_surr1 = ratios_continuous.squeeze() * advantage.squeeze()
            c_surr2 = torch.clamp(ratios_continuous, 1-self.clip_ratio, 1+self.clip_ratio) * advantage.squeeze()
            policy_loss_continuous = -(torch.min(c_surr1, c_surr2) + self.beta*entropy_continuous.squeeze()).mean()
            #policy_loss_continuous = -(torch.min(c_surr1, c_surr2)).mean()
            #policy_loss_continuous = -(log_prob_continuous * advantage.squeeze() + self.beta*entropy_continuous.squeeze()).mean()

            value_loss = self.mse_loss(returns.to(torch.float32).squeeze(), state_value.squeeze())

            #print(policy_loss_discrete, policy_loss_continuous, value_loss)
            if update_policy:
                self.optimiser_discrete.zero_grad()
                policy_loss_discrete.backward()
                self.optimiser_discrete.step()

            self.optimiser_continuous.zero_grad()
            policy_loss_continuous.backward()
            self.optimiser_continuous.step()

            if update_value:
                self.optimiser_value.zero_grad()
                value_loss.backward()
                self.optimiser_value.step()

        self.policy_discrete_old.load_state_dict(self.policy_discrete.state_dict())
        self.policy_continuous_old.load_state_dict(self.policy_continuous.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

        self.replay_buffer.clear()

        return policy_loss_discrete.item(), policy_loss_continuous.item(), value_loss.item()
    

    def select_action(self, state, layer_number=None, actionc=None, actiond=None, packed=False):

        if type(state) in [np.array, np.ndarray]:
            #state = torch.from_numpy(state).flatten().unsqueeze(0).to(torch.float32)
            state = torch.from_numpy(state).unsqueeze(0).to(torch.float32)

        if self.pre_type == "linear":
            state = state.flatten(1)
        
        if layer_number is not None and self.include_layer_number:
            if type(layer_number) in [np.array, np.ndarray]:
                layer_number = torch.from_numpy(layer_number).flatten().unsqueeze(0)
                
            layer_number = layer_number.to(torch.int)
            
            lyn = layer_number.clone().detach().view(-1)
            lyn[lyn == 0] = 1


        state_pack = pack_padded_sequence(state, lengths=lyn, batch_first=True, enforce_sorted=False)

        #print(state.size(), layer_number.size())
        pre_output_d = self.pre_network(state_pack, layer_number, packed=True)
        pre_output_c = self.pre_network(state_pack, layer_number, packed=True)
        pre_output_v = self.pre_network(state_pack, layer_number, packed=True)
        d_probs = self.policy_discrete(pre_output_d, layer_number)

        state_value = self.value(pre_output_v, layer_number)
        #d_onehot = F.one_hot(d_probs, num_classes=policy.output_dim_discrete)

        # mask out the material index which has the same material as this layer


        if layer_number is not None:
            #zeroidx = torch.argmax(state[torch.arange(state.size(0)), layer_number.squeeze()-2].squeeze(1)[:, 1:], dim=-1).unsqueeze(1)
            zeroidx = []
            for lidx,ly in enumerate(layer_number):
                if ly == 0:
                    zeroidx.append(self.substrate_material_index)
                else:
                    material_idx = torch.argmax(state[lidx, ly-1].squeeze(1)[:, 1:])
                    zeroidx.append(material_idx)

            t_d_probs = d_probs
            mask = torch.ones_like(t_d_probs)
            mask[torch.arange(d_probs.size(0)), zeroidx] = 1e-10
            mask[:, 0] *= 1e-2 # add low probabillity of choosing air to start with
            t_d_probs = t_d_probs * mask
            if self.ignore_air_option:
                t_d_probs = t_d_probs[:,1:]
            #t_d_probs[torch.arange(d_probs.size(0)), zeroidx] = 0

            if len(t_d_probs.size()) == 1:
                t_d_probs = t_d_probs.unsqueeze(0)
            d = torch.distributions.Categorical(t_d_probs)
        else:
            d = torch.distributions.Categorical(d_probs)

        if actiond is None:
            actiond = d.sample()
            if self.ignore_air_option:
                actiond += 1


        c_means, c_std = self.policy_continuous(pre_output_c, layer_number, actiond.unsqueeze(1))
        
        c = TruncatedNormalDist(
            c_means, 
            c_std, 
            self.lower_bound, 
            self.upper_bound)
 
        if actionc is None:
            actionc = c.sample()

        actionc = torch.clamp(actionc, self.lower_bound, self.upper_bound)

        #actionc[actionc == self.lower_bound] += 1e-4
        #actionc[actionc == self.upper_bound] -= 1e-4

        #print(d_probs.size(), actiond.size(), actionc.size(), c_means.size())
        
        if self.ignore_air_option:
            log_prob_discrete = d.log_prob(actiond-1) 
        else:
            log_prob_discrete = d.log_prob(actiond)
        log_prob_continuous = torch.sum(c.log_prob(actionc), dim=-1)#[:, actiond.detach()]

        # get the continuous action for sampled discrete element
        
        #c_action = actionc.detach()[:,actiond.detach()]


        #print(actiond.unsqueeze(0).T.size(), actionc.size())
        action = torch.cat([actiond.detach().unsqueeze(0).T, actionc], dim=-1)[0]

        entropy_discrete = d.entropy() 
        entropy_continuous = torch.sum(c._entropy, dim=-1)

        #policy.saved_log_probs.append(log_prob)

        return action, actiond, actionc, log_prob_discrete, log_prob_continuous, d_probs, c_means, c_std, state_value, entropy_discrete, entropy_continuous


    def save_networks(self, save_directory, episode=None):

        discrete_policy_fname = os.path.join(save_directory, "discrete_policy.pt")
        torch.save({
            "episode":episode,
            "model_state_dict":self.policy_discrete.state_dict(),
            "optimiser_state_dict":self.optimiser_discrete.state_dict()
        }, discrete_policy_fname)

        continuous_policy_fname = os.path.join(save_directory, "continuous_policy.pt")
        torch.save({
            "episode":episode,
            "model_state_dict":self.policy_continuous.state_dict(),
            "optimiser_state_dict":self.optimiser_continuous.state_dict()
        }, continuous_policy_fname)

        value_fname = os.path.join(save_directory, "value.pt")
        torch.save({
            "episode":episode,
            "model_state_dict":self.value.state_dict(),
            "optimiser_state_dict":self.optimiser_value.state_dict()
        }, value_fname)

    def load_networks(self, load_directory):
        discrete_policy_fname = os.path.join(load_directory, "discrete_policy.pt")
        dp = torch.load(discrete_policy_fname)
        self.policy_discrete.load_state_dict(dp["model_state_dict"])
        self.optimiser_discrete.load_state_dict(dp["optimiser_state_dict"])

        continuous_policy_fname = os.path.join(load_directory, "continuous_policy.pt")
        cp = torch.load(continuous_policy_fname)
        self.policy_continuous.load_state_dict(cp["model_state_dict"])
        self.optimiser_continuous.load_state_dict(cp["optimiser_state_dict"])

        value_fname = os.path.join(load_directory, "value.pt")
        vp = torch.load(value_fname)
        self.value.load_state_dict(vp["model_state_dict"])
        self.optimiser_value.load_state_dict(vp["optimiser_state_dict"])



class HPPOTrainer:

    def __init__(
            self,
            agent, 
            env, 
            n_iterations = 1000,  
            n_layers = 4, 
            root_dir="./",
            beta_start=1.0,
            beta_end=0.001,
            beta_decay_length=None,
            beta_decay_start=0,
            n_training_epochs=10,
            use_obs=True,
            scheduler_start=0,
            scheduler_end=np.inf,
            continue_training=False
            ):
        self.agent = agent
        self.env = env
        self.n_iterations = n_iterations
        self.root_dir = root_dir
        self.n_layers = n_layers
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_decay_length = beta_decay_length
        self.beta_decay_start = beta_decay_start
        self.n_training_epochs = n_training_epochs
        self.use_obs = use_obs

        self.scheduler_start = scheduler_start
        self.scheduler_end = np.inf if scheduler_end == -1 else scheduler_end

        if continue_training:
            self.load_metrics_from_file()
            self.start_episode = self.metrics["episode"].max()
        else:
            self.metrics = pd.DataFrame(columns=["episode", "loss_policy_continuous", "loss_policy_discrete", "beta", "lr_discrete", "lr_continuous", "lr_value", "reward", "reflectivity", "thermal_noise"])
            self.start_episode = 0

    def write_metrics_to_file(self):
        self.metrics.to_csv(os.path.join(self.root_dir, "training_metrics.csv"), index=False)

    def load_metrics_from_file(self):
        self.metrics = pd.read_csv(os.path.join(self.root_dir, "training_metrics.csv"))


    def make_reward_plot(self,):

        reward_fig, reward_ax = plt.subplots(nrows=5, figsize=(7,9))
        window_size = 20
        #downsamp_rewards = np.mean(np.reshape(self.rewards[:int((len(self.rewards)//window_size)*window_size)], (-1,window_size)), axis=1)
        downsamp_rewards = self.metrics['reward'].rolling(window=window_size, center=False).median()
        downsamp_episodes = self.metrics['episode'].rolling(window=window_size, center=False).median()
        reward_ax[0].plot(self.metrics["episode"], self.metrics["reward"])
        reward_ax[0].plot(downsamp_episodes, downsamp_rewards)
        reward_ax[0].set_xlabel("Episode number")
        reward_ax[0].set_ylabel("Reward")

        downsamp_reflectivites= self.metrics['reflectivity'].rolling(window=window_size, center=False).median()
        #downsamp_values = np.mean(np.reshape(self.reflectivities[:int((len(self.reflectivities)//window_size)*window_size)], (-1,window_size)), axis=1)
        reward_ax[1].plot(self.metrics["episode"], self.metrics["reflectivity"])
        reward_ax[1].plot(downsamp_episodes, downsamp_reflectivites)
        reward_ax[1].set_xlabel("Episode number")
        reward_ax[1].set_ylabel("Reflectivity ")

        #downsamp_values = np.mean(np.reshape(self.thermal_noises[:int((len(self.thermal_noises)//window_size)*window_size)], (-1,window_size)), axis=1)
        downsamp_thermal_noise = self.metrics['thermal_noise'].rolling(window=window_size, center=False).median()
        reward_ax[2].plot(self.metrics["episode"], self.metrics["thermal_noise"])
        reward_ax[2].plot(downsamp_episodes, downsamp_thermal_noise)
        reward_ax[2].set_xlabel("Episode number")
        reward_ax[2].set_ylabel("Thermal noise")

        reward_ax[3].plot(self.metrics["episode"], self.metrics["beta"])
        reward_ax[3].set_xlabel("Episode number")
        reward_ax[3].set_ylabel("Entropy weight")

        reward_ax[4].plot(self.metrics["episode"], self.metrics["lr_discrete"], label="discrete")
        reward_ax[4].plot(self.metrics["episode"], self.metrics["lr_continuous"], label="continuous")
        reward_ax[4].plot(self.metrics["episode"], self.metrics["lr_value"], label="value")
        reward_ax[4].set_xlabel("Episode number")
        reward_ax[4].set_ylabel("Learning Rate")
        reward_ax[4].legend()
        reward_fig.savefig(os.path.join(self.root_dir, "running_rewards.png"))

    def make_loss_plot(self,):
        loss_fig, loss_ax = plt.subplots(nrows=3)
        loss_ax[0].plot(self.metrics["episode"], self.metrics["loss_policy_discrete"])
        loss_ax[1].plot(self.metrics["episode"], self.metrics["loss_policy_continuous"])
        loss_ax[2].plot(self.metrics["episode"], self.metrics["loss_value"])
        loss_ax[0].set_ylabel("Policy discrete loss")
        loss_ax[1].set_ylabel("Policy continuous loss")
        loss_ax[2].set_ylabel("Value loss")
        loss_ax[2].set_yscale("log")
        loss_ax[2].set_xlabel("Episode number")
        loss_fig.savefig(os.path.join(self.root_dir, "running_losses.png"))

    def train(self):

        # Training loop
        all_means = []
        all_stds = []
        all_mats = []
        max_reward = -np.inf
        max_state = None

        state_dir = os.path.join(self.root_dir, "states")
        if not os.path.isdir(state_dir):
            os.makedirs(state_dir)

        self.betas = []
        self.lrs = []

        for episode in range(self.n_iterations):

            if episode < self.scheduler_start or episode > self.scheduler_end:
                make_step = False
            else:
                make_step = True 

            if self.beta_decay_length is not None and episode > self.beta_decay_length:
                update_value=True
            else:
                update_value=True

            if self.beta_decay_length is not None and episode > self.beta_decay_start:
                self.agent.beta = self.beta_start - (self.beta_start-self.beta_end)*np.min([(episode-self.beta_decay_start)/self.beta_decay_length, 1])
            else:
                self.agent.beta = self.beta_start

                #agent.optimiser_discrete.learning_rate = new_lr
                #agent.optimiser_continuous.learning_rate = new_lr
                #agent.optimiser_value.learning_rate = new_lr

            metric_update = {}

            metric_update["beta"] = self.agent.beta
            #lrs.append(agent.optimiser_discrete.learning_rate)

            states = []
            actions_discrete = []
            actions_continuous = []
            returns = []
            advantages = []
            for n in range(self.start_episode, self.n_training_epochs):
                state = self.env.reset()
                episode_reward = 0
                means = []
                stds = []
                mats = []
                t_rewards = []
                for t in range(100):
                    # Select action
                    fl_state = np.array([state.flatten(),])[0]
                    obs = self.env.get_observation_from_state(state)
                    if self.use_obs:
                        obs = obs
                    else:
                        obs = state
                    t = np.array([t])
                    
                    action, actiond, actionc, log_prob_discrete, log_prob_continuous, d_prob, c_means, c_std, value, entropy_discrete, entropy_continuous = self.agent.select_action(obs, t)

                    action[1] = action[1]*(self.env.max_thickness - self.env.min_thickness) + self.env.min_thickness
                
                    # Take action and observe reward and next state
                    next_state, reward, done, finished, _, full_action = self.env.step(action)

                    t_rewards.append(reward)
                    self.agent.replay_buffer.update(
                        actiond,
                        actionc,
                        obs,
                        log_prob_discrete,
                        log_prob_continuous,
                        reward,
                        value,
                        done,
                        entropy_discrete,
                        entropy_continuous,
                        t
                    )
                    #log_prob, state_value, entropy = agent.evaluate(fl_state, action[0], action[1])

                    means.append(c_means.detach().numpy())
                    stds.append(c_std.detach().numpy())
                    mats.append(d_prob.detach().numpy().tolist()[0])
                    fl_next_state = next_state.flatten()
                    # Store transition in replay buffer
                    #agent.policy.rewards.append(reward)

                    # Update state and episode rewardz
                    state = next_state
                    episode_reward += reward

                    # Check if episode is done
                    if done or finished:
                        break

                if episode_reward > max_reward:
                    max_reward = episode_reward
                    max_state = state
                    opt_value = self.env.compute_state_value(max_state)
                    fig, ax = self.env.plot_stack(max_state)
                    ax.set_title(f"Optimal rew: {max_reward}, opt val: {opt_value}")
                    fig.savefig(os.path.join(self.root_dir,  f"best_state.png"))

                    self.agent.save_networks(self.root_dir)

                returns = self.agent.get_returns(t_rewards)
                self.agent.replay_buffer.update_returns(returns)
                

            all_means.append(means)
            all_stds.append(stds)
            all_mats.append(mats)

            if episode > 10:
                loss1, loss2, loss3 = self.agent.update(update_policy=True, update_value=True)
                lr_outs = self.agent.scheduler_step(make_step)
                lr_outs = lr_outs[0][0], lr_outs[1][0], lr_outs[2][0]
                self.agent.replay_buffer.clear()
            else:
                lr_outs = self.agent.disc_lr_policy, self.agent.cont_lr_policy, self.agent.lr_value 
                loss1, loss2, loss3 = np.nan, np.nan, np.nan

            metric_update["lr_discrete"] = lr_outs[0]
            metric_update["lr_continuous"] = lr_outs[1]
            metric_update["lr_value"] = lr_outs[2]
            metric_update["loss_policy_discrete"] = loss1
            metric_update["loss_policy_continuous"] = loss2
            metric_update["loss_value"] = loss3

            metric_update["episode"] = episode
            metric_update["reward"] = episode_reward

            _, reflectivity, thermal_noise = self.env.compute_reward(state)
            metric_update["reflectivity"] = reflectivity
            metric_update["thermal_noise"] = thermal_noise

            self.metrics = pd.concat([self.metrics, pd.DataFrame([metric_update])], ignore_index=True)

            if episode % 20 == 0 and episode !=0 :
                self.write_metrics_to_file()
                self.make_reward_plot()
                self.make_loss_plot()

                
                n_layers = self.n_layers
                loss_fig, loss_ax = plt.subplots(nrows = n_layers)
                all_mats2 = pad_lists(all_mats, [0.0,]*self.env.n_materials, n_layers)

                #all_mats2 = all_mats2[:,:,:]
                color_map = {
                    0: 'gray',    # No active material
                    1: 'blue',    # m1
                    2: 'green',   # m2
                    3: 'red'      # m3
                }
                for i in range(n_layers):
                    for mind in range(len(all_mats2[0, i])):
                        loss_ax[i].scatter(np.arange(len(all_mats2)), np.ones(len(all_mats2))*mind, s=100*all_mats2[:,i,mind], color=color_map[mind])
                    loss_ax[i].set_ylabel(f"Layer {i}")
                loss_ax[-1].set_xlabel("Episode number")
                loss_fig.savefig(os.path.join(self.root_dir, "running_mats.png"))
                
                # Print episode information
                print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

                if episode % 100 == 0:
                    fig, ax = self.env.plot_stack(state)
                    t_opt_value = self.env.compute_state_value(state)
                    ax.set_title(f"Reward: {episode_reward}, val: {t_opt_value}")
                    fig.savefig(os.path.join(self.root_dir,  "states", f"episode_{episode}.png"))
                

        print("Max_state: ", max_reward)
        print(max_state)

        return self.rewards, max_state



def pad_lists(list_of_lists, padding_value=0, max_length=3):
    if max_length is None:
        max_length = max(len(lst) for lst in list_of_lists)
    padded_lists = []
    for lst in list_of_lists:
        t_lst = []
        for l in lst:
            t_lst.append(l)

        if len(t_lst) < max_length:
            diff = int(max_length - len(t_lst))
            for i in range(diff):
                t_lst.append(padding_value)

        
        padded_lists.append(t_lst)

    #padded_lists = [lst + [padding_value] * (max_length - len(lst)) for lst in list_of_lists]
    return np.array(padded_lists)