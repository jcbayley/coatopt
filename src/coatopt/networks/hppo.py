import numpy as np
import torch 
from torch.nn import functional as F
from collections import deque
from coatopt.networks.truncated_normal import TruncatedNormalDist
from coatopt.networks.policy_nets import DiscretePolicy, ContinuousPolicy, Value
from coatopt.networks.pre_networks import PreNetworkLinear, PreNetworkLSTM, PreNetworkAttention
import os
import sys

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
            layer_number=0):
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

    def update_returns(self, returns):
        self.returns.extend(returns)



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
            activation_function="relu"):

        print("sd", state_dim)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.include_layer_number = include_layer_number

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
    

        self.policy_discrete = DiscretePolicy(
            self.pre_output_dim, 
            num_discrete, 
            discrete_hidden_size,
            n_layers=n_discrete_layers,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            include_layer_number=include_layer_number,
            activation=activation_function)
        
        self.policy_continuous = ContinuousPolicy(
            self.pre_output_dim, 
            num_cont, 
            continuous_hidden_size,
            n_layers=n_continuous_layers,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            include_layer_number=include_layer_number,
            activation=activation_function)
        
        self.value = Value(
            self.pre_output_dim, 
            value_hidden_size,
            n_layers=n_value_layers,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            include_layer_number=include_layer_number,
            activation=activation_function)
        
        self.policy_old_discrete = DiscretePolicy(
            self.pre_output_dim, 
            num_discrete, 
            discrete_hidden_size,
            n_layers=n_discrete_layers,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            include_layer_number=include_layer_number,
            activation=activation_function)
        
        self.policy_old_continuous = ContinuousPolicy(
            self.pre_output_dim, 
            num_cont, 
            continuous_hidden_size,
            n_layers=n_continuous_layers,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            include_layer_number=include_layer_number,
            activation=activation_function)
        
        self.value_old = Value(
            self.pre_output_dim, 
            value_hidden_size,
            n_layers=n_value_layers,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            include_layer_number=include_layer_number,
            activation=activation_function)
        
        self.policy_old_discrete.load_state_dict(self.policy_discrete.state_dict())
        self.policy_old_continuous.load_state_dict(self.policy_continuous.state_dict())
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
        
        self.scheduler_discrete = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimiser_discrete, T_max=lr_step, eta_min=lr_min)  # Decrease LR by factor of 0.1 every 10 epochs
        self.scheduler_continuous = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimiser_continuous, T_max=lr_step, eta_min=lr_min)  # Decrease LR by factor of 0.1 every 10 epochs
        self.scheduler_value = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimiser_value, T_max=lr_step, eta_min=lr_min)  # Decrease LR by factor of 0.1 every 10 epochs


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

    def scheduler_step(self):
        self.scheduler_discrete.step()
        self.scheduler_continuous.step()
        self.scheduler_value.step()

    def update(self, update_policy=True, update_value=True):

        R = 0
        policy_loss = []

        eps = 1e-8

        returns = torch.from_numpy(np.array(self.replay_buffer.returns))
        returns = (returns - returns.mean()) / (returns.std() + eps)

        #print(self.replay_buffer.state_values)
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
            action, actiond, actionc, log_prob_discrete, log_prob_continuous, d_probs, c_means, c_std, state_value, entropy_discrete, entropy_continuous = self.select_action(states, layer_numbers, actionsc, actionsd)
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

        self.policy_old_discrete.load_state_dict(self.policy_discrete.state_dict())
        self.policy_old_continuous.load_state_dict(self.policy_continuous.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

        self.replay_buffer.clear()

        return policy_loss_discrete.item(), policy_loss_continuous.item(), value_loss.item()
    

    def select_action(self, state, layer_number=None, actionc=None, actiond=None):

        if type(state) in [np.array, np.ndarray]:
            #state = torch.from_numpy(state).flatten().unsqueeze(0).to(torch.float32)
            state = torch.from_numpy(state).unsqueeze(0).to(torch.float32)

        if self.pre_type == "linear":
            state = state.flatten(1)
        
        if layer_number is not None:
            if type(layer_number) in [np.array, np.ndarray]:
                layer_number = torch.from_numpy(layer_number).flatten().unsqueeze(0)
                
            layer_number = layer_number.to(torch.int)

        #print(state.size(), layer_number.size())
        pre_output_d = self.pre_network(state, layer_number)
        pre_output_c = self.pre_network(state, layer_number)
        pre_output_v = self.pre_network(state, layer_number)
        d_probs = self.policy_discrete(pre_output_d, layer_number)
        c_means, c_std = self.policy_continuous(pre_output_c, layer_number)

        state_value = self.value(pre_output_v, layer_number)
        #d_onehot = F.one_hot(d_probs, num_classes=policy.output_dim_discrete)
        d = torch.distributions.Categorical(d_probs)

        if actiond is None:
            actiond = d.sample()

        
        c = TruncatedNormalDist(
            c_means, 
            c_std, 
            self.lower_bound, 
            self.upper_bound)
 
        if actionc is None:
            actionc = c.sample()

        #actionc[actionc == self.lower_bound] += 1e-4
        #actionc[actionc == self.upper_bound] -= 1e-4

        #print(d_probs.size(), actiond.size(), actionc.size(), c_means.size())
        
        log_prob_discrete = d.log_prob(actiond) 
        log_prob_continuous = torch.sum(c.log_prob(actionc), dim=-1)#[:, actiond.detach()]

        # get the continuous action for sampled discrete element
        
        c_action = actionc.detach()[:,actiond.detach()]

        #print(actiond.unsqueeze(0).T.size(), actionc.size())
        action = torch.cat([actiond.detach().unsqueeze(0).T, c_action], dim=-1)[0]

        entropy_discrete = d.entropy() 
        entropy_continuous = torch.sum(c._entropy, dim=-1)

        #policy.saved_log_probs.append(log_prob)

        return action, actiond, actionc, log_prob_discrete, log_prob_continuous, d_probs, c_means, c_std, state_value, entropy_discrete, entropy_continuous


    def save_networks(self, save_directory, epoch=None):

        discrete_policy_fname = os.path.join(save_directory, "discrete_policy.pt")
        torch.save({
            "epoch":epoch,
            "model_state_dict":self.policy_discrete.state_dict(),
            "optimiser_state_dict":self.optimiser_discrete.state_dict()
        }, discrete_policy_fname)

        continuous_policy_fname = os.path.join(save_directory, "continuous_policy.pt")
        torch.save({
            "epoch":epoch,
            "model_state_dict":self.policy_continuous.state_dict(),
            "optimiser_state_dict":self.optimiser_continuous.state_dict()
        }, continuous_policy_fname)

        value_fname = os.path.join(save_directory, "value.pt")
        torch.save({
            "epoch":epoch,
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





