from collections import namedtuple, deque
import random
import torch
from torch.nn import functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

class PreNetworkAttention(torch.nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=2, num_layers=2):
        super(PreNetworkAttention, self).__init__()
        self.embedding = torch.nn.Linear(input_dim, embed_dim)
        encoder_layers = torch.nn.TransformerEncoderLayer(embed_dim, num_heads)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = torch.nn.Linear(embed_dim, output_dim)
        
    def forward(self, x, layer_number=None):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Change to shape (seq_len, batch_size, embed_dim) for Transformer encoder
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Change back to shape (batch_size, seq_len, embed_dim)
        x = self.fc(x)
        if layer_number != None:
            indices = layer_number.flatten().view(x.size(0), 1, 1).to(torch.int64)
            indices = indices.expand(x.size(0), 1, x.size(2))
            #x = torch.gather(x, 1, indices)
            x = torch.mean(x, dim=1)  # Global average pooling
        else:
            x = torch.mean(x, dim=1)  # Global average pooling
        return x.flatten(start_dim=1)
    
class PreNetworkLinear(torch.nn.Module):
    def __init__(
            self, 
            input_dim, 
            output_dim, 
            hidden_dim, 
            lower_bound=0, 
            upper_bound=1,):
        super(PreNetworkLinear, self).__init__()
        self.output_dim = output_dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.input = torch.nn.Linear(input_dim, hidden_dim)

        self.affine1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.affine2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.affine3 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.output = torch.nn.Linear(hidden_dim, output_dim)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x, layer_number=None):
        x = x.flatten(1)
        if layer_number is not None:
            x = torch.cat([x, layer_number], dim=1)
        x = self.input(x)
        #x = self.dropout(x)
        x = F.relu(x)
        x = self.affine1(x)
        x = F.relu(x)
        x = self.affine2(x)
        x = F.relu(x)
        x = self.affine3(x)
        x = F.relu(x)
        #x = self.dropout(x)
        out= self.output(x)
        
        return out

# setup a simple q network 
class QNetwork(torch.nn.Module):

    def __init__(self, n_observations, n_actions, hidden_size):
        super(QNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x)

Experience = namedtuple('Experience', ['state', 'obs', 'action', 'reward', 'next_state', 'next_obs', 'done'])

class ReplayBuffer:
    def __init__(self, buffer_size):
        # this has a buffer of the about quantities of some desired length (usually restricted by memory)
        self.buffer = deque(maxlen=buffer_size)

    def add_experience(self, experience):
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Deep Q-Learning agent class
class DQNAgent:
    def __init__(self, 
                state_dim,
                action_size, 
                learning_rate=0.001, 
                gamma=0.99, 
                epsilon_start=1.0, 
                epsilon_decay=0.995, 
                epsilon_min=0.01, 
                buffer_size=10000, 
                batch_size=64, 
                hidden_size=128, 
                update_frequency=100,
                use_attn=False,
                n_heads=4,
                n_attn_layers=2):
        
        # environment parameters
        self.state_dim = state_dim
        self.state_size = np.prod(state_dim)
        self.action_size = action_size
        # parameters associated with the action choice probability
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # discount factor
        self.gamma = gamma
        self.update_frequency = update_frequency
        self.pre_output_dim = hidden_size
        self.use_attn = use_attn

        if use_attn:
            self.pre_network = PreNetworkAttention(
            state_dim[-1],
            self.pre_output_dim,
            hidden_size,
            num_heads=n_heads,
            num_layers=n_attn_layers
        )
            self.target_pre_network = PreNetworkAttention(
            state_dim[-1],
            self.pre_output_dim,
            hidden_size,
            num_heads=n_heads,
            num_layers=n_attn_layers
        )
        else:
            self.pre_network = PreNetworkLinear(
                np.prod(state_dim),
                hidden_size,
                self.pre_output_dim
            )
            self.target_pre_network = PreNetworkLinear(
                np.prod(state_dim),
                hidden_size,
                self.pre_output_dim
            )
        
        # Q-networks 
        self.q_network = QNetwork(self.pre_output_dim, action_size, hidden_size)
        self.target_q_network = QNetwork(self.pre_output_dim, action_size, hidden_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        self.target_pre_network.load_state_dict(self.pre_network.state_dict())
        self.target_pre_network.eval()

        # Optimizer
        self.optimiser = torch.optim.AdamW(
                            list(self.q_network.parameters()) + list(self.pre_network.parameters()),
                            lr=learning_rate)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

    def select_action(self, state):
        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size), None
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state)
                pre_output = self.pre_network(state)
                q_values = self.q_network(pre_output)
                return int(torch.argmax(q_values)), None

    def train(self):
        # Sample a batch from experience replay
        batch = self.replay_buffer.sample_batch(self.batch_size)

        states, obss, actions, rewards, next_states, next_obss, dones = zip(*batch)


        states = torch.FloatTensor(states).to(torch.float32)
        obss = torch.FloatTensor(obss).to(torch.float32)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards).to(torch.float32)
        next_states = torch.FloatTensor(next_states).to(torch.float32)
        next_obss = torch.FloatTensor(next_obss).to(torch.float32)
        dones = torch.FloatTensor(dones)

        pre_output = self.pre_network(obss)
        next_pre_output = self.pre_network(next_obss)

        # Q-values for current states
        q_values = self.q_network(pre_output)

        # Q-values for next states using target Q-network
        next_q_values = self.target_q_network(next_pre_output).detach()

        # Compute target Q-values
        # this takes the maxmimum over the 
        target_q_values = rewards + self.gamma * (1 - dones) * torch.max(next_q_values, dim=1)[0]
        # Compute loss
        # this is the difference between the target qvalues and those from one (or more) step in the past
        loss = torch.nn.MSELoss()(q_values.gather(1, actions.view(-1, 1)), target_q_values.unsqueeze(1))

        # Backpropagation
        self.optimiser.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimiser.step()

        return loss.item()

    def update_target_network(self):
        # Update target Q-network with the weights of the current Q-network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_pre_network.load_state_dict(self.pre_network.state_dict())

    def memorize(self, state, obs, action, reward, next_state, next_obs, done):
        # Add experience to replay buffer
        self.replay_buffer.add_experience(Experience(state, obs, action, reward, next_state, next_obs, done))

    def save_model(self, filename):

        torch.save({
            "pre_network_state_dict": self.pre_network.state_dict(),
            "q_network_state_dict": self.q_network.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }, 
        filename)





class DQNTrainer():

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

        self.memory = ReplayBuffer(int(1e4))

    def train(self,):

        # Training loop
        rewards = []
        losses = []
        game_lengths = []
        returns = []
        epsilons = []
        test_values = []

        opt_state = None
        opt_reward = -np.inf

        if not os.path.isdir(os.path.join(self.root_dir, "stackplots")):
            os.makedirs(os.path.join(self.root_dir, "stackplots"))

        dqn_game_lengths = []
        for i_episode in range(self.episodes):
            state  = self.env.reset()
            #popt.reset()
            done = False
            total_reward = 0
            it = 0
            while not done:
                # select the action based on the dqn model
                fl_state = np.array([state.flatten(),])[0]
                obs = self.env.get_observation_from_state(state)
                if self.useobs:
                    obs = obs
                else:
                    obs = state
                action, _ = self.agent.select_action(np.expand_dims(obs, 0))
                # take the action in the enviromnment and return next state and reward

                material = int(action // len(self.env.thickness_options))
                thickness = self.env.thickness_options[int(action % len(self.env.thickness_options)) -1]
                
                actions = [material, thickness]

                next_state, reward, terminated, finished, _, _ = self.env.step(actions)
                done = terminated or finished
                next_obs = self.env.get_observation_from_state(next_state)

                # save the actions, rewards and states to memory
                self.agent.memorize(state, obs, action, reward, next_state, next_obs, done)

                # update the state with the new one
                state = next_state
                
                #Place a limit on how many moves the agent can make in training
                if it > 1000:
                    done = True
                # save the total reward (return)
                total_reward += reward
                it += 1

            # have a little delay before training so some data can be saved to memory (can set this to batch size)
                if i_episode > self.batch_size:
                    loss = self.agent.train()
                else:
                    loss = 0

            if self.agent.epsilon > self.agent.epsilon_min:
                self.agent.epsilon *= self.agent.epsilon_decay
                epsilons.append(self.agent.epsilon)

            if it % self.agent.update_frequency == 0:
                self.agent.update_target_network()
            
            game_lengths.append(it)
            returns.append(total_reward)
            losses.append(loss)
            rewards.append(total_reward)

            if opt_reward < total_reward:
                opt_reward = total_reward
                opt_state = state
                opt_value = self.env.compute_state_value(state)
                fig, ax = self.env.plot_stack(state)
                ax.set_title(f"Optimal rew: {opt_reward}, opt val: {opt_value}")
                fig.savefig(os.path.join(self.root_dir,  f"best_state.png"))


            if i_episode % 300 == 0:
                
                print("Episode", i_episode, total_reward)
                self.loss_plot(losses, rewards, epsilons, savefig=True)

                c_value = self.env.compute_state_value(state)
                fig, ax = self.env.plot_stack(state)
                ax.set_title(f"Optimal rew: {total_reward}, opt val: {c_value}")
                fig.savefig(os.path.join(self.root_dir, "stackplots", f"state_{i_episode}.png"))
                    


    def loss_plot(self, losses, rewards, epsilons, savefig=False, window_size=100):
        
        fig, ax = plt.subplots(nrows=3)
        times = np.arange(len(losses))
        rtimes = np.arange(len(rewards))
        downsamp_rewards = np.mean(np.reshape(rewards[:int((len(rewards)//window_size)*window_size)], (-1,window_size)), axis=1)
        downsamp_times = np.mean(np.reshape(rtimes[:int((len(rtimes)//window_size)*window_size)], (-1,window_size)), axis=1)
        ax[0].plot(times, losses, label="Loss")
        ax[1].plot(rewards, label="Rewards")
        ax[1].plot(downsamp_times, downsamp_rewards, label="av rewards")
        ax[2].plot(epsilons, label="Epsilon")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
    

        if savefig:
            fig.savefig(os.path.join(self.root_dir, "losses.png"))
