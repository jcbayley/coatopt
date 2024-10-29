import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import time
import matplotlib.pyplot as plt
import copy 
from coatopt.environments.thermal_noise_environment import CoatingStack
#from simple_environment import CoatingStack
from itertools import count
from coatopt.networks import hppo
import itertools

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

def plot_state(state):
    thicknesses = state[:,0]
    materials = np.argmax(state[:,1:], axis=1)

    colors = ["k", "C0", "C1", "C2", "C3", "C4"]

    fig,ax = plt.subplots()
    scoord = 0
    for i in range(len(state)):
        ax.bar(scoord, 1, color=colors[materials[i]], width=thicknesses[i], align="edge")
        scoord += thicknesses[i]

    return fig

def training_loop(
        outdir, 
        env, 
        agent, 
        max_episodes=1000, 
        batch_size=256, 
        n_ep_train=10, 
        max_layers=4, 
        lower_bound=0,
        upper_bound=1, 
        useobs=False,
        beta_start=1.0,
        beta_end=0.001,
        beta_decay_length=None,
        beta_decay_start=0,
        save_interval=10):

    # Training loop
    rewards = []
    values = []
    losses_pold = []
    losses_polc = []
    losses_val = []
    all_means = []
    all_stds = []
    all_mats = []
    max_reward = -np.inf
    max_state = None

    state_dir = os.path.join(outdir, "states")
    if not os.path.isdir(state_dir):
        os.makedirs(state_dir)

    betas = []
    lrs = []

    for episode in range(max_episodes):
        if episode > 0:
            update_policy = True
        else:
            update_policy=True
        if beta_decay_length is not None and episode > beta_decay_length:
            update_value=True
        else:
            update_value=True

        if beta_decay_length is not None and episode > beta_decay_start:
            agent.beta = beta_start - (beta_start-beta_end)*np.min([(episode-beta_decay_start)/beta_decay_length, 1])
        else:
            agent.beta = beta_start

            #agent.optimiser_discrete.learning_rate = new_lr
            #agent.optimiser_continuous.learning_rate = new_lr
            #agent.optimiser_value.learning_rate = new_lr

        betas.append(agent.beta)
        #lrs.append(agent.optimiser_discrete.learning_rate)

        states = []
        actions_discrete = []
        actions_continuous = []
        returns = []
        advantages = []
        for n in range(n_ep_train):
            state = env.reset()
            episode_reward = 0
            means = []
            stds = []
            mats = []
            t_rewards = []
            for t in range(100):
                # Select action
                fl_state = np.array([state.flatten(),])[0]
                obs = env.get_observation_from_state(state)
                if useobs:
                    obs = obs
                else:
                    obs = state
                t = np.array([t])
                
                action, actiond, actionc, log_prob_discrete, log_prob_continuous, d_prob, c_means, c_std, value, entropy_discrete, entropy_continuous = agent.select_action(obs, t)

                action[1] = action[1]*(upper_bound - lower_bound) + lower_bound
                """
                if t == 0:
                    action[0] = 2
                if t == 1:
                    action[0] = 1
                if t == 2:
                    action[0] = 2
                if t == 3:
                    action[0] = 1
                """
                # Take action and observe reward and next state
                next_state, reward, done, finished, _, full_action = env.step(action)

                t_rewards.append(reward)
                agent.replay_buffer.update(
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
                opt_value = env.compute_state_value(max_state)
                fig, ax = env.plot_stack(max_state)
                ax.set_title(f"Optimal rew: {max_reward}, opt val: {opt_value}")
                fig.savefig(os.path.join(outdir,  f"best_state.png"))

                agent.save_networks(outdir)

            returns = agent.get_returns(t_rewards)
            agent.replay_buffer.update_returns(returns)
            

        all_means.append(means)
        all_stds.append(stds)
        all_mats.append(mats)

        if episode > 10:
            loss1, loss2, loss3 = agent.update(update_policy=update_policy, update_value=update_value)
            lr_outs = agent.scheduler_step()
            lrs.append(lr_outs)
            losses_pold.append(loss1)
            losses_polc.append(loss2)
            losses_val.append(loss3)
            agent.replay_buffer.clear()

        rewards.append(episode_reward)
        values.append(env.compute_state_value(state))

        if episode % 20 == 0 and episode !=0 :
            reward_fig, reward_ax = plt.subplots(nrows=4, figsize=(7,9))
            window_size = 20
            downsamp_rewards = np.mean(np.reshape(rewards[:int((len(rewards)//window_size)*window_size)], (-1,window_size)), axis=1)
            reward_ax[0].plot(np.arange(episode+1), rewards)
            reward_ax[0].plot(np.arange(episode).reshape(-1,window_size)[:,0], downsamp_rewards)
            reward_ax[0].set_xlabel("Episode number")
            reward_ax[0].set_ylabel("Reward (Reflectivity)")

            downsamp_values = np.mean(np.reshape(values[:int((len(values)//window_size)*window_size)], (-1,window_size)), axis=1)
            reward_ax[1].plot(np.arange(episode+1), values)
            reward_ax[1].plot(np.arange(episode).reshape(-1,window_size)[:,0], downsamp_values)
            reward_ax[1].set_xlabel("Episode number")
            reward_ax[1].set_ylabel("Reflectivity (Reflectivity)")

            reward_ax[2].plot(np.arange(episode+1), betas)
            reward_ax[2].set_xlabel("Episode number")
            reward_ax[2].set_ylabel("Entropy weight , beta param")

            reward_ax[3].plot(np.arange(len(lrs)) + 10, np.array(lrs)[:,0], label="discrete")
            reward_ax[3].plot(np.arange(len(lrs)) + 10, np.array(lrs)[:,1], label="continuous")
            reward_ax[3].plot(np.arange(len(lrs)) + 10, np.array(lrs)[:,2], label="value")
            reward_ax[3].set_xlabel("Episode number")
            reward_ax[3].set_ylabel("Learning Rate")
            reward_ax[3].legend()
            reward_fig.savefig(os.path.join(outdir, "running_rewards.png"))


            loss_fig, loss_ax = plt.subplots(nrows=3)
            loss_ax[0].plot(losses_pold)
            loss_ax[1].plot(losses_polc)
            loss_ax[2].plot(losses_val)
            loss_ax[0].set_ylabel("Policy discrete loss")
            loss_ax[1].set_ylabel("Policy continuous loss")
            loss_ax[2].set_ylabel("Value loss")
            loss_ax[2].set_yscale("log")
            loss_ax[2].set_xlabel("Episode number")
            loss_fig.savefig(os.path.join(outdir, "running_losses.png"))

            """
            n_layers = max_layers
            loss_fig, loss_ax = plt.subplots(nrows = n_layers)
            all_means2 = pad_lists(all_means, np.nan, n_layers)
            all_stds2 = pad_lists(all_stds, np.nan, n_layers)
            for i in range(n_layers):
                loss_ax[i].plot(all_means2[:,i])
                loss_ax[i].plot(all_stds2[:,i])
            loss_fig.savefig(os.path.join(outdir, "running_means.png"))
            """
            
            n_layers = max_layers
            loss_fig, loss_ax = plt.subplots(nrows = n_layers)
            all_mats2 = pad_lists(all_mats, [0.0,]*env.n_materials, n_layers)

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
            loss_fig.savefig(os.path.join(outdir, "running_mats.png"))
            
            # Print episode information
            print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

            if episode % 100 == 0:
                fig, ax = env.plot_stack(state)
                t_opt_value = env.compute_state_value(state)
                ax.set_title(f", opt val: {t_opt_value}")
                fig.savefig(os.path.join(outdir,  "states", f"episode_{episode}.png"))
            

    print("Max_state: ", max_reward)
    print(max_state)

    return rewards, max_state

if __name__ == "__main__":
    root_dir = "./hppo_output/tmm_obs_sparse_test_hppoc_lstm_2l_4h_8layer_init4layer_mse_b256_adam_lr1e-3-5"
    load_model_dir = ".//hppo_output/tmm_obs_sparse_test_hppoc_lstm_2l_4h_4layer_mse_b256_adam_lr1e-3-5"

    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    n_layers = 8
    min_thickness = 1e-9
    max_thickness = 300e-9
    #min_thickness = 0.1
    #max_thickness = 1
    load_model = False
    useobs=True

    materials = {
        0:{
            'name' : 'air', 
            'n'    : 1,
            'a'    : 0,
            'alpha': np.NaN,
            'beta' : np.NaN,
            'kappa': np.NaN,
            'C'    : np.NaN,
            'Y'    : np.NaN,
            'prat' : np.NaN,
            'phiM' : np.NaN,
            'k'    : 0
            },
        1: {
            'name' : 'SiO2',
            'n'    : 1.44,
            'a'    : 0,
            'alpha': 0.51e-6,
            'beta' : 8e-6,
            'kappa': 1.38,
            'C'    : 1.64e6,
            'Y'    : 72e9,
            'prat' : 0.17,
            'phiM' : 4.6e-5,
            'k'    : 1
        },
        2: {
            'name' : 'ta2o5',
            'n'    : 2.07,
            'a'    : 2,
            'alpha': 3.6e-6,
            'beta' : 14e-6,
            'kappa': 33,
            'C'    : 2.1e6,
            'Y'    : 140e9,
            'prat' : 0.23,
            'phiM' : 2.44e-4,
            'k'    :1
        },
        3: {
            'name' : 'new',
            'n'    : 1.67,
            'a'    : 1,
            'alpha': 3.6e-6,
            'beta' : 14e-6,
            'kappa': 33,
            'C'    : 2.1e6,
            'Y'    : 140e9,
            'prat' : 0.23,
            'phiM' : 2.44e-4,
            'k'    :1
        },
    }

    thickness_options = [0.0]

    env = CoatingStack(
        n_layers, 
        min_thickness, 
        max_thickness, 
        materials,
        opt_init=False)

    
    num_iterations = 10000

    device = "cpu"

    insize = env.obs_space_size if useobs==True else env.state_space_size

    agent = hppo.HPPO(
            env.obs_space_shape,
            env.n_materials, 
            env.n_materials,
            hidden_size=32,
            disc_lr_policy=2e-3,
            cont_lr_policy=1e-3, 
            lr_value=5e-3,
            lr_step=500,
            lr_min=1e-5,
            lower_bound=0,
            upper_bound=1,
            n_updates=5,
            beta=0.2,
            clip_ratio=0.01,
            gamma=0.999,
            include_layer_number=True,
            pre_type="lstm",
            n_heads=2,
            n_attn_layers=2,
            optimiser="adam"
            )
    
    if load_model_dir is not None:
        agent.load_networks(load_model_dir)
    
    if load_model:
        agent.load_networks(root_dir)


    optimal_state = env.get_optimal_state()
    optimal_value = env.compute_state_value(optimal_state)
    fig, ax = env.plot_stack(optimal_state)
    ax.set_title(f" opt val: {optimal_value}")
    fig.savefig(os.path.join(root_dir,  f"opt_state.png"))
    
    
    rewards, max_train_state = training_loop(
        root_dir, 
        env, 
        agent, 
        max_episodes=num_iterations,
        n_ep_train=256,
        max_layers=n_layers,
        useobs=useobs,
        beta_start=0.01,
        beta_end=0.01,
        beta_decay_length=1000,
        beta_decay_start=100,
        lr_start=1e-3,
        lr_end=1e-3,
        lr_decay_length=4000,
        upper_bound=max_thickness,
        lower_bound=min_thickness,
        save_interval=10)

    episode_rewards = []
    episode_returns = []
    states = []
    for i in range(100):
        state = env.reset()
        episode_return = 0
        for t in range(100):
            # Select action
            #action = agent.select_action(fl_state)
            fl_state = np.array([state.flatten(),])
            obs = env.get_observation_from_state(state)
            if useobs:
                obs = obs
            else:
                obs = fl_state
            t = np.array([t])
            action, actiond, actionc, log_prob_d, log_prob_c, d_prob, c_means, c_std, value, entropy_d, entropy_c = agent.select_action(obs, t)

            action[1] = action[1]*(max_thickness - min_thickness) + min_thickness
            # Take action and observe reward and next state
            next_state, reward, done, finished, _, full_action = env.step(action)


            # Update state and episode reward
            state = next_state
            episode_return += reward

            if done or finished:
                break
        
        episode_rewards.append(reward)
        episode_returns.append(episode_return)
        states.append(state)

    maxind = -1#np.argmax(episode_returns)
    print(states[maxind])
    print("Return: ", episode_rewards[maxind])
    print("Reward: ", episode_rewards[maxind])

    maxind = np.argmax(episode_returns)
    print(states[maxind])
    print("Max Return: ", episode_rewards[maxind])
    print("Max Reward: ", episode_rewards[maxind])

    fig, ax = plt.subplots()
    ax.hist(episode_rewards)
    fig.savefig(os.path.join(root_dir, "./return_hist.png"))

    thickness1 = 1064e-9 /(4*materials[1]["n"])
    thickness2 = 1064e-9 /(4*materials[2]["n"])
    print("thickness", thickness1, thickness2)
    
    opt_state = []
    material = 2
    for i in range(n_layers):
        if material == 1:
            thickness = thickness1
        elif material == 2:
            thickness = thickness2
        l_state = [0,]*(env.n_materials+1)
        l_state[0] = thickness
        l_state[material+1] = 1

        opt_state.append(l_state)
        if material == 1: material = 2
        elif material == 2: material = 1
    opt_state2 = []
    material = 1
    for i in range(n_layers):
        if material == 1:
            thickness = thickness1
        elif material == 2:
            thickness = thickness2
        l_state = [0,]*(env.n_materials+1)
        l_state[0] = thickness
        l_state[material+1] = 1

        opt_state2.append(l_state)
        if material == 1: material = 2
        elif material == 2: material = 1

    opt_reward = env.compute_state_value(opt_state)
    opt_reward2 = env.compute_state_value(opt_state2)
    max_reward = env.compute_state_value(states[maxind])
    max_train_reward = env.compute_state_value(max_train_state)
    print("Opt state: ", opt_reward, opt_reward2)
    print("Max test state: ", max_reward)
    print("Max train state: ", max_train_reward)

    with open(os.path.join(root_dir, f"max_train_state_{max_train_reward}.txt"), "w") as f:
        np.savetxt(f, max_train_state)

    with open(os.path.join(root_dir, f"max_test_state_{max_reward}.txt"), "w") as f:
        np.savetxt(f, states[maxind])

    with open(os.path.join(root_dir, f"opt_state_{opt_reward}.txt"), "w") as f:
        np.savetxt(f, opt_state)


    print(np.round(opt_state, 1))
    print(np.round(max_train_state))

    """
    max_state = np.array([[thickness1, 0, 1, 0],
                          [thickness2, 0, 0, 1],
                          [thickness1, 0, 1, 0],
                          [thickness2, 0, 0, 1]])
    
    max_reward = env.compute_state_value(max_state)
    max_reward_ligo = env.compute_state_value_ligo(max_state)
    max_reward_ligo2 = env.compute_state_value_ligo(max_state[::-1])

    max_state1 = np.array([[thickness2*1e9, 0, 0, 1],
                          [thickness1*1e9, 0, 1, 0],
                          [thickness2*1e9, 0, 0, 1],
                          [thickness1*1e9, 0, 1, 0]])
    
    max_reward1 = env.compute_state_value(max_state1)
    max_reward1_ligo = env.compute_state_value_ligo(max_state1)
    max_reward1_ligo2 = env.compute_state_value_ligo(max_state1[::-1])

    max_state2 = np.array([[thickness2*1e9, 0, 0, 1],
                          [thickness1, 0, 1, 0],
                          [thickness2*1e9, 0, 0, 1],
                          [thickness1, 0, 1, 0]])
    
    max_reward2 = env.compute_state_value(max_state2)

    max_state3 = np.array([[thickness1, 0, 1, 0],
                          [thickness1, 0, 1, 0],
                          [thickness1, 0, 1, 0],
                          [thickness1, 0, 1, 0]])
    
    max_reward3 = env.compute_state_value(max_state3)
    print("optimal reward", max_reward, max_reward1)
    print("nonopt: ", max_reward2, max_reward3)

    print("ligocomp1", max_reward1, max_reward1_ligo, max_reward1_ligo2)
    print("ligocomp0", max_reward, max_reward_ligo, max_reward_ligo2)
    """

