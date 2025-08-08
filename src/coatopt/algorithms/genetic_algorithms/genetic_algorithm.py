import numpy as np

class StatePool():

    def __init__(self, environment, n_states=50, states_fraction_keep = 0.1, fraction_random_add=0.0):
        self.environment = environment
        self.n_states = n_states
        self.states_fraction_keep = states_fraction_keep
        self.fraction_random_add = fraction_random_add
        #if self.n_states % self.n_keep_states != 0:
        #    raise Exception(f"keep states must divide into n states")
  
        self.current_states = self.get_new_states(self.n_states)
        self.current_values = np.concatenate([np.arange(len(self.current_states))[:,np.newaxis], -np.inf*np.ones((self.n_states, 1))], axis=1)
        print(np.shape(self.current_values))

    def order_states(self, ):
        """set the order of states based on their value

        Returns:
            array: array of sorted state values
        """
        state_values = np.zeros((self.n_states, 2))
        for i in range(self.n_states):
            temp_stval, vals, rewards = self.environment.compute_reward(self.current_states[i])

            if np.isnan(temp_stval) or np.isinf(temp_stval):
                temp_stval = -1000
            state_values[i] = [i,temp_stval]

        sorted_state_values = sorted(state_values, key=lambda a: (-np.nan_to_num(a[1], nan=-1001)))

        return np.array(sorted_state_values)
    
    def check_state_possible(self, state):
        """make sure that adjacent layers cannot have the same material

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        current_material = -1
        possible = True
        for layer in state:
            new_material = np.argmax(layer[1:])
            if new_material == current_material:
                possible = False
                break
            current_material = new_material

        return possible

    def evolve_state(self, state):
        """Evolve a state, pick a random action and apply this to a state

        Args:
            state (int): action choice

        Returns:
            array: new state to pick
        """
        
        #action = np.random.randint(self.environment.n_actions)
        #actions = self.environment.get_actions(action)
        #new_state = self.environment.get_new_state(state, actions)
        action = self.environment.sample_action_space(state)
        #new_state, reward = self.environment.step(state, action)
        new_state, rewards, terminated, finished, reward, full_action = self.environment.step(action, state=state, layer_index=action[2], always_return_value=True)
        #new_state, rewards = self.environment.step(state, action)

        #new_state = self.environment.sample_state_space()
        #done = self.check_state_possible(new_state)
        return new_state, reward
    
    def state_crossover(self, states):
        """swap some attributes of the given states

        Args:
            states (array): set of states 

        Returns:
            array: set of states
        """
        n_states, n_layers, n_features = np.shape(states)
        # creates indicies for each of the example states and shuffles them
        data_inds = np.arange(n_states)
        num_swaps = 3
        nswitch = int(n_states/2)
        for i in range(num_swaps):
            np.random.shuffle(data_inds)
            # define that half of states will switch a layer with another state
            ninds_switch_from = data_inds[:nswitch]
            ninds_switch_to = data_inds[nswitch:]
            layers = np.random.randint(n_layers, size=nswitch)
            states[ninds_switch_from, layers] = states[ninds_switch_to, layers]

        return states

    def evolve_step(self,):
        """Evolve one step, by sorting states, taking top n% duplicating them and applying crossover

        Returns:
            _type_: _description_
        """
        sorted_state_values = self.order_states()
        n_keep_states = int(self.states_fraction_keep * self.n_states)
        top_state_values = sorted_state_values[:n_keep_states]
        top_state = self.current_states[top_state_values[0,0].astype(int)]
        top_states = self.current_states[top_state_values[:,0].astype(int)]
        self.current_states = np.tile(top_states, (np.ceil(self.n_states/n_keep_states).astype(int), 1, 1))[:self.n_states]
        self.current_states = self.state_crossover(self.current_states)
        
        if self.fraction_random_add != 0 :
            num_random_add = int(self.n_states * self.fraction_random_add())
            self.current_states[-num_random_add:] = self.get_new_states(num_random_add)

        for i in range(self.n_states):
            self.current_states[i], self.current_values[i] = self.evolve_state(self.current_states[i])

        return top_state_values, top_state

    def get_new_states(self, N):
        """Get N new random states

        Args:
            N (int): number of states to generatae

        Returns:
            array: set of states
        """
        states = []
        for i in range(N):
            new_state = self.environment.sample_state_space()
            states.append(new_state)
        return np.array(states)