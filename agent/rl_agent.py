import random 
import numpy as np 

# rl just needs to calculate the next step(action) for the camera setting. 
# it doesn't have to return defect probability where it can be obtained by the features in
# analyzer when it captures the image

class rl_agent:
    def __init__(self, actions, learning_rate = 0.1, discount_factor = 0.9, epsilon = 0.2):
        self.actions = actions  # list 
        
        self.q_table = {}
        self.lr = learning_rate 
        self.gamma = discount_factor
        self.epsilon = epsilon

    
    def state_to_key(self, state):

        if isinstance(state, dict):
            return(
                round(float(state.get("brightness", 0.0)), 2),
                round(float(state.get("contrast", 0.0)) , 2),
                round(float(state.get("sharpness", 0.0)) , 2),
            )   
        return tuple(np.round(state, 2)) if isinstance(state, (list, np.ndarray)) else state;

    

    def choose_action(self, state):
        #state with tuble -> imuatble
        state_key = self.state_to_key(state)
        #epsilon is bigger , more explore
        # random learning
        if random.random() < self.epsilon or state_key not in self.q_table:
            action = random.choice(self.actions)
        
        # based on what it has been learning
        else:
            #reward per the key obtained by state- state stored as keys with reward(integer)
            # compares the value and return the key with highest value
            best_idx = max(self.q_table[state_key], key = self.q_table[state_key].get)
            action= self.actions[best_idx]

        return action
    
    def learn(self, state, action, reward, next_state=None):
        state_key = self.state_to_key(state)
        next_key = self.state_to_key(next_state) if next_state is not None else state_key
        action_index = self.actions.index(action)


        if state_key not in self.q_table:
            self.q_table[state_key] = {i: 0.0 for i in range(len(self.actions))}
        if next_key not in self.q_table:
            self.q_table[next_key] = {i: 0.0 for i in range(len(self.actions))}

        current_q = self.q_table[state_key][action_index]
        next_max_q = max(self.q_table[next_key].values())


        #Q-learning update rule
        new_q = current_q + self.lr*(reward + self.gamma*next_max_q - current_q)
        self.q_table[state_key][action_index] = new_q

        