from FACL import FACL
import numpy as np
from numpy import savetxt

# This class inherits FACL and implements the :
# reward function
# state update
# saves the path the agent is taking in a given epoch
# resets the game after an epoch


class TestController(FACL):

    def __init__(self, state, max, min, num_mf):
        self.state = state.copy()
        self.path = state.copy()
        self.initial_position = state.copy()
        self.territory_coordinates = [20, 20]  # these will eventually be in the game class and passed into the actor
        self.r = 1 #radius of the territory
        self.v = 0.75  # velocity
        self.distance_away_from_target_t_plus_1 = 0 #this gets set later
        self.distance_away_from_target_t = self.distance_from_target()
        self.reward_track =[] # to keep track of the rewards
        FACL.__init__(self, max, min, num_mf) #explicit call to the base class constructor
        self.fuzzy_info_max = max
        self.fuzzy_info_min = min
        self.fuzzy_info_nmf = num_mf
    def get_reward(self):
        self.distance_away_from_target_t_plus_1 = self.distance_from_target()
        if (abs(self.state[0]  - self.territory_coordinates[0]) <= self.r and abs(self.state[1] - self.territory_coordinates[1]) <= self.r):
            r = 0

        else:
            r = (self.distance_away_from_target_t - self.distance_away_from_target_t_plus_1)
        # print("reward", self.distance_away_from_target_t, '-', self.distance_away_from_target_t_plus_1, '=', r)
        self.distance_away_from_target_t = self.distance_away_from_target_t_plus_1
        # heading_desired = np.arctan( (self.territory_coordinates[1] - self.state[1]) / (self.territory_coordinates[0] - self.state[0]))
        # heading_error = heading_desired - self.u_t
        # r = 6*np.exp(-(heading_error/0.5)**2)-3
        # print('reward', 'exp(-', heading_error, '/0.5)^2)-3 = ',r )
        self.update_reward_graph(r)
        return r

    def update_state(self):
        self.state[0] = self.state[0] + self.v * np.cos(self.u_t)
        self.state[1] = self.state[1] + self.v * np.sin(self.u_t)
        self.update_path(self.state)
        pass

    def reset(self):
        # Edited for each controller
        self.state = self.initial_position.copy()
        self.path = []
        self.path = self.initial_position.copy()
        self.reward_track = []
        self.distance_away_from_target_t = self.distance_from_target()
        pass

    def update_path(self, state):

        self.path = np.vstack([self.path, state])
        pass

    def update_reward_graph(self, r):
        self.reward_track.append(r)

    def distance_from_target(self):
        distance_away_from_target = np.sqrt(
            (self.state[0] - self.territory_coordinates[0]) ** 2 + (self.state[1] - self.territory_coordinates[1]) ** 2)
        return distance_away_from_target
    def save(self):
        # save the actor weight list
        savetxt('actor_weights.csv', self.omega, delimiter=',')
        # save the critic weight list
        savetxt('critic_weights.csv', self.zeta, delimiter=',')
        # save the fuzzy system information
        # savetxt('fuzzy_info.txt',self.fuzzy_info)
        np.savetxt("fuzzy_info.txt",self.fuzzy_info_max, fmt='%1.3f', newline="\n")
        with open("fuzzy_info.txt", "a") as f:
             np.savetxt(f, self.fuzzy_info_min, fmt='%1.3f', newline="\n")
             np.savetxt(f, self.fuzzy_info_nmf,fmt='%1.3f', newline="\n")

        pass
    def load(self):
        self.omega = np.loadtxt('actor_weights.csv', delimiter=',')
        self.zeta = np.loadtxt('critic_weights.csv', delimiter=',')