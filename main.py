
import numpy as np
import FACL
from Agent import Agent
from TestController import TestController
from TestController2 import TestController2
# creation of sharon, the agent who goes to a location
state = [5, 5] # start position on the grid
state_max = [50, 50] # max values of the grid [x,y]
state_min = [-50, -50] # smallest value of the grid [x,y]
num_of_mf = [19, 19] # breaking up the state space (grid in this case) into 29 membership functions
action_space = [2.356125, -1.57075, -0.785375, 0, 0.785375, 1.57075, -2.356125]

selection = 0

if(selection == 0) :
    #FACL TEST
    FACLcontroller = TestController(state, state_max, state_min, num_of_mf) #create the FACL controller
    sharon = Agent(FACLcontroller) # create the agent with the above controller
    #print out all the rule sets
    print("rules:")
    print(sharon.controller.rules)
    for i in range(500):
        sharon.run_one_epoch()
    print(' total num of successes:')
    print(sharon.success)
    # Print the path that our agent sharon took in her last epoch
    print(sharon.controller.path)
    sharon.print_path()
    sharon.print_reward_graph()
else:
    #FQL TEST
    print("main action space", action_space)
    FQLcontroller = TestController2(state, state_max, state_min, num_of_mf, action_space)
    carole = Agent(FQLcontroller)
    print(carole.controller.rules)
    for i in range(600):
        carole.run_one_epoch()

    print(' total num of successes:')
    print(carole.success)
    carole.print_path()
    print(carole.controller.path)
    carole.print_reward_graph()
    # Print the path that our agent sharon took in her last epoch
    #print(carole.controller.path)
    #carole.print_path()
    np.set_printoptions(threshold=np.inf)
    print('q table below')
    #print(carole.controller.q_table)
    #f =open("q_table.txt", "a")
    #f.write("RULES:")
    #f.write(str(carole.controller.rules))
    #f.write("Q TABLE:")
    #f.write(str(carole.controller.q_table))


