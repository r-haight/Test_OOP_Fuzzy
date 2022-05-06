# Libraries and Classes
import numpy as np
import FACL
from Agent import Agent
from TestController import TestController
from TestController2 import TestController2
import time
#
# This driver program is used for training an agent and then playing a series of games
# First the fuzzy inference (FIS) system is setup (number of membership functions etc)
# Then the type of training is selected: either fuzzy Q learning or fuzzy actor critic
# The controller is made and then plugged into an agent object. It is trained for a # of epochs
# Infomation like the time it took to train and the reward plot are shown after. Finally a new
# agent and controller object a made using the saved data and then they play a game 100 times, recording
# the outcome of each game (success of fail)

# This driver program simply gets an agent to learn to go to a location

# General Fuzzy Parameters
state = [5, 5] # start position on the grid
state_max = [50, 50] # max values of the grid [x,y]
state_min = [-50, -50] # smallest value of the grid [x,y]
num_of_mf = [9, 9] # breaking up the state space (grid in this case) into 29 membership functions
action_space = [0.785375, 2.356125, -1.57075, -0.785375, 0,  1.57075, -2.356125]

# Train and Run Games for either an agent trained by FACL or FQL
# FACL -> selection = 0
# FQL -> selection = any other number
selection = 0

########## TRAINING SECTION ###############

if(selection == 0) :
    start = time.time() # used to see how long the training time took
    FACLcontroller = TestController(state, state_max, state_min, num_of_mf) #create the FACL controller
    sharon = Agent(FACLcontroller) # create the agent with the above controller
    #print out all the rule sets
    print("rules:")
    print(sharon.controller.rules)
    for i in range(1500):
        sharon.run_one_epoch()
    end = time.time()
    print('total train time : ', end-start)
    print(' total num of successes during training : ', sharon.success)

    # Print the path that our agent sharon took in her last epoch
    print(sharon.controller.path) #numerical values of path
    sharon.print_path() #graph
    sharon.print_reward_graph()
    sharon.save_epoch_training_info() #save all the important info from our training sesh
else:
    start = time.time() #used to figure out how much time training took
    # FQL TEST
    print("action space: ", action_space)
    FQLcontroller = TestController2(state, state_max, state_min, num_of_mf, action_space)
    carole = Agent(FQLcontroller)
    print(carole.controller.rules)
    for i in range(1500):
        carole.run_one_epoch()
        #carole.print_path()
    end = time.time()

    #Show training Info
    print('total train time : ', end-start)
    print(' total num of successes during training : ', carole.success)
    carole.print_path() #Graph of path taken in last epoch
    print(carole.controller.path) #Numerical values of path taken in last epoch
    carole.print_reward_graph() #Rewards obtained in each epoch
    np.set_printoptions(threshold=np.inf)
    #print('q table below')
    #print(carole.controller.q_table)
    #f =open("q_table.txt", "a")
    #f.write("RULES:")
    #f.write(str(carole.controller.rules))
    #f.write("Q TABLE:")
    #f.write(str(carole.controller.q_table))

######## RUN GAMES ############
if(selection ==0):
    # RUN A SET OF GAMES FOR FACL AGENT
    #Step 1: load the fuzzy info to recreate the membership functions and rules
    # Note: Turn into a function to do this dynamically
    fuzzy_info = np.loadtxt("fuzzy_info.txt")
    max_state_vals = fuzzy_info[0:2]
    min_state_vals = fuzzy_info[2:4]
    number_of_membership_functions = [0,0]
    number_of_membership_functions[0] = int(fuzzy_info[4])
    number_of_membership_functions[1] = int(fuzzy_info[5])
    print('max ',max_state_vals) # Print out for troubleshooting purposes
    print('min ',min_state_vals)
    print('num mf',number_of_membership_functions)
    #Step 2: create the controller using the fuzzy info
    run_controller = TestController(state, max_state_vals, min_state_vals, number_of_membership_functions)
    #Step 3: create agent
    eli = Agent(run_controller)
    #Step 4: call the load function for actor critic weights
    eli.controller.load()
    #Step 5: run the game 100 times
    for i in range(100):
        eli.run_one_game()
    print("num of sucessful games : ",eli.success)
    eli.print_path()
else:
    # RUN A SET OF GAMES FOR FQL AGENT

    #Step 1: Load the fuzzy info from previous training experience
    # Note: Turn into a function to do this dynamically
    fuzzy_info = np.loadtxt("fuzzy_info.txt")
    max_state_vals = fuzzy_info[0:2]
    min_state_vals = fuzzy_info[2:4]
    number_of_membership_functions = [0, 0]
    number_of_membership_functions[0] = int(fuzzy_info[4])
    number_of_membership_functions[1] = int(fuzzy_info[5])

    # Step 2: create the controller using the fuzzy info
    run_controller = TestController(state, max_state_vals, min_state_vals, number_of_membership_functions)
    # Step 3: create agent
    victoria = Agent(run_controller)
    # Step 4: call the load function to get the q table
    victoria.controller.load()
    # Step 5: run the game 100 times
    for i in range(100):
        victoria.run_one_game()
    print("num of sucessful games : ", victoria.success)