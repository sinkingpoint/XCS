import numpy.random
import xcs

"""
    An implementation of an N-bit multiplexer problem for the X classifier system
"""

#The number of bits to use for the address in the multiplexer
bits = 1

#The maximum reward
rho = 1000

#The number of steps we learn for
learning_steps = 5000

#The number of steps we validate for
validation_steps = 100

"""
    Returns a random state of the multiplexer
"""
def state():
    return ''.join(['0' if numpy.random.rand() > 0.5 else '1' for i in range(0, bits + 2**bits)])

"""
    The N-bit multiplexer is a single step problem, and thus always is at the end of the problem
"""
def eop():
    return True

"""
    Calculates the reward for performing the action in the given state
"""
def reward(state, action):
    #Extract the parts from the state
    address = state[:bits]
    data = state[bits:]

    #Check the action
    if str(action) == data[int(address, 2)]:
        return rho
    else:
        return 0

#Set some parameters
parameters = xcs.parameters()
parameters.state_length = bits + 2**bits
parameters.theta_mna = 2
parameters.e0 = 1000 * 0.01
parameters.theta_ga = 25
parameters.gamma = 0
parameters.N = 400

#Construct an XCS instance
my_xcs = xcs.xcs(parameters, state, reward, eop)

#Train
for j in range(learning_steps):
    my_xcs.run_experiment()

#Validate
this_correct = 0
for j in range(validation_steps):
    rand_state = state()
    this_correct = this_correct + reward(rand_state, my_xcs.classify(rand_state))

print("Performance " + str(i+1) + ": " + str((this_correct / validation_steps / rho) * 100) + "%");
