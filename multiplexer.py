import numpy.random
import nxcs

bits = 2

rho = 1000
learning_steps = 5000
validation_steps = 100
trials = 30

def state():
    return ''.join(['0' if numpy.random.rand() > 0.5 else '1' for i in range(0, bits + 2**bits)])

def eop():
    return True

def reward(state, action):
    address = state[:bits]
    data = state[bits:]

    if str(action) == data[int(address, 2)]:
        return 1000
    else:
        return 0

parameters = nxcs.parameters()
parameters.state_length = bits + 2**bits
parameters.theta_mna = 2
parameters.e0 = 1000 * 0.01
parameters.theta_ga = 25
parameters.prob_exploration = 1

correct = 0

for i in range(trials):
    my_xcs = nxcs.xcs(parameters, state, reward, eop)

    for j in range(learning_steps):
        my_xcs.run_experiment()

    this_correct = 0
    for j in range(validation_steps):
        rand_state = state()
        this_correct = this_correct + reward(rand_state, my_xcs.classify(rand_state))

    correct = correct + this_correct
    print("Trial " + str(i+1) + ": " + str((this_correct / validation_steps / rho) * 100) + "%");
    #my_xcs.print_population();

print("Overall: " + str((correct / validation_steps / rho / trials) * 100) + "%");
