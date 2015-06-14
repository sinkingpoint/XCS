import numpy.random
import nxcs

numpy.random.seed(0)

bits = 2

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
parameters.theta_ga = 50

my_xcs = nxcs.xcs(parameters, state, reward, eop)

for i in range(5000):
    my_xcs.run_experiment()

correct = 0
for i in [0] * 100:
    rand_state = state()
    correct = correct + reward(rand_state, my_xcs.classify(rand_state))

my_xcs.print_population()
print(correct / 100 / 1000);
