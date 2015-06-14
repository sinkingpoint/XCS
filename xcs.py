import numpy
import numpy.random
import itertools
from copy import deepcopy

"""
A class that represents the parameters of an XCS system
"""
class parameters:
    def __init__(self):
        self.state_length = 5     #The number of bits in the state
        self.num_actions = 2      #The number of actions in this system
        self.theta_mna = 2        #The minimum number of elements in the match set
        self.initial_prediction = 0.01        #The initial prediction value in classifiers
        self.initial_error = 0.01             #The initial error value in classifiers
        self.initial_fitness = 0.01           #The initial fitness value in classifiers
        self.p_hash = 0.3                     #The probability of generating a hash in a condition
        self.prob_exploration = 0.5           #The probability of the system exploring the environment
        self.gamma = 0.71                     #The payoff decay rate
        self.alpha = 0.1
        self.beta = 0.2
        self.nu = 5                           #
        self.N = 400                          #The maximum number of classifiers in the population
        self.e0 = 0.01                        #The minimum error value
        self.theta_del = 25                   #The experience level below which we don't delete classifiers
        self.delta = 0.1                      #The multiplier for the deletion vote of a classifier
        self.theta_sub = 20                   #The rate of subsumption
        self.theta_ga = 25                    #The rate of the genetic algorithm
        self.crossover_rate = 0.8
        self.mutation_rate = 0.04
        self.do_GA_subsumption = True
        self.do_action_set_subsumption = True

"""
A classifier in the X Classifier System
"""
class classifier:
    global_id = 0 #A Globally unique identifier
    def __init__(self, parameters, state = None):
        self.id = classifier.global_id
        classifier.global_id = classifier.global_id + 1
        self.action = numpy.random.randint(0, parameters.num_actions)
        self.prediction = parameters.initial_prediction
        self.error = parameters.initial_error
        self.fitness = parameters.initial_fitness
        self.experience = 0
        self.time_stamp = 0
        self.average_size = 1
        self.numerosity = 1
        if state == None:
            self.condition = ''.join(['#' if numpy.random.rand() < parameters.p_hash else '0' if numpy.random.rand() > 0.5 else '1' for i in [0] * parameters.state_length])
        else:
            #Generate the condition from the state (if supplied)
            self.condition = ''.join(['#' if numpy.random.rand() < parameters.p_hash else state[i] for i in range(parameters.state_length)])

    def __str__(self):
        return "Classifier " + str(self.id) + ": " + self.condition + " = " + str(self.action) + " Fitness: " + str(self.fitness) + " Prediction: " + str(self.prediction) + " Error: " + str(self.error) + " Experience: " + str(self.experience)

    """
       Mutates this classifier, changing the condition and action
       @param state - The state of the system to mutate around
       @param mutation_rate - The probability with which to mutate
       @param num_actions - The number of actions in the system
    """
    def _mutate(self, state, mutation_rate, num_actions):
        self.condition = ''.join([self.condition[i] if numpy.random.rand() > mutation_rate else state[i] if self.condition[i] == '#' else '#' for i in range(len(self.condition))])
        if numpy.random.rand() < mutation_rate:
            self.action = numpy.random.randint(0, num_actions)

    """
       Calculates the deletion vote for this classifier, that is, how much it thinks it should be deleted
       @param average_fitness - The average fitness in the current action set
       @param theta_del - See parameters above
       @param delta - See parameters above
    """
    def _delete_vote(self, average_fitness, theta_del, delta):
        vote = self.average_size * self.numerosity
        if self.experience > theta_del and self.fitness / self.numerosity < delta * average_fitness:
            return vote * average_fitness / (self.fitness / self.numerosity)
        else:
            return vote

    """
        Returns whether this classifier can subsume others
        @param theta_sub - See parameters above
        @param e0 - See parameters above
    """
    def _could_subsume(self, theta_sub, e0):
        return self.experience > theta_sub and self.error < e0

    """
        Returns whether this classifier is more general than another
        @param other - the classifier to check against
    """
    def _is_more_general(self, other):
        if len([i for i in self.condition if i == '#']) <= len([i for i in other.condition if i == '#']):
            return False

        return all([s == '#' or s == o for s, o in zip(self.condition, other.condition)])

    """
        Returns whether this classifier subsumes another
        @param other - the classifier to check against
        @param theta_sub - See parameters above
        @param e0 - See parameters above
    """
    def _does_subsume(self, other, theta_sub, e0):
        return self.action == other.action and self._could_subsume(theta_sub, e0) and self._is_more_general(other)

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if other == None:
            return False
        return self.id == other.id

"""
   The main XCS class
"""
class xcs:
    """
        Initializes an instance of the X classifier system
        @param parameters - A parameters instance (See above), containing the parameters for this system
        @param state_function - A function which returns the current state of the system, as a string
        @param reward_function - A function which takes a state and an action, performs the action and returns the reward
        @param eop_function - A function which returns whether the state is at the end of the problem
    """
    def __init__(self, parameters, state_function, reward_function, eop_function):
        self.pre_rho = 0
        self.parameters = parameters
        self.state_function = state_function
        self.reward_function = reward_function
        self.eop_function = eop_function
        self.population = []
        self.time_stamp = 0

        self.previous_action_set = None
        self.previous_reward = None
        self.previous_state = None

    """
        Prints the current population to stdout
    """
    def print_population(self):
        for i in self.population:
            print(i)

    """
       Classifies the given state, returning the class
       @param state - the state to classify
    """
    def classify(self, state):
        match_set = self._generate_match_set(state)
        predictions = self._generate_predictions(match_set)
        action = numpy.argmax(predictions)
        return action

    """
        Runs a single iteration of the learning algorithm for this XCS instance
    """
    def run_experiment(self):
        curr_state = self.state_function()
        match_set = self._generate_match_set(curr_state)
        predictions = self._generate_predictions(match_set)
        action = self._select_action(predictions)
        action_set = [clas for clas in match_set if clas.action == action]
        reward = self.reward_function(curr_state, action)

        #Update the previous set
        if self.previous_action_set:
            P = self.previous_reward + self.parameters.gamma * max(predictions)
            self._update_set(self.previous_action_set, P)
            self._run_ga(self.previous_action_set, self.previous_state)

        if self.eop_function():
            self._update_set(action_set, reward)
            self._run_ga(action_set, curr_state)
            self.previous_action_set = None
        else:
            self.previous_action_set = action_set
            self.previous_reward = reward
            self.previous_state = curr_state
        self.time_stamp = self.time_stamp + 1

    """
        Generates the match set for the given state, covering as necessary
        @param state - the state to generate a match set for
    """
    def _generate_match_set(self, state):
        set_m = []
        while len(set_m) == 0:
            set_m = [clas for clas in self.population if state_matches(clas.condition, state)]
            if len(set_m) < self.parameters.theta_mna:#Cover
                clas = self._generate_covering_classifier(state, set_m)
                self._insert_to_population(clas)
                self._delete_from_population()
                set_m = []
        return set_m

    """
        Deletes a classifier from the population, if necessary
    """
    def _delete_from_population(self):
        numerosity_sum = sum([clas.numerosity for clas in self.population])
        if numerosity_sum <= self.parameters.N:
            return

        average_fitness = sum([clas.fitness for clas in self.population]) / numerosity_sum
        votes = [clas._delete_vote(average_fitness, self.parameters.theta_del, self.parameters.delta) for clas in self.population]
        vote_sum = sum(votes)
        choice = numpy.random.choice(self.population, p=[vote / vote_sum for vote in votes])
        if choice.numerosity > 1:
            choice.numerosity = choice.numerosity - 1
        else:
            self.population.remove(choice)

    """
        Inserts the given classifier into the population, if it isn't able to be
        subsumed by some other classifier in the population
        @param clas - the classifier to insert
    """
    def _insert_to_population(self, clas):
        same = [c for c in self.population if (c.action, c.condition) == (clas.action, clas.condition)]
        if same:
            same[0].numerosity = same[0].numerosity + 1
            return
        self.population.append(clas)

    """
        Generates a classifier that conforms to the given state, and has an unused action from
        the given match set
        @param state - The state to make the classifier conform to
        @param match_set - The set of current matches
    """
    def _generate_covering_classifier(self, state, match_set):
        clas = classifier(self.parameters, state)
        used_actions = [classifier.action for classifier in match_set]
        available_actions = list(set(range(self.parameters.num_actions)) - set(used_actions))
        clas.action = numpy.random.choice(available_actions)
        clas.time_stamp = self.time_stamp
        return clas

    """
        Generates a prediction array for the given match set
        @param match_set - The match set to generate predictions for
    """
    def _generate_predictions(self, match_set):
        PA = [0] * self.parameters.num_actions
        FSA = [0] * self.parameters.num_actions
        for clas in match_set:
            PA[clas.action] += clas.prediction * clas.fitness
            FSA[clas.action] += clas.fitness

        normal = [PA[i] if FSA[i] == 0 else PA[i]/FSA[i] for i in range(self.parameters.num_actions)]

        return normal

    """
        Selects the action to run from the given prediction array. Takes into account exploration
        vs exploitation
        @param predictions - The prediction array to generate an action from
    """
    def _select_action(self, predictions):
        valid_actions = [action for action in range(self.parameters.num_actions) if predictions[action] != 0]
        if len(valid_actions) == 0:
            return numpy.random.randint(0, self.parameters.num_actions)

        if numpy.random.rand() < self.parameters.prob_exploration:
            return numpy.random.choice(valid_actions)
        else:
            return numpy.argmax(predictions)

    """
       Updates the given action set's prediction, error, average size and fitness using the given decayed performance
       @param action_set - The set to update
       @param P - The reward to use
    """
    def _update_set(self, action_set, P):
        set_numerosity = sum([clas.numerosity for clas in action_set])
        for clas in action_set:
            clas.experience = clas.experience + 1
            if clas.experience < 1. / self.parameters.beta:
                clas.prediction = clas.prediction + (P - clas.prediction) / clas.experience
                clas.error = clas.error + (abs(P - clas.prediction) - clas.error) / clas.experience
                clas.average_size = clas.average_size + (set_numerosity - clas.numerosity) / clas.experience
            else:
                clas.prediction = clas.prediction + (P - clas.prediction) * self.parameters.beta
                clas.error = clas.error + (abs(P - clas.prediction) - clas.error) * self.parameters.beta
                clas.average_size = clas.average_size + (set_numerosity - clas.numerosity) * self.parameters.beta

        #Update fitness
        kappa = {clas: 1 if clas.error < self.parameters.e0 else self.parameters.alpha * (clas.error / self.parameters.e0) ** -self.parameters.nu for clas in action_set}
        accuracy_sum = sum([kappa[clas] * clas.numerosity for clas in action_set])

        for clas in action_set:
            clas.fitness = clas.fitness + self.parameters.beta * (kappa[clas] * clas.numerosity / accuracy_sum - clas.fitness)
        if self.parameters.do_action_set_subsumption:
            self._action_set_subsumption(action_set);

    """
        Does subsumption inside the action set, finding the most general classifier
        and merging things into it
        @param action_set - the set to perform subsumption on
    """
    def _action_set_subsumption(self, action_set):
        cl = None
        for clas in action_set:
            if clas._could_subsume(self.parameters.theta_sub, self.parameters.e0):
                if cl == None or len([i for i in clas.condition if i == '#']) > len([i for i in cl.condition if i == '#']) or numpy.random.rand() > 0.5:
                    cl = clas

        if cl:
            for clas in action_set:
                if cl._is_more_general(clas):
                    cl.numerosity = cl.numerosity + clas.numerosity
                    action_set.remove(clas)
                    self.population.remove(clas)

    """
        Runs the genetic algorithm on the given set, generating two new classifers
        to be inserted into the population
        @param action_set - the action set to choose parents from
        @param state - The state mutate with
    """
    def _run_ga(self, action_set, state):
        if len(action_set) == 0:
            return

        if self.time_stamp - numpy.average([clas.time_stamp for clas in action_set], weights=[clas.numerosity for clas in action_set]) > self.parameters.theta_ga:
            for clas in action_set:
                clas.time_stamp = self.time_stamp

            fitness_sum = sum([clas.fitness for clas in action_set])

            probs = [clas.fitness / fitness_sum for clas in action_set]
            parent_1 = numpy.random.choice(action_set, p=probs)
            parent_2 = numpy.random.choice(action_set, p=probs)
            child_1 = deepcopy(parent_1)
            child_2 = deepcopy(parent_2)
            child_1.id = classifier.global_id
            child_2.id = classifier.global_id + 1
            classifier.global_id = classifier.global_id + 2
            child_1.numerosity = 1
            child_2.numerosity = 1
            child_1.experience = 0
            child_2.experience = 0

            if numpy.random.rand() < self.parameters.crossover_rate:
                _crossover(child_1, child_2)
                child_1.prediction = child_2.prediction = numpy.average([parent_1.prediction, parent_2.prediction])
                child_1.error = child_2.error = numpy.average([parent_1.error, parent_2.error])
                child_1.fitness = child_2.fitness = numpy.average([parent_1.fitness, parent_2.fitness])

            child_1.fitness = child_1.fitness * 0.1
            child_2.fitness = child_2.fitness * 0.1

            for child in [child_1, child_2]:
                child._mutate(state, self.parameters.mutation_rate, self.parameters.num_actions)
                if self.parameters.do_GA_subsumption == True:
                    if parent_1._does_subsume(child, self.parameters.theta_sub, self.parameters.e0):
                        parent_1.numerosity = parent_1.numerosity + 1
                    elif parent_2._does_subsume(child, self.parameters.theta_sub, self.parameters.e0):
                        parent_2.numerosity = parent_2.numerosity + 1
                    else:
                        self._insert_to_population(child)
                else:
                    self._insert_to_population(child)

                self._delete_from_population()

"""
    Returns whether the given state matches the given condition
    @param condition - The condition to match against
    @param state - The state to match against
"""
def state_matches(condition, state):
    return all([c == '#' or c == s for c, s in zip(condition, state)])

"""
    Cross's over the given children, modifying their conditions
    @param child_1 - The first child to crossover
    @param child_2 - The second child to crossover
"""
def _crossover(child_1, child_2):
    x = numpy.random.randint(0, len(child_1.condition))
    y = numpy.random.randint(0, len(child_1.condition))

    child_1_condition = list(child_1.condition)
    child_2_condition = list(child_2.condition)

    for i in range(x, y):
        child_1_condition[i], child_2_condition[i] = child_2_condition[i], child_1_condition[i]

    child_1.condition = ''.join(child_1_condition)
    child_2.condition = ''.join(child_2_condition)
