import numpy as np
from matplotlib import pyplot as plt


def exponential_decay_method(time, episodes, epsilon_min):
    A = 0.5
    B = 0.1
    C = 0.1
    std_time = (time - A*episodes)/(B*episodes)
    cosh = np.cosh(np.exp(-std_time))
    epsilon = 1 - (1/cosh+((time*C)/episodes))

    return max(epsilon, epsilon_min)

def decrement_epsilon(epsilon, min_epsilon, epsilon_decay):
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay
    else:
        epsilon = min_epsilon
    
    return epsilon

def old_method(exploration_rate, exploration_decay, time):
    return 0.001 + (exploration_rate-0.01)*np.exp(-exploration_decay*(time+1))

