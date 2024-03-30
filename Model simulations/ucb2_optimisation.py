import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import simulations as sim
import pickle
import math
import scipy.optimize as opt

distribution = input('Distribution: ') # Bernoulli or Gaussian; use Gaussian for the Gershman task
task = input('Task: ') # bernoulli1, bernoulli2, bernoulli3, gaussian, gershman
random_mean = input('Random mean: ') # True or False; True for the Gershman task only

random_mean = bool(random_mean)

params_bernoulli1 = np.array([.45, .45, .45, .45, .45, .45, .45, .45, .45, .55])
params_bernoulli2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2, .3])
params_bernoulli3 = np.array([.8, .8, .8, .8, .8, .8, .8, .8, .8, .9])

params_gaussian = np.array([[.45, .45, .45, .45, .45, .45, .45, .45, .45, .55],  
                             [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]])
params_gershman = [[[0, 0], [np.sqrt(10), np.sqrt(10)]], 
                             [1, 1]]

mean_regret = lambda params: np.mean(sim.run(distribution, 
                                                  globals()['params_'+task],
                                                  'ucb2', 
                                                  params, 
                                                  200, 
                                                  5000)['regrets'])

optimal_params = opt.minimize_scalar(mean_regret, bounds=(0,1))