import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import simulations as sim
import pickle
import math
import scipy.optimize as opt


method = input('Method: ') # local minimisation method available in scipy.optimize
distribution = input('Distribution: ') # Bernoulli or Gaussian; use Gaussian for the Gershman task
task = input('Task: ') # bernoulli1, bernoulli2, bernoulli3, gaussian, gershman
dynamic = input('Dynamic: ') # True or False; dynamic learning rate
power = input('Power: ') # power parameter of novelty function
n = input('n: ')
constraint = input('Constraint: ') # True or False; constraint on the learning rate so that alpha_q > alpha_s
random_mean = input('Random mean: ') # True or False; True for the Gershman task only

dynamic = bool(dynamic)
random_mean = bool(random_mean)
power = float(power)
n = int(n)
constraint = bool(constraint)

if constraint == True:

    constraint = opt.LinearConstraint(np.array([[1,-1,0], [0,0,0], [0,0,0]]), lb=[0,0,0])

else:
    
    constraint = None

params_bernoulli1 = np.array([.45, .45, .45, .45, .45, .45, .45, .45, .45, .55])
params_bernoulli2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2, .3])
params_bernoulli3 = np.array([.8, .8, .8, .8, .8, .8, .8, .8, .8, .9])

params_gaussian = np.array([[.45, .45, .45, .45, .45, .45, .45, .45, .45, .55],  
                             [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]])
params_gershman = [[[0, 0], [np.sqrt(10), np.sqrt(10)]], 
                             [1, 1]]

mean_regret = lambda params: np.mean(sim.run(distribution, 
                                                  globals()['params_'+task],
                                                  'neural', 
                                                  [dynamic, params[0], params[1], params[2], power, 0], 
                                                  200, 
                                                  5000,
                                                  random_mean=random_mean)['regrets'])

optimal_params = opt.shgo(mean_regret, ((0, 1), (0, 1), (0, 10)), options={'disp': True}, minimizer_kwargs={'method': method}, n=n, constraints=constraint, iters=1)

print(optimal_params)

