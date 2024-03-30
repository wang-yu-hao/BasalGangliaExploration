import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import simulations as sim
import pickle
import math
import scipy.optimize as opt
import matplotlib.pylab as pl

params_bernoulli1 = np.array([.45, .45, .45, .45, .45, .45, .45, .45, .45, .55])
params_bernoulli3 = np.array([.8, .8, .8, .8, .8, .8, .8, .8, .8, .9])
params_bernoulli2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2, .3])


env_params = params_bernoulli2
print(env_params)


mean_regret = lambda params: np.mean(sim.run("Gaussian", 
                                                env_params,
                                                  'opal-star', 
                                                  [params[0], params[1], params[2]], 
                                                  200, 
                                                  5000)['regrets'])

min_regret = np.inf
best_params = []

for alpha_c in (0.025, 0.05, 0.1):
    
    for alpha_gn in np.arange(0.1, 1.01, 0.05):
        
        for beta in np.arange(1, 10.1, 0.5):
            
            regret = mean_regret([alpha_c, alpha_gn, beta])

            if regret < min_regret:
                
                min_regret = regret
                best_params = [alpha_c, alpha_gn, beta]

                print('NEW MINIMUM:')
                print(min_regret)
                print(best_params)
