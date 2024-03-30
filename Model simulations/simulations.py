import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
import math


def init(model, num_arms, sim_params, env_params, distribution, random_mean=False):

    match model:

        case 'ucb1':

            cum_reward = np.zeros(num_arms)
            counters = np.zeros(num_arms)
            
            cum_reward_table = []
            counters_table = []


            latents = {'reward': cum_reward_table,
                       'counter': counters_table
                       }
            
            if random_mean == False:

                return {'model': model, 'arms': num_arms, 'params': sim_params, 'var': [cum_reward, counters], 'latents': latents} 
            
            else:

                return {'model': model, 'arms': num_arms, 'params': sim_params, 'var': [cum_reward, counters], 'latents': latents, 'mean_rewards': np.random.normal(env_params[0][0], env_params[0][1])} 
        
        case 'ucb-tuned':

            cum_reward = np.zeros(num_arms)
            counters = np.zeros(num_arms)
            cum_squared_reward = np.zeros(num_arms)
            
            cum_reward_table = []
            cum_squared_reward_table = []
            counters_table = []


            latents = {'reward': cum_reward_table,
                       'squared_reward': cum_squared_reward_table,
                       'counter': counters_table
                       }
            
            if random_mean == False:

                return {'model': model, 'arms': num_arms, 'params': sim_params, 'var': [cum_reward, counters, cum_squared_reward], 'latents': latents}
            
            else:

                return {'model': model, 'arms': num_arms, 'params': sim_params, 'var': [cum_reward, counters, cum_squared_reward], 'latents': latents, 'mean_rewards': np.random.normal(env_params[0][0], env_params[0][1])}

        case 'ucb2':

            cum_reward = np.zeros(num_arms)
            counters = np.zeros(num_arms)
            r = np.zeros(num_arms)
            carry_count_down = 0
            carry_choice = None
            
            cum_reward_table = []
            counters_table = []
            r_table = []

            latents = {'reward': cum_reward_table,
                       'counter': counters_table,
                       'r': r_table
                       }
            
            if random_mean == False:

                return {'model': model, 'arms': num_arms, 'params': sim_params, 'var': [cum_reward, counters, r, carry_count_down, carry_choice], 'latents': latents}
            
            else: 

                return {'model': model, 'arms': num_arms, 'params': sim_params, 'var': [cum_reward, counters, r, carry_count_down, carry_choice], 'latents': latents, 'mean_rewards': np.random.normal(env_params[0][0], env_params[0][1])}


        case 'ucb-normal':
        
            cum_reward = np.zeros(num_arms)
            cum_squared_reward = np.zeros(num_arms)
            counters = np.zeros(num_arms)
            
            cum_reward_table = []
            cum_squared_reward_table = []
            counters_table = []


            latents = {'reward': cum_reward_table,
                       'squared_reward': cum_squared_reward_table,
                       'counter': counters_table
                       }
            
            if random_mean == False:

                return {'model': model, 'arms': num_arms, 'params': sim_params, 'var': [cum_reward, counters, cum_squared_reward], 'latents': latents}
            
            else:

                return {'model': model, 'arms': num_arms, 'params': sim_params, 'var': [cum_reward, counters, cum_squared_reward], 'latents': latents, 'mean_rewards': np.random.normal(env_params[0][0], env_params[0][1])}

        case 'neural':

            dynamic = sim_params[0]
            alpha_q = sim_params[1]
            alpha_s = sim_params[2]
            llambda = sim_params[3]
            power = sim_params[4]
            e = sim_params[5]

            if random_mean == False:
                if distribution == 'Bernoulli':
                    sim_params.append(np.mean(env_params))
                    sim_params.append(np.std(env_params))
                else:
                    sim_params.append(np.mean(env_params[0, :]))
                    sim_params.append(np.std(env_params[0, :]))
            else: 
                sim_params.append(env_params[0][0][0])
                sim_params.append(env_params[0][1][0])

            # Q_0 = sim_params[6]
            # S_0 = sim_params[7]
            Q_0 = 0.5
            S_0 = 0.5
            # Q_0 = 0
            # S_0 = np.sqrt(10)

            Q = Q_0 * np.ones(num_arms) 
            S = S_0 * np.ones(num_arms)
            counters = np.zeros(num_arms)

            Q_table = []
            S_table = []
            counters_table = []

            latents = {'Q': Q_table,
                       'S': S_table,
                       'counter': counters_table
                       }            

            if random_mean == False:

                return {'model': model, 'arms': num_arms, 'params': sim_params, 'var': [Q, S, counters], 'latents': latents}
            
            else:

                return {'model': model, 'arms': num_arms, 'params': sim_params, 'var': [Q, S, counters], 'latents': latents, 'mean_rewards': np.random.normal(env_params[0][0], env_params[0][1])}
        
        case 'kalman':
            
            if random_mean == False:
                if distribution == 'Bernoulli':
                    sim_params.append(np.mean(env_params))
                    sim_params.append(np.var(env_params))
                    sim_params.append(env_params * (1-env_params))
                else:
                    sim_params.append(np.mean(env_params[0, :]))
                    sim_params.append(np.var(env_params[0, :]))
                    sim_params.append(env_params[1, :] **2)
            else:
                sim_params.append(env_params[0][0][0])
                sim_params.append(env_params[0][1][0] **2)
                sim_params.append(np.array(env_params[1]) **2)

            # Q_0 = sim_params[2]
            # var_0 = sim_params[3]

            Q_0 = 0.5
            var_0 = 0.25

            # Q_0 = 0
            # var_0 = 10

            Q = Q_0 * np.ones(num_arms) 
            var = var_0 * np.ones(num_arms) 

            Q_table = []
            var_table = []

            latents = {'Q': Q_table,
                       'var': var_table
                       }            

            if random_mean == False:

                return {'model': model, 'arms': num_arms, 'params': sim_params, 'var': [Q, var], 'latents': latents}
            
            else:

                return {'model': model, 'arms': num_arms, 'params': sim_params, 'var': [Q, var], 'latents': latents, 'mean_rewards': np.random.normal(env_params[0][0], env_params[0][1])}
        
        case 'opal-star':

            V_0 = 0.5
            G_0 = 0.5
            N_0 = 0.5

            # V_0 = 0
            # G_0 = 0
            # N_0 = 0

            eta_0 = 1
            gamma_0 = 1

            G = G_0 * np.ones(num_arms) 
            N = N_0 * np.ones(num_arms)
            V = V_0 * np.ones(num_arms)

            beta_dist = [eta_0, gamma_0]

            G_table = []
            N_table = []
            V_table = []
            beta_dist_table = []

            latents = {'G': G_table,
                       'N': N_table,
                       'V': V_table,
                       'beta_dist': beta_dist_table
                       }            

            if random_mean == False:

                return {'model': model, 'arms': num_arms, 'params': sim_params, 'var': [V, G, N, beta_dist], 'latents': latents}
            
            else:

                return {'model': model, 'arms': num_arms, 'params': sim_params, 'var': [V, G, N, beta_dist], 'latents': latents, 'mean_rewards': np.random.normal(env_params[0][0], env_params[0][1])}
        


def act(sim, distribution, task_params, random_mean=False):

    '''
    Includes both choice selection and learning from result
    '''

    match sim['model']:

        case 'ucb1':

            if np.min(sim['var'][1]) < 1:
                
                choice = np.where(sim['var'][1] < 1)[0][0]

            else:

                mean_rewards = sim['var'][0] / sim['var'][1]
                D = mean_rewards + np.sqrt(2 * np.log(sum(sim['var'][1] + 1)) * np.ones(sim['arms']) / sim['var'][1])
                choice = np.argmax(D)

            if random_mean == False:

                if distribution == 'Bernoulli':

                    outcome = np.random.binomial(1, task_params[choice])

                elif distribution == 'Gaussian':

                    outcome = np.random.normal(task_params[0, choice], task_params[1, choice])

            else:

                    outcome = np.random.normal(sim['mean_rewards'][choice], task_params[1][choice])


            sim['var'][0][choice] += outcome
            sim['var'][1][choice] += 1
            
            sim['latents']['reward'].append(list[sim['var'][0]])
            sim['latents']['counter'].append(list(sim['var'][1]))

        case 'ucb-tuned':

            if np.min(sim['var'][1]) < 1:
                
                choice = np.where(sim['var'][1] < 1)[0][0]

            else:

                mean_rewards = sim['var'][0] / sim['var'][1]
                mean_squared_rewards = sim['var'][2] / sim['var'][1]
                V = mean_squared_rewards - mean_rewards **2 + np.sqrt(2* np.log(sum(sim['var'][1]))/sim['var'][1])
                D = mean_rewards + np.sqrt(np.log(sum(sim['var'][1])+1)/sim['var'][1] * np.minimum(0.25, V))
                choice = np.argmax(D)


            if random_mean == False:

                if distribution == 'Bernoulli':

                    outcome = np.random.binomial(1, task_params[choice])

                elif distribution == 'Gaussian':

                    outcome = np.random.normal(task_params[0, choice], task_params[1, choice])

            else:

                    outcome = np.random.normal(sim['mean_rewards'][choice], task_params[1][choice])

            sim['var'][0][choice] += outcome
            sim['var'][1][choice] += 1
            sim['var'][2][choice] += outcome **2
            
            sim['latents']['reward'].append(list[sim['var'][0]])
            sim['latents']['squared_reward'].append(list[sim['var'][2]])
            sim['latents']['counter'].append(list(sim['var'][1]))

        case 'ucb2':

            if np.min(sim['var'][1]) < 1:
                
                choice = np.where(sim['var'][1] < 1)[0][0]

            elif sim['var'][3] > 0:

                choice = sim['var'][4]
                sim['var'][3] -= 1

            else:

                mean_rewards = sim['var'][0] / sim['var'][1]
                D = mean_rewards + np.sqrt((1+sim['params']) * np.log(math.e * (sum(sim['var'][1] + 1)) / np.ceil((1 + sim['params']) ** sim['var'][2])) / (2* np.ceil((1 + sim['params']) ** sim['var'][2])))
                choice = np.argmax(D)
                sim['var'][4] = choice
                sim['var'][3] = (np.ceil((1 + sim['params']) ** (sim['var'][2]+1))  -  np.ceil((1 + sim['params']) ** sim['var'][2]))[choice] -1
                sim['var'][2][choice] += 1


            if random_mean == False:

                if distribution == 'Bernoulli':

                    outcome = np.random.binomial(1, task_params[choice])

                elif distribution == 'Gaussian':

                    outcome = np.random.normal(task_params[0, choice], task_params[1, choice])

            else:

                    outcome = np.random.normal(sim['mean_rewards'][choice], task_params[1][choice])

            sim['var'][0][choice] += outcome
            sim['var'][1][choice] += 1
            
            sim['latents']['reward'].append(list[sim['var'][0]])
            sim['latents']['counter'].append(list(sim['var'][1]))
            sim['latents']['r'].append(list(sim['var'][2]))

        case 'ucb-normal':

            if np.min(sim['var'][1]) <= 8 * np.log10(sum(sim['var'][1])+1):
                
                choice = np.where(sim['var'][1] <= 8 * np.log10(sum(sim['var'][1])+1))[0][0]

            else:



                mean_rewards = sim['var'][0] / sim['var'][1]

                D = mean_rewards + np.sqrt(16 * (sim['var'][2] - sim['var'][1] * mean_rewards **2) / (sim['var'][1]-1) * np.log(sum(sim['var'][1])) / (sim['var'][1]))
                choice = np.argmax(D)


            if random_mean == False:

                if distribution == 'Bernoulli':

                    outcome = np.random.binomial(1, task_params[choice])

                elif distribution == 'Gaussian':

                    outcome = np.random.normal(task_params[0, choice], task_params[1, choice])

            else:

                    outcome = np.random.normal(sim['mean_rewards'][choice], task_params[1][choice])

            sim['var'][0][choice] += outcome
            sim['var'][1][choice] += 1
            sim['var'][2][choice] += outcome **2
            
            sim['latents']['reward'].append(list[sim['var'][0]])
            sim['latents']['counter'].append(list(sim['var'][1]))
            sim['latents']['squared_reward'].append(list[sim['var'][2]])

        case 'neural':

            a = 0
            b = 0
            m = 0
            k = 1

            dynamic = sim['params'][0]
            alpha_q = sim['params'][1]
            alpha_s = sim['params'][2]
            llambda = sim['params'][3]
            power = sim['params'][4]
            e = sim['params'][5]


            if np.min(sim['var'][2]) < 1:
                
                choice = np.where(sim['var'][2] < 1)[0][0]

            else:

                dopamine = llambda * (m + k * (sim['var'][2] ** power) + np.random.randn(sim['arms']) * (a + b* (m + k* (sim['var'][2] ** power))))
                E = e * np.random.randn(sim['arms'])
                D = sim['var'][0] + dopamine * sim['var'][1] + E
                choice = np.argmax(D)


            if random_mean == False:

                if distribution == 'Bernoulli':

                    outcome = np.random.binomial(1, task_params[choice])

                elif distribution == 'Gaussian':

                    outcome = np.random.normal(task_params[0, choice], task_params[1, choice])

            else:

                    outcome = np.random.normal(sim['mean_rewards'][choice], task_params[1][choice])

            sim['var'][2][choice] += 1

            if dynamic:

                sim['var'][0][choice] += alpha_q * (m+k * sim['var'][2][choice] ** power)/(m+k) * (outcome - sim['var'][0][choice])
                sim['var'][1][choice] += alpha_s * (m+k * (sim['var'][2][choice]) ** power)/(m+k) * (np.abs(outcome - sim['var'][0][choice]) - sim['var'][1][choice])
                
            else: 


                sim['var'][0][choice] += alpha_q * (outcome - sim['var'][0][choice])
                sim['var'][1][choice] += alpha_s * (np.abs(outcome - sim['var'][0][choice]) - sim['var'][1][choice])

            sim['latents']['Q'].append(list(sim['var'][0]))
            sim['latents']['S'].append(list(sim['var'][1]))
            sim['latents']['counter'].append(list(sim['var'][2]))

        case 'kalman':

            tau_squared = sim['params'][4]
            Q_0 = sim['params'][2]
            var_0 = sim['params'][3]
            llambda = sim['params'][0]
            e = sim['params'][1]

            E = e * np.random.randn(sim['arms'])
            D = sim['var'][0] + llambda * sim['var'][1] ** 0.5 + E
            choice = np.argmax(D)

            if random_mean == False:

                if distribution == 'Bernoulli':

                    outcome = np.random.binomial(1, task_params[choice])

                elif distribution == 'Gaussian':

                    outcome = np.random.normal(task_params[0, choice], task_params[1, choice])

            else:

                    outcome = np.random.normal(sim['mean_rewards'][choice], task_params[1][choice])

            alpha = sim['var'][1][choice]/ (sim['var'][1][choice] + tau_squared[choice])
            sim['var'][1][choice] -= alpha * sim['var'][1][choice]
            sim['var'][0][choice] += alpha * (outcome - sim['var'][0][choice])

            sim['latents']['Q'].append(list(sim['var'][0]))
            sim['latents']['var'].append(list(sim['var'][1]))

        case 'opal-star':

            T = 10
            phi = 1
            k = 20

            alpha_c = sim['params'][0]
            alpha_gn = sim['params'][1]
            beta = sim['params'][2]


            beta_param_1 = sim['var'][3][0]/sim['arms']
            beta_param_2 = sim['var'][3][1]/sim['arms']


            beta_mean, beta_var = stats.beta.stats(beta_param_1, beta_param_2, moments='mv')
            beta_std = beta_var ** 0.5

            if beta_mean - phi * beta_std > 0.5 or beta_mean + phi * beta_std < 0.5:
                S = 1
            else: 
                S = 0

            rho = S * (beta_mean-0.5) * k

            beta_g = beta * max(0, 1+rho)
            beta_n = beta * max(0, 1-rho)

            A = beta_g * sim['var'][1] - beta_n * sim['var'][2]


            choice = np.argmax(A)

            if random_mean == False:

                if distribution == 'Bernoulli':

                    outcome = np.random.binomial(1, task_params[choice])

                elif distribution == 'Gaussian':

                    outcome = np.random.normal(task_params[0, choice], task_params[1, choice])

            else:

                    outcome = np.random.normal(sim['mean_rewards'][choice], task_params[1][choice])

            sim['latents']['V'].append(list(sim['var'][0]))
            sim['latents']['G'].append(list(sim['var'][1]))
            sim['latents']['N'].append(list(sim['var'][2]))
            sim['latents']['beta_dist'].append(list(sim['var'][3]))


            alpha_gn_modulated = alpha_gn / (1 + 1/(T*beta_var))

            delta = outcome - sim['var'][0][choice]
            
            sim['var'][0][choice] += alpha_c * delta
            sim['var'][1][choice] += alpha_gn_modulated * sim['var'][1][choice] * delta
            sim['var'][2][choice] += alpha_gn_modulated * sim['var'][2][choice] * (-delta)

            if sim['var'][1][choice] > 10:

                sim['var'][1][choice] = 10

            if sim['var'][2][choice] > 10:

                sim['var'][2][choice] = 10



            # for bernoulli task and gaussian task with mean ~0.5
            sim['var'][3][0] += outcome
            sim['var'][3][1] += (1-outcome)
            
            # for gaussian task with mean ~0
            # sim['var'][3][0] += outcome
            # sim['var'][3][1] += -outcome


    return sim, choice, outcome


def run(distribution, env_parameters, model, sim_params, N=1000, T=5000, random_mean=False):

    if random_mean == False:
        num_arms = env_parameters.shape[-1]
    else:
        num_arms = len(env_parameters[0][0])

    all_prob_chosen = []
    all_regrets = []

    for n in range(N):

        prob_chosen = []
        
        sim = init(model, num_arms, sim_params, env_parameters, distribution, random_mean)

        for t in range(T):

            sim, choice, outcome = act(sim, distribution, env_parameters, random_mean)


            if random_mean == False:

                if distribution == 'Bernoulli':
                    prob_chosen.append(env_parameters[choice])
                elif distribution == 'Gaussian':
                    prob_chosen.append(env_parameters[0, choice])

            else:

                prob_chosen.append(sim['mean_rewards'][choice])

        all_prob_chosen.append(prob_chosen)

        if random_mean == False:

            if distribution == 'Bernoulli':
                max_prob = max(env_parameters)
            else:
                max_prob = max(env_parameters[0, :])

        else:

            max_prob = max(sim['mean_rewards'])

        regrets = max_prob - np.array(prob_chosen)
        all_regrets.append(list(regrets))

    all_regrets = np.array(all_regrets)

    return({'distribution':distribution, 
            'env_parameters': env_parameters, 
            'model': sim['model'], 
            'regrets': all_regrets})
