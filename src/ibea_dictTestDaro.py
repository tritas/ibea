#!/usr/bin/env python2
# -*- coding : utf8 -*-

""" Indicator-based Evolutionary Algorithm with Epsilon indicator
is an evolutionary algorithm for searching a multi-objective space 
using the concept of the Pareto front.
Formally, the algorithm is classified as a (\mu/\rho + \lambda)-ES.
Indeed, one parent is selected after a binary tournament for each offspring.
Furthermore, the ES applies `+` selection, 
meaning that only fitness (and not age) is taken into account for the selection phase.
"""

from operator import mod
from pprint import pprint
import numpy as np

class IBEA(object):
    def __init__(self,
                 kappa=0.05, 
                 alpha=100, 
                 n_offspring=8, 
                 seed=1,
                 pr_x=1.0, # 0.5
                 pr_mut=0.01, # 0.8
                 var=0.01):
        # --- Algorithm parameters
        self.kappa = kappa             # Fitness scaling ratio
        self.alpha = alpha             # Population size
        self.pr_crossover = pr_x       # Crossover probability
        self.pr_mutation = pr_mut      # Mutation probability
        self.noise_variance = var      # Noise level for mutation
        self.n_offspring = n_offspring # Number of offspring individuals
        self._min = None               # Objective function minima
        self._max = None               # Objective function maxima
        self.indicator_max = None      # Indicator function maximum
        
        # --- Data structure containing: population vectors, fitness and objective values
        self.pop_data = dict() 

        # --- Random state
        np.random.seed(seed)

    def __str__(self):
        return 'Indicator-based Evolutionary Algorithm with Epsilon indicator'

    def ibea(self, fun, lbounds, ubounds, budget):
        lbounds, ubounds = np.array(lbounds), np.array(ubounds) # [-100, 100]
        dim, f_min = len(lbounds), None
        dim_sqrt = np.sqrt(dim+1)
        # TO Test: Pr(mutation) = 1/n
        self.pr_mutation = 1 / float(dim)
        
        # 1. Initial population of size alpha
        # Sampled from the uniform distribution [lbounds, ubounds]
        particles = np.random.rand(self.alpha, dim) \
                 * (ubounds - lbounds) + lbounds
        # Rescaling objective values to [0,1]
        objective_values = np.array([fun(x) for x in particles])
        objective_values = self.rescale(objective_values)
        # Datastructure containing all the population info
        self.pop_data = {
            p : {
                'x': particles[p],
                'obj': objective_values[p],
                'fitness': 0.0,
            } for p in range(self.alpha)
        }
        # Lazy compute max absolute value of pairwise epsilon indicator 
        self.indicator_max = max([abs(self.eps_indic_fun(i1, i2))
                                  for i1 in range(self.alpha)
                                  for i2 in range(self.alpha)
                                  if i1 != i2])
        generation = 0
        population_size = self.alpha
        done = False

        while not done:
            # 2. Fitness assignment
            for i in self.pop_data.keys():
                self.compute_fitness(i)
            # 3 Environmental selection
            env_selection_ok = False
            while not env_selection_ok:
                # 3.1 Environmental selection
                fit_min = 10
                worst_fit = None
                for (k, v) in self.pop_data.items():
                    if v['fitness'] <= fit_min:
                        fit_min = v['fitness'] 
                        worst_fit = k

                # 3.3 Update fitness values
                for i in self.pop_data.keys():
                    self.pop_data[i]['fitness'] += np.exp(
                        - self.eps_indic_fun(worst_fit, i) \
                        / (self.indicator_max * self.kappa))
                # 3.2 Remove individual
                population_size -= 1
                self.pop_data.pop(worst_fit)
                # Continue while P does not exceed alpha
                # TOFIX: does not work with population_size variable
                env_selection_ok = len(self.pop_data) <= self.alpha

            # 4. Check convergence condition
            done = generation >= budget
            if done: break
            # 5. Mating selection
            # Perform binary tournament selection with replacement on P in order
            # to fill the temporary mating pool P'.
            item_keys = list(self.pop_data.keys())
            pool = []
            
            for i in range(2*self.n_offspring):
                p1, p2 = np.random.choice(item_keys, 2)
                if self.pop_data[p1]['fitness'] >= self.pop_data[p2]['fitness']:
                    pool.append(p1)
                else:
                    pool.append(p2)
            
            # 6. Variation
            for i in range(self.n_offspring):
               #x1 = self.pop_data[pool[i]]['x']
                #x2 = self.pop_data[pool[i+1]]['x']
                x1,x2 = cxSimulatedBinary(pool,pool,5)
                offspring = np.empty(dim, dtype=np.float64)

                if np.random.binomial(1, self.pr_crossover):
                    '''Apply recombination operators'''
                    # Discrete (dominant) recombination
                    #   -> Good for separable functions
                    #pr_genes = np.random.binomial(1, 0.5, dim)
                    #for d in range(dim):
                    #    offspring[d] = x1[d] if pr_genes[d] else x2[d]

                    # One-point crossover
                    x_ind = np.random.randint(dim) 
                    offspring[:x_ind] = x1[:x_ind]
                    offspring[x_ind:] = x2[x_ind:]
                else:
                    # Intermediate recombination (noted \rho_I)
                    offspring = np.divide(x1 + x2, 2)

                    # TODO: Weighted recombination (noted \rho_W)

                if np.random.binomial(1, self.pr_mutation):
                    # Apply isotropic mutation operator
                    offspring += np.random.randn(dim) * self.noise_variance

                ''' Adapt step-size - 1/5-th rule for (1+1)-ES: 
                self.compute_fitness(offspring)
                indicator = int(f_parent <= f_offspring)
                sigma *= np.power(np.exp(indicator - 0.2), 1/dim_sqrt)
                '''

                # Add the resulting offspring to P                
                population_size += 1
                self.pop_data[population_size] = {
                    'x' : offspring,
                    'obj': self.rescale_one(fun(offspring)),
                    'fitness': 0.0
                }
                
            generation += 1

        best_fitness = -np.infty
        best_fit = None
        for (item, data) in self.pop_data.items():
            if data['fitness'] >= best_fitness:
                best_fitness = data['fitness']
                best_fit = item

        return self.pop_data[best_fit]['x']
    








    def DerandomizedMutation(x,sigma,E,z,n):
        # E = global step-size
        # z = N(0,I) = mutation steps
        # d = sqrt(n)
        # sigma = vector of step-sizes and/or standard deviations

        d=np.sqrt(n)
        xRes=x+np.exp(E)*np.dot(sigma,z,out=None)
        sigmaRes=np.dot(sigma , np.power(np.exp(np.norm(z)/np.mean(E*3) -1), 1/n),out=None)*np.power(np.exp(E),1/d)
        
        return xRes, sigmaRes

    def recombinationESsearchPath(x,sigma,z,n,lamda,E):
        
        '''
        sigma = vector of step-sizes and/or standard deviations
        n =
        lamda = number of offspring, offspring population size
        E = global step-size = N(0,1)
        z = N(0,I) = mutation steps
        u = number of parents, parental population size
        s = search path or evolution path
        
        '''
        c= np.sqrt(u/(n+4))
        u= lamda/4
        d= 1+np.sqrt(u/n)
        di=3*n
        s=0
        s=(1-c)*s+np.sqrt(c*(2-c))*(np.sqrt(u)/u)*np.sum(z)
        sigma = np.dot(sigma,np.power(np.exp((np.norm(s)/np.mean(E))-1),1/di))*np.power(np.exp(np.norm(s)/np.mean(z) -1),c/d)
        xRes = (1/u)*np.sum(x)
        return xRes




    def cxSimulatedBinary(ind1, ind2, eta):
        """Executes a simulated binary crossover that modify in-place the input
         individuals. The simulated binary crossover expects :term:`sequence`
         individuals of floating point numbers.
         
         :param ind1: The first individual participating in the crossover.
         :param ind2: The second individual participating in the crossover.
         :param eta: Crowding degree of the crossover. A high eta will produce
         children resembling to their parents, while a small eta will
         produce solutions much more different.
         :returns: A tuple of two individuals.
         
         This function uses the :func:`~random.random` function from the python base
         :mod:`random` module.
         """
        for i, (x1, x2) in enumerate(zip(ind1, ind2)):
            rand = random.random()
            if rand <= 0.5:
                beta = 2. * rand
            else:
                beta = 1. / (2. * (1. - rand))
                beta **= 1. / (eta + 1.)
                ind1[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))
                ind2[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))
        
        return ind1, ind2










    def compute_fitness(self, particle):
        ''' For all vectors in P\{particle}, compute pairwise indicator function
        and sum to get fitness value.'''
        neg_sum = 0.0
        for indx in self.pop_data.keys():
            if indx != particle:
                neg_sum -= np.exp(-self.eps_indic_fun(indx, particle)\
                                  /(self.indicator_max * self.kappa))

        self.pop_data[particle]['fitness'] = neg_sum

    def eps_indic_fun(self, i1, i2):
        obj1 = self.pop_data[i1]['obj']
        obj2 = self.pop_data[i2]['obj']
        diff = obj1 - obj2
        eps = diff.min()

        assert -1 <= eps <= 1, \
            'Bounds not respected: O1 = {}, O2 = {}, eps = {}'\
            .format(obj1, obj2, eps)

        return eps

    def rescale(self, objective):
        self._min = objective.min(axis=0)
        self._max = objective.max(axis=0)

        _, ndims = objective.shape
        for dim in range(ndims):
            objective[:, dim] = (objective[:, dim] - self._min[dim]) \
                      / (self._max[dim] - self._min[dim])
        return objective

    def rescale_one(self, objective):
        self._min = np.minimum(self._min, objective)
        self._max = np.maximum(self._max, objective)
        for dim in range(objective.shape[0]):
            objective[dim] = (objective[dim] - self._min[dim]) \
                   / (self._max[dim] - self._min[dim])
        return objective

if __name__ == '__main__':
    """call `experiment.main()`"""
    import experiment as cocoexp
    # main(budget, max_runs, current_batch, number_of_batches)
    cocoexp.main()
