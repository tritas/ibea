#!/usr/bin/env python2
# -*- coding : utf8 -*-

from operator import mod
from pprint import pprint
import numpy as np

class IBEA(object):
    def __init__(self,
                 kappa=0.05, 
                 alpha=20, 
                 n_offspring=8, 
                 seed=1,
                 var=0.01):
        # --- Algorithm parameters
        self.kappa = kappa             # Fitness scaling ratio
        self.alpha = alpha             # Population size
        self.noise_variance = var      # Noise level for mutation
        self.n_offspring = n_offspring # Number of offspring individuals
        self._min = None               # Objective function minima
        self._max = None               # Objective function maxima
        self.indicator_max = None      # Indicator function maximum
        
        # --- Data structure containing: population vectors, fitness and objective values
        self.environment = dict() 

        # --- Random state
        np.random.seed(seed)

    def __str__(self):
        return 'Indicator-based Evolutionary Algorithm with Epsilon indicator'

    def ibea(self, fun, lbounds, ubounds, budget):
        lbounds, ubounds = np.array(lbounds), np.array(ubounds) # [-100, 100]
        dim, f_min = len(lbounds), None
        
        # 1. Initial population of size alpha
        # Sampled from the uniform distribution [lbounds, ubounds]
        particles = np.random.rand(self.alpha, dim) \
                 * (ubounds - lbounds) + lbounds
        # Rescaling objective values to [0,1]
        objective_values = np.array([fun(x) for x in particles])
        objective_values = self.rescale(objective_values)
        # Datastructure containing all the population info
        self.environment = {
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
            for i in self.environment.keys():
                self.compute_fitness(i)
            # 3 Environmental selection
            env_selection_ok = False
            while not env_selection_ok:
                # 3.1 Environmental selection
                fit_min = 10
                worst_fit = None
                for (k, v) in self.environment.items():
                    if v['fitness'] <= fit_min:
                        fit_min = v['fitness'] 
                        worst_fit = k

                # 3.3 Update fitness values
                for i in self.environment.keys():
                    self.environment[i]['fitness'] += np.exp(
                        - self.eps_indic_fun(worst_fit, i) \
                        / (self.indicator_max * self.kappa))
                # 3.2 Remove individual
                population_size -= 1
                self.environment.pop(worst_fit)
                # Continue while P does not exceed alpha
                # TOFIX: does not work with population_size variable
                env_selection_ok = len(self.environment) <= self.alpha

            # 4. Check convergence condition
            done = generation >= budget
            if done: break
            # 5. Mating selection
            # Perform binary tournament selection with replacement on P in order
            # to fill the temporary mating pool P'.
            item_keys = list(self.environment.keys())
            pool = []
            for i in range(2*self.n_offspring):
                p1, p2 = np.random.choice(item_keys, 2)
                if self.environment[p1]['fitness'] >= self.environment[p2]['fitness']:
                    pool.append(p1)
                else:
                    pool.append(p2)
            # 6. Variation
            assert mod(len(pool), 2) == 0, 'Parents not divisible by 2'
            for i in range(self.n_offspring):
                x1 = self.environment[pool[i]]['x']
                x2 = self.environment[pool[i+1]]['x']
                # Apply recombination and mutation operators to the mating pool P'
                # and add the resulting offspring to P
                pr_genes = np.random.binomial(1, 0.5, dim)
                offspring = np.empty(dim, dtype=np.float64)
                for d in range(dim):
                    offspring[d] = x1[d] if pr_genes[d] else x2[d]
                offspring += np.random.randn(dim) * self.noise_variance
                
                population_size += 1
                self.environment[population_size] = {
                    'x' : offspring,
                    'obj': self.rescale_one(fun(offspring)),
                    'fitness': 0.0
                }
            generation += 1

        best_fitness = -np.infty
        best_fit = None
        for (item, data) in self.environment.items():
            if data['fitness'] >= best_fitness:
                best_fitness = data['fitness']
                best_fit = item

        return self.environment[best_fit]['x']
    
    def compute_fitness(self, particle):
        ''' For all vectors in P\{particle}, compute pairwise indicator function
        and sum to get fitness value.'''
        neg_sum = 0.0
        for indx in self.environment.keys():
            if indx != particle:
                neg_sum -= np.exp(-self.eps_indic_fun(indx, particle)\
                                  /(self.indicator_max * self.kappa))

        self.environment[particle]['fitness'] = neg_sum

    def eps_indic_fun(self, i1, i2):
        obj1 = self.environment[i1]['obj']
        obj2 = self.environment[i2]['obj']
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
