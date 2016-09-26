#!/usr/bin/env python2
# -*- coding : utf8 -*-

from operator import mod
from pprint import pprint
import numpy as np

class IBEA(object):
    def __init__(self,
                 rho=1.1,
                 kappa=0.05, 
                 alpha=100, 
                 mu=4, 
                 n_offspring=8, 
                 seed=1,
                 var=0.01):
        # --- Algorithm parameters
        self.rho = rho
        self.kappa = kappa             # fitness scaling ratio
        self.alpha = alpha             # population size
        self.mu = mu                   # number of individuals selected as parents
        self.noise_variance = var
        self.n_offspring = n_offspring # number of offspring individuals
        self._min = None 
        self._max = None
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
        self.population_size = self.alpha
        done = False
        generation = 0 
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

                # 3.2 Remove individual (1)
                self.population_size -= 1
                # 3.3 Update fitness values
                for i in self.environment.keys():
                    try:
                        self.environment[i]['fitness'] += np.exp(
                            - self.eps_indic_fun(worst_fit, i) / self.kappa)
                    except TypeError:
                        pprint(self.environment)
                # 3.2 Remove individual (2)
                self.environment.pop(worst_fit)
                env_selection_ok = len(self.environment) <= self.alpha

            # 4. Check convergence condition
            done = generation >= budget
            if done: break
            # 5. Mating selection
            # Perform binary tournament selection with replacement on P in order
            # to fill the temporary mating pool P'.
            item_keys = list(self.environment.keys())
            pool = []
            for i in range(self.n_offspring):
                p1, p2 = np.random.choice(item_keys, 2)
                if self.environment[p1]['fitness'] >= self.environment[p2]['fitness']:
                    pool.append(p1)
                else:
                    pool.append(p2)
            # 6. Variation
            # Apply recombination and mutation operators to the mating pool P' and add
            # the resulting offspring to P
            assert mod(len(pool), 2) == 0, 'Parents not divisible by 2'
            for i in range(len(pool)/2):
                x1 = self.environment[pool[i]]['x']
                x2 = self.environment[pool[i+1]]['x']                
                offspring = (x1+x2)/2
                offspring += np.random.randn(dim) * self.noise_variance
                
                self.population_size += 1
                self.environment[self.population_size] = {
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
    
    def compute_fitness(self, point):
        ''' For all vectors in P\{point}, compute pairwise indicator function
        and sum to get fitness value.'''
        neg_sum = 0.0
        for indx in self.environment.keys():
            if indx != point:
                neg_sum -= np.exp(-self.eps_indic_fun(indx, point)/self.kappa)

        self.environment[point]['fitness'] = neg_sum

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
