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
        self.noise_var = var
        self.n_offspring = n_offspring # number of offspring individuals
        self._min = None 
        self._max = None
        # --- Data structure containing: population vectors, fitness and objective values
        self.env = dict() 
        # --- Random state
        np.random.seed(seed)

    def __str__(self):
        return 'Indicator-based Evolutionary Algorithm with Epsilon indicator'

    def ibea(self, fun, lbounds, ubounds, budget):
        lbounds, ubounds = np.array(lbounds), np.array(ubounds) # [-100, 100]
        dim, f_min = len(lbounds), None
        # 1. Initial population of size alpha
        # Sampled from the uniform distribution [lbounds, ubounds]
        X = np.random.rand(self.alpha, dim) \
                 * (ubounds - lbounds) + lbounds
        # Rescaling objective values to [0,1]
        F = np.array([fun(x) for x in X])
        F = self.rescale(F)
        # Datastructure containing all the population info
        self.env = {
            p : {
                'x': X[p],
                'obj': F[p],
                'fitness': 0.0,
            } for p in range(self.alpha)
        }
        self.n_pop = self.alpha
        done = False
        generation = 0 
        while not done:
            # 2. Fitness assignment
            for i in self.env.keys():
                self.compute_fitness(i)
            # 3 Environmental selection
            env_selection_ok = False
            while not env_selection_ok:
                # 3.1 Environmental selection
                fit_min = 10
                unfit_individual = None
                for (k, v) in self.env.items():
                    if v['fitness'] <= fit_min:
                        fit_min = v['fitness'] 
                        unfit_individual = k

                # 3.2 Remove individual (1)
                self.n_pop -= 1
                # 3.3 Update fitness values
                for i in self.env.keys():
                    try:
                        self.env[i]['fitness'] += np.exp(
                            - self.indicator_e(unfit_individual, i) / self.kappa)
                    except TypeError:
                        pprint(self.env)
                # 3.2 Remove individual (2)
                self.env.pop(unfit_individual)
                env_selection_ok = len(self.env) <= self.alpha

            # 4. Check convergence condition
            done = generation >= budget
            if done: break
            # 5. Mating selection
            # Perform binary tournament selection with replacement on P in order
            # to fill the temporary mating pool P'.
            keys_lst = list(self.env.keys())
            winner_lst = []
            for i in range(self.n_offspring):
                p1, p2 = np.random.choice(keys_lst, 2)
                if self.env[p1]['fitness'] >= self.env[p2]['fitness']:
                    winner_lst.append(p1)
                else:
                    winner_lst.append(p2)
            # 6. Variation
            # Apply recombination and mutation operators to the mating pool P' and add
            # the resulting offspring to P
            assert mod(len(winner_lst), 2) == 0, 'Parents not divisible by 2'
            for i in range(len(winner_lst)/2):
                x1 = self.env[winner_lst[i]]['x']
                x2 = self.env[winner_lst[i+1]]['x']                
                offspring = (x1+x2)/2
                offspring += np.random.randn(dim) * self.noise_var
                
                self.n_pop += 1
                self.env[self.n_pop] = {
                    'x' : offspring,
                    'obj': self.rescale_one(fun(offspring)),
                    'fitness': 0.0
                }
            generation += 1

        best_fitness = -np.infty
        best_fit = None
        for (k, v) in self.env.items():
            if v['fitness'] >= best_fitness:
                best_fitness = v['fitness']
                best_fit = k

        return self.env[best_fit]['x']
    
    def compute_fitness(self, ind):
        ''' For all vectors in P\{ind}, compute pairwise indicator function
        and sum to get fitness value.'''
        neg_sum = 0.0
        for i in self.env.keys():
            if i != ind:
                neg_sum -= np.exp(-self.indicator_e(i, ind)/self.kappa)

        self.env[ind]['fitness'] = neg_sum

    def indicator_e(self, i1, i2):
        o1 = self.env[i1]['obj']
        o2 = self.env[i2]['obj']
        df = o1 - o2
        eps = df.min()
        assert -1 <= eps <= 1, 'Bounds not respected: O1 = {}, O2 = {}, eps = {}'.format(o1, o2, eps)
        return eps

    def rescale(self, F):
        self._min = F.min(axis=0)
        self._max = F.max(axis=0)

        _, dim = F.shape
        for i in range(dim):
            F[:, i] = (F[:, i] - self._min[i]) \
                      / (self._max[i] - self._min[i])
        return F

    def rescale_one(self, f):
        self._min = np.minimum(self._min, f)
        self._max = np.maximum(self._max, f)
        for i in range(f.shape[0]):
            f[i] = (f[i] - self._min[i]) \
                   / (self._max[i] - self._min[i])
        return f

if __name__ == '__main__':
    """call `experiment.main()`"""
    import experiment as cocoexp
    # main(budget, max_runs, current_batch, number_of_batches)
    cocoexp.main()
