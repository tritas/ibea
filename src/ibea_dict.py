#!/usr/bin/env python2
# -*- coding : utf8 -*-

from operator import mod
from pprint import pprint
import numpy as np

class IBEA(object):
    def __init__(self,
                 rho=1.1,
                 kappa=0.05, 
                 alpha=10, 
                 mu=2, 
                 n_offspring=4, 
                 seed=1,
                 noise_level=0.01):
        # --- Algorithm parameters
        self.rho = rho
        self.kappa = kappa             # fitness scaling ratio
        self.alpha = alpha             # population size
        self.mu = mu                   # number of individuals selected as parents
        self.noise_level = noise_level
        self.n_offspring = n_offspring # number of offspring individuals
        assert self.n_offspring == 2*self.mu, 'Two parents per offspring'
        # --- Population data structures
        self.X = None
        self.population = []
        # --- Random state
        np.random.seed(seed)
        self.pop_dict = dict()

    def __str__(self):
        return 'Indicator-based Evolutionary Algorithm - Epsilon with params {}'\
            .format()

    def ibea(self, fun, lbounds, ubounds, budget):
        lbounds, ubounds = np.array(lbounds), np.array(ubounds) # [-100, 100]
        dim, f_min = len(lbounds), None
        # 1. Initial population of size alpha
        # Sampled from the uniform distribution [lbounds, ubounds]
        X = np.random.rand(self.alpha, dim) \
                 * (ubounds - lbounds) + lbounds
        # Rescaling objective values to [0,1]
        F = np.array([fun(x) for x in X])
        for i in range(dim):
            F[:, i] = (F[:, i] - F[:, i].min())/(F[:,i].max()- F[:,i].min())
        # Datastructure containing all the population info
        self.pop_dict = {
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
            for i in self.pop_dict.keys():
                self.compute_fitness(i)
            # 3 Environmental selection
            env_selection_ok = False
            while not env_selection_ok:
                # 3.1 Environmental selection
                fit_min = 10
                unfit_individual = None
                for (k, v) in self.pop_dict.items():
                    if v['fitness'] <= fit_min:
                        fit_min = v['fitness'] 
                        unfit_individual = k
                assert unfit_individual is not None, 'Iter: {}, min:{}'.format(generation, fit_min)
                # 3.2 Remove individual (1)
                self.n_pop -= 1
                # 3.3 Update fitness values
                for i in self.pop_dict.keys():
                    try:
                        self.pop_dict[i]['fitness'] += np.exp(
                            - self.indicator_e(unfit_individual, i) / self.kappa)
                    except TypeError:
                        pprint(self.pop_dict)
                # 3.2 Remove individual (2)
                self.pop_dict.pop(unfit_individual)
                env_selection_ok = len(self.pop_dict) == self.alpha

            # 4. Check convergence condition
            done = generation >= budget
            if done: break
            # 5. Mating selection
            # Perform binary tournament selection with replacement on P in order
            # to fill the temporary mating pool P'.
            winner_lst = []
            for i in range(self.n_offspring):
                p1, p2 = np.random.choice(len(self.population), 2)
                if self.pop_dict[p1]['fitness'] >= self.pop_dict[p2]['fitness']:
                    winner_lst.append(p1)
                else:
                    winner_lst.append(p2)
            # 6. Variation
            # Apply recombination and mutation operators to the mating pool P' and add
            # the resulting offspring to P
            assert mod(len(winner_lst), 2) == 0, 'Parents not divisible by 2'
            for i in range(len(winner_lst)/2):
                x1 = self.pop_dict[winner_lst[i]]['x']
                x2 = self.pop_dict[winner_lst[i+1]]['x']                
                offspring = (x1+x2)/2
                offspring += np.random.randn(dim) * self.noise_level
                
                self.n_pop += 1
                self.pop_dict[self.n_pop] = {
                    'x' : offspring,
                    'obj': fun(offspring),
                    'fitness': 0.0
                }
            generation += 1

        best_fitness = -1
        best_fit = None
        for i in self.pop_dict.keys():
            if self.pop_dict[i]['fitness'] >= best_fitness:
                best_fitness = self.pop_dict[i]['fitness']
                best_fit = i

        return self.pop_dict[best_fit]['x']
    
    def compute_fitness(self, ind):
        ''' For all vectors in P\{ind}, compute pairwise indicator function
        and sum to get fitness value.'''
        neg_sum = 0.0
        for i in self.pop_dict.keys():
            if i != ind:
                neg_sum -= np.exp(-self.indicator_e(i, ind)/self.kappa)

        self.pop_dict[ind]['fitness'] = neg_sum

    def indicator_e(self, i1, i2):
        df = self.pop_dict[i1]['obj'] - self.pop_dict[i2]['obj']
        assert -1 <= df.min() <= 1, 'Bounds not respected for epsilon'
        return df.min()

if __name__ == '__main__':
    """call `experiment.main()`"""
    import experiment as cocoexp
    # main(budget, max_runs, current_batch, number_of_batches)
    cocoexp.main()
