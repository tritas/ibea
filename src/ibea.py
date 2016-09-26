#!/usr/bin/env python2
# -*- coding : utf8 -*-
import numpy as np

class IBEA(object):
    def __init__(self,
                 rho=1.1,
                 kappa=0.05, # fitness scaling ratio
                 alpha=1, # population size
                 mu=1, # number of individuals selected as parents
                 _lambda=1, # number of offspring individuals
                 seed=1):
        self._rho = rho
        self._kappa = kappa
        self._alpha = alpha
        self._mu = mu
        self._lambda = _lambda
        np.random.seed(seed)

    def __str__(self):
        return 'Indicator-based Evolutionary Algorithm - Epsilon with params {}'\
            .format()

    def _initialize(self, ndims):
        self.population = np.zeros((self.alpha, ndims), dtype=np.float64)
        self.fitness_values = np.zeros(self.alpha, dtype=np.float64)
        
    def ibea(self, fun, lbounds, ubounds, budget):
        """Efficient implementation of uniform random search between `lbounds` and `ubounds`."""
        lbounds, ubounds = np.array(lbounds), np.array(ubounds)
        gen = 0

        dim, x_min, f_min = len(lbounds), (lbounds + ubounds) / 2, None
        max_chunk_size = 1 + 4e4 / dim
        while budget > 0:
            chunk = int(min([budget, max_chunk_size]))
            # about five times faster than "for k in range(budget):..."
            X = lbounds + (ubounds - lbounds) * np.random.rand(chunk, dim)
            F = [fun(x) for x in X]
            if fun.number_of_objectives == 1:
                index = np.argmin(F)
                if f_min is None or F[index] < f_min:
                    x_min, f_min = X[index], F[index]
            budget -= chunk
        return x_min

    def compute_fitness(self, ind):
        self.fitness_values[ind] = 0
        for i in range(self.alpha):
            if i != ind:
                self.fitness_values[ind] -= \
                    np.exp(-epsilon_indicator(i, ind)/self.kappa)
        return
    
    def dominates(x, y):
        component_wise_cmp = x < y

    def epsilon_indicator(set_A, set_B): pass

if __name__ == '__main__':
    """call `experiment.main()`"""
    import experiment as cocoexp
    # main(budget, max_runs, current_batch, number_of_batches)
    cocoexp.main()
