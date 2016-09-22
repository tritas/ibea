#!/usr/bin/env python2
# -*- coding : utf8 -*-
import numpy as np

class IBEA(object):
    def __init__(self,
                 rho=1.1,
                 kappa=0.05,
                 alpha=1, # number of individuals in initial pop
                 mu=1, # number of individuals selected as parents
                 lambda_=1, # number of offspring individuals
                 seed=1):
        pass

    def __str__(self):
        return 'Indicator-based Evolutionary Algorithm - Epsilon with params {}'\
            .format()
    
    def ibea(self, fun, lbounds, ubounds, budget):
        """Efficient implementation of uniform random search between `lbounds` and `ubounds`."""
        lbounds, ubounds = np.array(lbounds), np.array(ubounds)
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

    
    def compute_fitness(self): pass

    def dominates(x, y): pass


if __name__ == '__main__':
    """call `experiment.main()`"""
    import experiment as cocoexp
    # main(budget, max_runs, current_batch, number_of_batches)
    cocoexp.main()
