#!/usr/bin/env python2
# -*- coding : utf8 -*-
import numpy as np

class IBEA(object):
    def __init__(self,
                 rho=1.1,
                 kappa=0.05, # fitness scaling ratio
                 alpha=10, # population size
                 mu=1, # number of individuals selected as parents
                 offspring=1, # number of offspring individuals
                 seed=1):
        self.rho = rho
        self.kappa = kappa
        self.alpha = alpha
        self.mu = mu
        self.offspring = offspring
        np.random.seed(seed)

    def __str__(self):
        return 'Indicator-based Evolutionary Algorithm - Epsilon with params {}'\
            .format()

    def ibea(self, fun, lbounds, ubounds, budget):
        lbounds, ubounds = np.array(lbounds), np.array(ubounds)
        dim, x_min, f_min = len(lbounds), (lbounds + ubounds) / 2, None
        done = False
        # 1. Initialization
        self.population = np.zeros((self.alpha, dim), dtype=np.float64)
        self.fitness_values = np.zeros(self.alpha, dtype=np.float64)        
        generation = 0 
        while not done:
            # 2. Fitness assignment
            for i in range(self.alpha):
                self.fitness_values[i] = self.compute_fitness(i)
            # 3.1 Environmental selection
            unfit_individual = self.fitness_values.argmin()
            # 3.2 Remove individual (temporary hack)
            self.population[unfit_individual, :] = np.infty
            # 3.3 Update fitness values
            for i in range(self.alpha):
                self.fitness_values[i] += np.exp(-self.indicator_e(unfit_individual,i)/self.kappa)

            # 4. Check convergence condition
            done = generation >= budget
            if done: break
            # 5. Mating selection
            
            # 6. Variation
            generation += 1

        return f_min
    
    def compute_fitness(self, ind):
        ''' For all vectors in P\{ind}, compute pairwise indicator function
        and sum to get fitness value.'''
        for i in range(self.alpha):
            if i != ind:
                exp_indic = np.exp(-self.indicator_e(i, ind)/self.kappa)
                self.fitness_values[ind] -= exp_indic # -= instead of += because of negative sum

        return
    
    def dominates(x, y):
        component_wise_cmp = x <= y

    def rescale(f_x, lb, ub):
        ''' Rescale vector in [0, 1] ''' 
        return (f_x - lb)/(ub - lb)

    def indicator_e(self, i1, i2):
        x1 = self.population[i1]
        x2 = self.population[i2]        
        df = x1 - x2
        return max(0, df.min())

if __name__ == '__main__':
    """call `experiment.main()`"""
    import experiment as cocoexp
    # main(budget, max_runs, current_batch, number_of_batches)
    cocoexp.main()
