#!/usr/bin/env python2
# -*- coding : utf8 -*-
import numpy as np

class IBEA(object):
    def __init__(self,
                 rho=1.1,
                 kappa=0.05, 
                 alpha=10, 
                 mu=1, 
                 offspring=1, 
                 seed=1):
        # --- Algorithm parameters
        self.rho = rho
        self.kappa = kappa         # fitness scaling ratio
        self.alpha = alpha         # population size
        self.mu = mu               # number of individuals selected as parents
        self.offspring = offspring # number of offspring individuals
        # --- Population data structures
        self.pop_vectors = None
        self.population = []
        # --- Random state
        np.random.seed(seed)

    def __str__(self):
        return 'Indicator-based Evolutionary Algorithm - Epsilon with params {}'\
            .format()

    def ibea(self, fun, lbounds, ubounds, budget):
        self.lbounds, self.ubounds = np.array(lbounds), np.array(ubounds)
        dim, f_min = len(lbounds), None

        # 1. Initial population of size alpha
        # Sampled from the uniform distribution [lbounds, ubounds]
        self.population = list(range(self.alpha))
        self.pop_vectors = np.random.rand(self.alpha, dim) \
                           * (ubounds - lbounds) + lbounds
        # Rescaling objective values to [0,1]
        self.objective_values = self.rescale(np.array([fun(x) for x in self.pop_vectors]))
        assert self.objective_values.shape == (self.alpha, dim), 'Not the right dimension'
        self.fitness_values = np.zeros(self.alpha, dtype=np.float64)        
        done = False
        generation = 0 
        while not done:
            # 2. Fitness assignment
            for i in range(self.alpha):
                self.fitness_values[i] = self.compute_fitness(i)
            # 3 Environmental selection
            env_selection_ok = False
            while not env_selection_ok:
                # 3.1 Environmental selection
                unfit_individual = self.fitness_values.argmin()
                # 3.2 Remove individual
                self.population.remove(unfit_individual)
                # 3.3 Update fitness values
                for i in self.population:
                    self.fitness_values[i] += np.exp(
                        -self.indicator_e(unfit_individual,i)/self.kappa)
                env_selection_ok = len(self.population) == self.alpha

            # 4. Check convergence condition
            done = generation >= budget
            if done: break
            # 5. Mating selection
            # Perform binary tournament selection with replacement on P in order
            # to fill the temporary mating pool P'.
            # 6. Variation
            # Apply recombination and mutation operators to the mating pool P' and add
            # the resulting offspring to P
            generation += 1

        return f_min
    
    def compute_fitness(self, ind):
        ''' For all vectors in P\{ind}, compute pairwise indicator function
        and sum to get fitness value.'''
        for i in self.population:
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
        df = self.objective_values[i1] - self.objective_values[i2]
        return df.min()

if __name__ == '__main__':
    """call `experiment.main()`"""
    import experiment as cocoexp
    # main(budget, max_runs, current_batch, number_of_batches)
    cocoexp.main()
