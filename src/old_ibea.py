#!/usr/bin/env python2
# -*- coding : utf8 -*-
from operator import mod
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
        assert 2*self.n_offspring == self.mu, 'Two parents per offspring'
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
        lbounds, ubounds = np.array(lbounds), np.array(ubounds)
        dim, f_min = len(lbounds), None

        # 1. Initial population of size alpha
        # Sampled from the uniform distribution [lbounds, ubounds]
        self.population = list(range(self.alpha))
        X = np.random.rand(self.alpha, dim) \
                 * (ubounds - lbounds) + lbounds
        # Rescaling objective values to [0,1]
        objective_values = np.array([fun(x) for x in self.X])
        objective_values = (self.objective_values - lbounds)/(ubounds - lbounds)
        assert self.objective_values.shape == (self.alpha, dim), 'Not the right dimension'
        self.fitness_values = np.zeros(self.alpha, dtype=np.float64)
        done = False
        generation = 0 
        while not done:
            # 2. Fitness assignment
            for i in self.population:
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
            winner_lst = []
            for i in range(self.n_offspring):
                p1, p2 = np.random.choice(len(self.population), 2)
                if self.fitness_values[p1] > self.fitness_values[p2]:
                    winner_lst.append(p1)
                else:
                    winner_lst.append(p2)
            # 6. Variation
            # Apply recombination and mutation operators to the mating pool P' and add
            # the resulting offspring to P
            assert mod(len(winner_lst), 2) == 0, 'Parents not divisible by 2'
            for i in range(len(winner_lst)/2):
                offspring = (self.X[winner_lst[i]] + self.X[winner_lst[i+1]])/2
                offspring += np.random.randn(dim) * self.noise_level
                #offspring = np.clip(offspring, 0, 1)
                self.X = np.hstack((self.X, offspring))
                self.population.append(len(self.population)+1)
                self.fitness_values = np.hstack((self.fitness_values, 0))
                self.objective_values = np.hstack((self.objective_values, fun(offspring)))
            generation += 1

        best_fit = self.fitness_values.argmax()
        return self.X[best_fit]
    
    def compute_fitness(self, ind):
        ''' For all vectors in P\{ind}, compute pairwise indicator function
        and sum to get fitness value.'''
        for i in self.population:
            if i != ind:
                exp_indic = np.exp(-self.indicator_e(i, ind)/self.kappa)
                self.fitness_values[ind] -= exp_indic # -= instead of += because of negative sum

        return

    def indicator_e(self, i1, i2):
        df = self.objective_values[i1] - self.objective_values[i2]
        return df.min()

if __name__ == '__main__':
    """call `experiment.main()`"""
    import experiment as cocoexp
    # main(budget, max_runs, current_batch, number_of_batches)
    cocoexp.main()
