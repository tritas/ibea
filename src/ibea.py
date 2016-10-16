#!/usr/bin/env python2
# -*- coding : utf8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# Licence: BSD 3 clause

""" Indicator-based Evolutionary Algorithm with Epsilon indicator
is an evolutionary algorithm for searching a multi-objective space by improving on the Pareto front.
Formally, the algorithm is classified as a (\mu/\rho + \lambda)-ES, i.e.
\lambda offsprings are produced by \rho mates selected from \mu after a binary tournament.
An Evolution Strategy applies `+` selection if it only takes fitness into account for the selection phase.
"""
from __future__ import division
from traceback import format_exc
from operator import mod
from collections import deque

from numpy import sqrt, power, exp, float64, infty, seterr, all
from numpy import array, empty, full, divide, minimum, maximum, clip
from numpy.random import seed, choice, binomial
from numpy.random import rand, randint, randn

from crossover import bounded_sbx
from mutation import DerandomizedMutation , SearchPathMutationUpdate, one_fifth_success

seterr(all='raise')

class IBEA(object):
    def __init__(self,
                 kappa=0.05, 
                 alpha=100, 
                 n_offspring=8, 
                 seedit=42,
                 pr_x=1.0,
                 pr_mut=1.0,
                 var=2.0,  # TODO: Find sensible default
                 max_generations=200,
                 n_sbx=5, # Can be [2, 20], typically {2, 5}
                 mutation_operator='derandomized'): 
        # --- Algorithm parameters
        self.kappa = kappa             # Fitness scaling ratio
        self.alpha = alpha             # Population size
        self.pr_crossover = pr_x       # Crossover probability
        self.pr_mutation = pr_mut      # Mutation probability
        self.sigma_init = var          # Noise level for mutation
        self.n_offspring = n_offspring # Number of offspring individuals
        self._min = None               # Objective function minima
        self._max = None               # Objective function maxima
        self.indicator_max = None      # Indicator function maximum
        self.max_generations = max_generations
        self.n_sbx = n_sbx             # Simulated Binary Crossover distribution index
        self.mutation_operator = mutation_operator
        # --- Data structure containing: population vectors, fitness and objective values
        self.pop_data = dict()
        # --- Free indices for the population dictionary
        self.free_indices = deque()
        # --- Population counter
        self.population_size = 0

        # --- Random state
        seed(seedit)

    def __str__(self):
        #return 'Indicator-based Evolutionary Algorithm with Epsilon indicator'
        desc = 'ibea_pop{}_offs{}_mut{}_recomb{}_var{}{}_sbx{}_max_gen{}'\
               .format(self.alpha, self.n_offspring,
                       self.pr_mutation, self.pr_crossover,
                       self.sigma_init, self.mutation_operator,
                       self.n_sbx, self.max_generations)
        return desc
    
    def ibea(self, fun, lbounds, ubounds, remaining_budget):
        lbounds, ubounds = array(lbounds), array(ubounds) # [-100, 100]
        dim, f_min = len(lbounds), None
        dim_sqrt = sqrt(dim+1)
        # Pr(mutation) = 1/n
        # self.pr_mutation = 1/dim
        sigma = full(dim, self.sigma_init)
        # 1. Initial population of size alpha
        # Sampled from the uniform distribution [lbounds, ubounds]
        particles = rand(self.alpha, dim) * (ubounds - lbounds) + lbounds
        # Rescaling objective values to [0,1]
        objective_values = array([fun(x) for x in particles])
        objective_values = self.rescale(objective_values)
        remaining_budget -= self.alpha
        # Datastructure containing all the population info
        self.pop_data = {
            p : {
                'x': particles[p],
                'obj': objective_values[p],
                'fitness': 0.0,
            } for p in range(self.alpha)
        }
        # Lazy compute max absolute value of pairwise epsilon indicator 
        self.indicator_max = max([abs(self.epsilon_indicator(i1, i2))
                                  for i1 in range(self.alpha)
                                  for i2 in range(self.alpha)
                                  if i1 != i2])
        # --- Initialize variables
        done = False
        generation = 0
        self.population_size = self.alpha
        self.free_indices = deque(range(self.alpha, self.alpha+2*self.n_offspring))

        while not done:
            # 2. Fitness assignment
            for i in self.pop_data.keys():
                self.compute_set_fitness(i)
            # 3 Environmental selection
            env_selection_ok = False
            while not env_selection_ok:
                # 3.1 Environmental selection
                fit_min = infty
                worst_fit = None
                for (k, v) in self.pop_data.items():
                    if v['fitness'] <= fit_min:
                        fit_min = v['fitness'] 
                        worst_fit = k

                # 3.3 Update fitness values
                for i in self.pop_data.keys():
                    self.pop_data[i]['fitness'] += exp(- self.epsilon_indicator(worst_fit, i) \
                                                       / (self.indicator_max * self.kappa))
                # 3.2 Remove individual
                self.population_size -= 1
                self.pop_data.pop(worst_fit)
                self.free_indices.append(worst_fit)
                # Continue while P does not exceed alpha
                env_selection_ok = self.population_size <= self.alpha

            # 4. Check convergence condition
            done = remaining_budget <= self.alpha+2*self.n_offspring \
                   or generation >= self.max_generations
            if done: break
            # 5. Mating selection
            # Perform binary tournament selection with replacement on P in order
            # to fill the temporary mating pool P'.
            item_keys = list(self.pop_data.keys())
            pool = []
            for i in range(2*self.n_offspring):
                p1, p2 = choice(item_keys, 2)
                if self.pop_data[p1]['fitness'] >= self.pop_data[p2]['fitness']:
                    pool.append(p1)
                else:
                    pool.append(p2)
            # 6. Recombination and Variation applied to the mating pool.
            for i in range(self.n_offspring):
                parent1 = self.pop_data[pool[i]]
                parent2 = self.pop_data[pool[i+1]]

                # Recombination (crossover) operator
                if binomial(1, self.pr_crossover):
                    child1, child2 = bounded_sbx(parent1['x'], parent2['x'],
                                                 lbounds, ubounds, self.n_sbx)
                else:
                    child1 = parent1['x']
                    child2 = parent2['x']

                if binomial(1, self.pr_mutation):
                    #assert all(sigma > 0), 'Dirac detected, Variance = {})'.format(sigma)
                    child1, sigma = DerandomizedMutation(child1, sigma, dim)
                    child2, sigma = DerandomizedMutation(child2, sigma, dim)                           
                    # (Isotropic) mutation
                    #child1 += randn(dim) * self.sigma_init
                    #child2 += randn(dim) * self.sigma_init

                # Make sure vectors are still bounded
                child1 = clip(child1, lbounds, ubounds)
                child2 = clip(child2, lbounds, ubounds)

                obj_c1 = self.rescale_one(fun(child1))
                obj_c2 = self.rescale_one(fun(child2))
                remaining_budget -= 2

                # Make sure the maximum indicator value is up to date
                # Costs `alpha` but is necessary
                self.update_max_indicator(obj_c1)
                self.update_max_indicator(obj_c2)

                '''
                try:
                    fitness_c1 = self.compute_fitness(obj_c1)
                    fitness_c2 = self.compute_fitness(obj_c2)

                    self.sigma = one_fifth_success(self.sigma,
                                                            max(parent1['fitness'], parent2['fitness']),
                                                            max(fitness_c1, fitness_c2),
                                                            1/dim_sqrt)
                except FloatingPointError, RuntimeWarning:
                    print(format_exc())
                    print("F1 : {}, F2: {}, I: {}, coef: {}, sigma:{}"\
                          .format(fitness_c1, fitness_c2, indicator, mult, self.sigma))
                    exit(42)
                '''

                self.add_offspring(child1, obj_c1)
                self.add_offspring(child2, obj_c2)

            generation += 1

        # Choose vector maximizing fitness
        best_fitness = -infty
        best_fit = None
        for (item, data) in self.pop_data.items():
            if data['fitness'] >= best_fitness:
                best_fitness = data['fitness']
                best_fit = item

        return self.pop_data[best_fit]['x']

    def add_offspring(self, vector, objective_value, fitness=0.0):
        # Add the resulting offspring to P                
        self.population_size += 1
        indx = self.free_indices.pop()
        self.pop_data[indx] = {
            'x' : vector,
            'obj': objective_value,
            'fitness': fitness
        }
        
    def compute_fitness(self, objective_value, exclude=-1):
        ''' For all vectors in P\{exclude}, compute pairwise indicator function
        and sum to get fitness value.'''
        exp_sum = 0.0
        for (indx, data) in self.pop_data.items():
            if indx != exclude:
                exp_sum -= exp(-self.compute_epsilon(data['obj'], objective_value)\
                               /(self.indicator_max * self.kappa))
        return exp_sum

    def compute_set_fitness(self, particle):
        particle_obj = self.pop_data[particle]['obj']
        fitness = self.compute_fitness(particle_obj, particle)
        self.pop_data[particle]['fitness'] = fitness

    def epsilon_indicator(self, i1, i2):
        obj1 = self.pop_data[i1]['obj']
        obj2 = self.pop_data[i2]['obj']
        return self.compute_epsilon(obj1, obj2)

    def update_max_indicator(self, added_obj):
        epsilons = array([self.compute_epsilon(x['obj'], added_obj)
                          for x in self.pop_data.values()])
        self.indicator_max = max(self.indicator_max, epsilons.max())
    
    def compute_epsilon(self, obj1, obj2):
        ''' Smallest epsilon such that f(x1) - \eps * f(x2) < 0'''
        diff = obj1 - obj2
        eps = diff.min()

        assert -1 <= eps <= 1, \
            'Bounds not respected: O1 = {}, O2 = {}, eps = {}'\
            .format(obj1, obj2, eps)

        return eps

    def rescale(self, objective):
        # Save objective lower and upper bounds
        self._min = objective.min(axis=0)
        self._max = objective.max(axis=0)

        # Column-wise rescaling 
        _, ndims = objective.shape
        for dim in range(ndims):
            objective[:, dim] = (objective[:, dim] - self._min[dim]) \
                      / (self._max[dim] - self._min[dim])
        return objective

    def rescale_one(self, objective):
        # Update objective lower and upper bounds
        self._min = minimum(self._min, objective)
        self._max = maximum(self._max, objective)
        # Rescale vector
        for dim in range(objective.shape[0]):
            objective[dim] = (objective[dim] - self._min[dim]) \
                   / (self._max[dim] - self._min[dim])
        return objective

if __name__ == '__main__':
    """call `experiment.main()`"""
    import experiment as cocoexp
    # main(budget, max_runs, current_batch, number_of_batches)
    cocoexp.main()
