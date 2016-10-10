#!/usr/bin/env python2
# -*- coding : utf8 -*-

""" Indicator-based Evolutionary Algorithm with Epsilon indicator
Recombination operators, also known as `crossover` operators.
"""
from __future__ import division, print_function
from sys import stderr
from traceback import format_exc
from numpy import empty, clip, divide, power, exp, float64, nan, inf, seterr
from numpy.random import binomial, randint, rand
seterr(all='raise')

def discrete_recombination(x1, x2, dim):
    ''' Discrete recombination: dimensions \times coin flips i.e Bernouilli(0.5)
    to decide which parent's value to inherit - no reason for that to work, it's too random '''
    pr_genes = binomial(1, 0.5, dim)
    offspring = empty(dim, dtype=float64)
    for d in range(dim):
        offspring[d] = x1[d] if pr_genes[d] else x2[d]
    return offspring

def weighted_recombination(x1, x2, coef=0.5):
    '''  Weighted recombination (noted \rho_W) - how to choose optimal weight coef?
    Intermediate recombination (noted \rho_I) is the default case with coef = 0.5 '''
    offspring = divide(coef * x1 + (1 - coef) * x2, 2)
    return offspring

def one_point_crossover(x1, x2, dim):
    # One-point crossover
    x_ind = randint(dim)
    offspring = empty(dim, dtype=float64)
    offspring[:x_ind] = x1[:x_ind]
    offspring[x_ind:] = x2[x_ind:]
    return offspring

def bounded_sbx(parent1, parent2, lbounds, ubounds, eta=0.5):
    ''' Implementation of the Simulated Binary Crossover operator
    The idea is to compute the spread factor `beta` at each index and produce child
    values bounded in (lbound, ubound) s.t the crossover is stationary with high probability.
    To do that we compute `beta` cumulative probability distribution
    :param eta: 
    :reference: Deb, Kalyanmoy, and Ram B. Agrawal. 
    "Simulated binary crossover for continuous search space." 
    Complex Systems 9.3 (1994): 1-15. ''' 
    
    dim = parent1.shape[0]
    child1 = empty(dim, dtype=float64)
    child2 = empty(dim, dtype=float64)
    alpha = 0.0
    beta = 0.0
    beta_cumul = 0.0
    
    for i in range(dim):
        if rand() <= 0.5 and abs(parent2[i] - parent1[i]) > 0:
            try:
                y1 = min(parent1[i], parent2[i])
                y2 = max(parent1[i], parent2[i])
                rand_float = rand()
                # Compute first child's value
                beta = 1.0 + 2.0 * (y1 - lbounds[i]) / (y2 - y1)
                alpha = 2.0 - power(beta, -(eta + 1.0))
                if rand_float <= (1.0/alpha):
                    beta_cumul = power(rand_float*alpha, -(eta + 1.0))
                else:
                    beta_cumul = power(1.0/(2.0-alpha*rand_float), 1.0/(eta+1.0))


                c1 = 0.5 * (y1+y2 - beta_cumul * (y2-y1))

                # Compute second child's value
                beta = 1.0 + 2.0 * (ubounds[i] - y2) / (y2 - y1)
                alpha = 2.0 - power(beta, -(eta + 1.0))
                if rand_float <= (1.0/alpha):
                    beta_cumul = power(rand_float*alpha, -(eta + 1.0))
                else:
                        beta_cumul = power(1.0/(2.0-alpha*rand_float), 1.0/(eta+1.0))

                c2 = 0.5 * (y1+y2 + beta_cumul * (y2-y1))

                # Clip in bounds
                c1, c2 = clip([c1, c2], lbounds[i], ubounds[i])

                # Place children values in final arrays with random permutation
                if rand() <= 0.5:
                    child1[i] = c1
                    child2[i] = c2
                else:
                    child2[i] = c1
                    child1[i] = c2
            except FloatingPointError, RuntimeWarning:
                print(format_exc())
                print('alpha={}, beta={}, beta_cumul={}'.format(alpha, beta, beta_cumul))
        else:
            child1[i] = parent1[i]
            child2[i] = parent2[i]

    return child1, child2
