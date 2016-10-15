#!/usr/bin/env python2
# -*- coding : utf8 -*-
# Copyright 2016 (c) Daro Heng <daro.heng@u-psud.fr>, Aris Tritas <aris.tritas@u-psud.fr>
# Licence: BSD 3 clause
""" Some mutation operators for evolution strategies. """

from __future__ import division
from traceback import format_exc
#from math import pow
from numpy import ones, sqrt, power, exp, float64, infty, dot
from numpy import power, mean, sum, abs
from numpy.linalg import norm
from numpy.random import randn

from numpy import seterr
seterr(all='raise')

def DerandomizedMutation(x, sigma, n):
        ''' Intialization conditions: lamda ~ 10
        :sigma : vector of step-sizes and/or standard deviations 
        :reference : Nikolaus Hansen, Dirk V. Arnold and Anne Auger, 
        Evolution Strategies, February 2015.
        '''
        try:
                # Initialize variables
                d = sqrt(n)
                tau = 1/3
                ksi = tau * randn()
                z = randn(n)
                one = ones(n)
                # Expectation[|N(0,1)|] - Is this right?
                expct = abs(randn())
                assert expct > 1e-14, 'Divide by low expectation (error), draw was {}'.format(expct)
                # Mutate vector
                xRes = x + exp(ksi) * dot(sigma, z)
                # Compute sigma adaptation
                adaptation_vect = exp((abs(z) / expct - one)/n)
                # Compute new value of sigma
                adapted_sigma = dot(sigma, adaptation_vect) * exp(ksi/d)
        except FloatingPointError, RuntimeWarning:
                print(format_exc())
                exit(2)
        except ValueError:
                print(format_exc())
                exit(2)
        return xRes, adapted_sigma

def recombinationESsearchPath(x, sigma, n, lamda):
        ''' (mu/mu, lambda)-ES with Search Path Algorithm
        :reference : Nikolaus Hansen, Dirk V. Arnold and Anne Auger, 
        Evolution Strategies, February 2015.

        sigma = vector of step-sizes and/or standard deviations
        n =
        lamda = number of offspring, offspring population size
        E = global step-size = N(0,1)
        u = number of parents
        s = search path or evolution path
        '''
        E = abs(randn())
        z = randn(n)
        c= sqrt(u/(n+4))
        u= lamda/4
        d= 1+sqrt(u/n)
        di=3*n
        s=0
        s=(1-c)*s+sqrt(c*(2-c))*(sqrt(u)/u)*sum(z)
        sigma_norm = power(exp(norm(s)/mean(z) -1),c/d)
        sigma_abs = power(exp((norm(s)/mean(E))-1),1/di)
        sigma = dot(sigma, sigma_abs) * sigma_norm
        xRes = (1/u)*sum(x)
        return xRes

def one_fifth_success(sigma, offspring_fitness, parent_fitness, inv_dim_sqrt):
        ''' Adapt step-size using the 1/5-th rule, :references [Retchenberg, Schumer, Steiglitz]
        :param inv_dim_sqrt: inversed of the square root of problem dimension.
        The idea is to raise, in expectation, the log of the variance if the success probability
        is larger than 1/5, and decrease it otherwise. Note: for our fitness function bigger is better'''
        indicator = int(parent_fitness <= offspring_fitness)
        mult = power(exp(indicator - 0.2), inv_dim_sqrt)
        sigma *= mult
        return sigma
