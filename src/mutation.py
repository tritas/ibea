#!/usr/bin/env python2
# -*- coding : utf8 -*-
# Author: Daro Heng <daro.heng@u-psud.fr>
""" Indicator-based Evolutionary Algorithm with Epsilon indicator
Recombination operators, also known as `crossover` operators.
"""
from __future__ import division
from numpy import sqrt, power, exp, float64, infty, dot
from numpy import power, mean, norm, sum, abs
from numpy.random import randn

def DerandomizedMutation(x, sigma, E, n):
        ''' E = global step-size, 
        sigma = vector of step-sizes and/or standard deviations '''
        z = randn(n)
        expct = mean(abs(randn(n)))
        one = ones(n)
        d = sqrt(n)
        xRes = x + exp(E) * dot(sigma,z)
        num = power(exp( abs(z) / expct - one), 1/n)
        sigmaRes = dot(sigma, num) * power(exp(E),1/d)

        return xRes, sigmaRes

def recombinationESsearchPath(x, sigma, n, lamda):
        '''
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
    ''' Adapt step-size using the 1/5-th rule '''
    indicator = int(parent_fitness <= offspring_fitness)
    mult = power(exp(indicator - 0.2), inv_dim_sqrt)
    sigma *= mult
    return sigma
