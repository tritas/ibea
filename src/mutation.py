#!/usr/bin/env python2
# -*- coding : utf8 -*-

""" Indicator-based Evolutionary Algorithm with Epsilon indicator
Recombination operators, also known as `crossover` operators.
"""
from __future__ import division
from numpy import sqrt, power, exp, float64, infty
from numpy import array, empty, divide, minimum, maximum
from numpy.random import seed, choice, binomial
from numpy.random import rand, randint, randn

def DerandomizedMutation(x,sigma,E,z,n):
        # E = global step-size
        # z = N(0,I) = mutation steps
        # d = sqrt(n)
        # sigma = vector of step-sizes and/or standard deviations
        d = np.sqrt(n)
        xRes = x+np.exp(E)*np.dot(sigma,z,out=None)
        sigmaRes = np.dot(sigma , np.power(np.exp(np.norm(z)/np.mean(E*3) -1), 1/n),out=None)\
                   * np.power(np.exp(E),1/d)
        
        return xRes, sigmaRes

def recombinationESsearchPath(x,sigma,z,n,lamda,E):
    '''
    sigma = vector of step-sizes and/or standard deviations
    n =
    lamda = number of offspring, offspring population size
    E = global step-size = N(0,1)
    z = N(0,I) = mutation steps
    u = number of parents, parental population size
    s = search path or evolution path
    '''

    c= np.sqrt(u/(n+4))
    u= lamda/4
    d= 1+np.sqrt(u/n)
    di=3*n
    s=0
    s=(1-c)*s+np.sqrt(c*(2-c))*(np.sqrt(u)/u)*np.sum(z)
    sigma = np.dot(sigma, np.power(np.exp((np.norm(s)/np.mean(E))-1),1/di)) \
            *np.power(np.exp(np.norm(s)/np.mean(z) -1),c/d)
    xRes = (1/u)*np.sum(x)
    return xRes    
