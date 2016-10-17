#!/usr/bin/env python2
# -*- coding : utf8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
#         COCO contributed `experiment.py`
# Licence: BSD 3 clause
from multiprocessing import Pool
import time
import cocoex
from cocoex import Suite, Observer, log_level
from ibea import IBEA
from experiment import batch_loop, ascetime
from numpy import arange
verbose = 1
budget = 1000 # maxfevals = budget x dimension
max_runs = 1e6  # number of (almost) independent trials per problem instance
number_of_batches = 1  # allows to run everything in several batches
current_batch = 1      # 1..number_of_batches

def run(solver_object, budget=budget,
        max_runs=max_runs,
        current_batch=current_batch,
        number_of_batches=number_of_batches):
    """Initialize suite and observer, then benchmark solver by calling
    `batch_loop(SOLVER, suite, observer, budget,...`.
    """
    ##############################################################################
    solver = solver_object.ibea
    suite_name = "bbob-biobj"
    suite_instance = "year:2016"
    suite_options = "dimensions: 2,3,5"  # "dimensions: 2,3,5,10,20 "  # if 40 is not desired
    observer_name = suite_name
    observer_options = (
    ' result_folder: %s_on_%s_budget%04dxD '
                 % (str(solver_object), suite_name, budget) +
    ' algorithm_name: %s ' % solver.__name__ +
    ' algorithm_info: "Indicator-based Evolutionary Algorithm (epsilon)" ')
    observer = Observer(observer_name, observer_options)
    suite = Suite(suite_name, suite_instance, suite_options)
    print("Benchmarking solver '%s' with budget=%d*dimension on %s suite, %s"
          % (' '.join(str(solver).split()[:2]), budget,
             suite.name, time.asctime()))
    if number_of_batches > 1:
        print('Batch usecase, make sure you run *all* %d batches.\n' %
              number_of_batches)
    t0 = time.clock()
    batch_loop(solver, suite, observer, budget, max_runs,
               current_batch, number_of_batches)
    print(", %s (%s total elapsed time)." % (time.asctime(), ascetime(time.clock() - t0)))

if __name__ == '__main__':
    grid = []
    
    fix_mutation = 0.8
    fix_crossover = 0.7
    
    pr_mutation = arange(0.1, 1.0, 0.2)
    pr_crossover = arange(0.5, 1.0, 0.1)
    variance = arange(3, 9, 2)
    n_sbx = [2, 5, 10]
    alpha = arange(50, 110, 10)
    offspring = arange(20, 60, 10)

    for m in pr_mutation:
        grid.append(IBEA(pr_mut=m, pr_x=fix_crossover))
    for c in pr_crossover:
        grid.append(IBEA(pr_mut=fix_mutation, pr_x=c))
    for v in variance:
        grid.append(IBEA(pr_mut=fix_mutation, pr_x=fix_crossover, var=v))
    for n in n_sbx:
        grid.append(IBEA(pr_mut=fix_mutation, pr_x=fix_crossover, n_sbx=n))
    for a in alpha:
        grid.append(IBEA(pr_mut=fix_mutation, pr_x=fix_crossover, alpha=a))
    for o in offspring:
        grid.append(IBEA(pr_mut=fix_mutation, pr_x=fix_crossover, n_offspring=o))

    print("Total configurations to test: {}".format(len(grid)))

    # Run grid search with pool
    #with Pool() as pool:
    pool = Pool()
    try:
        results = pool.map(run, grid)
    except KeyboardInterrupt:
        pool.terminate()
