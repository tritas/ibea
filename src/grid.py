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
from experiment import batch_loop, ascetime, 
verbose = 1

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
    suite_options = "dimensions: 2,5,10,20"  # "dimensions: 2,3,5,10,20 "  # if 40 is not desired
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
    budget = 500 # maxfevals = budget x dimension
    max_runs = 1e6  # number of (almost) independent trials per problem instance
    number_of_batches = 1  # allows to run everything in several batches
    current_batch = 1      # 1..number_of_batches
    
    probas_crossover_mutation = [(1.0, 0.01), (0.8, 1.0), (0.8, 0.8), (0.5, 0.8)]
    initial_variance_lst = [2, 5]
    grid = [{'pr_mut' : mut, 'pr_x' : x, 'var': var}
                  for var in initial_variance_lst
                  for (x, mut) in probas_crossover_mutation]
    optimizer_instances = [IBEA(**kwargs) for kwargs in grid]

    # Run grid search with pool
    with Pool(max(8, len(params_lst))) as pool:
        pass
