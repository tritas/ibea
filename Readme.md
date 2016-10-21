# Indicator-Based Evolutionary Algorithm (epsilon indicator) [1]

## Coco documentation

[Biobjective Performance Assessment with the COCO Platform](http://numbbo.github.io/coco-doc/bbob-biobj/perf-assessment/)

## Running the optimizer
There are two ways to run the optimizer: it is possible to change the experiment parameters in the `IBEA.py` file by parameterizing the call to `experiment.main`.
```shell
python2 ibea.py
```

It is also possible to parametrize and run `experiment.py`: 
```shell
python2 experiment.py <budget> <current_batch> <no_of_batches>
```

## Keeping testing parameters local
The idea is to keep the parameters we are testing local. Freely modify the solver object initialization in `experiment.py` in line 255: 

```python
SOLVER = IBEA().ibea
``` 

When you commit, please do `git add -p experiment.py` and ignore the above hunk. That way you do not to write your params on git and do not risk overwriting ours.

## Sharing experimental results
The idea is that `exdata`, the sub-directory in which results are saved, is ignored by git so that everyone can local experiments.

If you have nice results to share, you may copy the raw or preprocessed results in the toplevel `shared_exdata` directory and push!

```bash
python2 -m bbob_pproc -o latex/final_report/comparison ./contrib/results/IBEAe_C/ ./contrib/results/IBEAe_Python/ ./contrib/results/IBEA_HV_Python/ ./contrib/results/Random_Search-5/
```

## Variation step

Application of crossover and mutation both have tunable probabilities. For most problem functions best results were achieved with both probabilities being close (but not necessarily equal) to 1.

### Recombination
The paper applies Simulated Binary Crossover-20, following paper[2]. 

Lower values of indices for the approximated `spread` distribution produce offspring different from their parents, whereas higher ones produce offspring more similar to their parents. 

In practice, n=20 produces some numerical instability, the authors of [2] recommend using n = {2, 5}.

### Mutation
For fixed variance, prefer a higher value, e.g sigma in [2, 5].

The idea is to adapt the step size depending on the success of previous mutations for each input space dimension.

## Timing
    done in 2.7e-03 seconds/evaluation
  dimension seconds/evaluations
  -----------------------------
      2      2.3e-03 
      5      2.7e-03 
     10      2.8e-03 
     20      2.7e-03 


###### References
[comment]: # (BIBLIOGRAPHY STYLE: MLA)

1. Eckart Zitzler and Simon Künzli, “Indicator-Based Selection in Multiobjective Search”. In Parallel Problem Solving from Nature (PPSN 2004), pp. 832-842, 2004.
2. Deb, Kalyanmoy, and Ram B. Agrawal. "Simulated binary crossover for continuous search space." Complex Systems 9.3 (1994): 1-15.
3. [ref 14] L. Thiele, S. Chakraborty, M. Gries, and S. Künzli. Design space exploration of
network processor architectures. In M. Franklin, P. Crowley, H. Hadimioglu, and
P. Onufryk, editors, Network Processor Design Issues and Practices, Volume 1,
chapter 4, pages 55–90. Morgan Kaufmann, October 2002.
