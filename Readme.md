# Coco documentation

[Biobjective Performance Assessment with the COCO Platform](http://numbbo.github.io/coco-doc/bbob-biobj/perf-assessment/)

# Implementation of the IBEA-epsilon algorithm [1]

## Running the optimizer
There are two ways to run the optimizer:

```shell
python2 ibea.py
```

```shell
python2 experiment.py
```

Note that it is easier to change the experiment parameters in the `IBEA.py` file, by parameterizing the call to `experiment.main`.

## Convention: Separate parameters
The idea is to keep the parameters we are testing local.
The best way is to keep the solver definition in `experiment.py` as: 

```python
SOLVER = IBEA().ibea # line 254
``` 

You may change the parameters when initializing the object, however when you commit some changes, please do:

```shell
git add -p experiment.py
```

so as not to write your params on git (and risk overwriting our own when we merge).

## Convention: Shared experimental results
The idea is that the local sub-directory in which results are saved, namely `exdata`, is git ignored so that everyone can local experiments.

If you have nice results to share, you may copy the raw or preprocessed results in the toplevel `shared_exdata` directory and push!

## Variation step

### Recombination operator
The paper applies SBX-20, which stands for Simulated Binary Crossover [2].

### Mutation


###### References
[comment]: # (BIBLIOGRAPHY STYLE: MLA)

1. Eckart Zitzler and Simon Künzli, “Indicator-Based Selection in Multiobjective Search”. In Parallel Problem Solving from Nature (PPSN 2004), pp. 832-842, 2004.
2. Deb, Kalyanmoy, and Ram B. Agrawal. "Simulated binary crossover for continuous search space." Complex Systems 9.3 (1994): 1-15.
3. [ref 14] L. Thiele, S. Chakraborty, M. Gries, and S. Künzli. Design space exploration of
network processor architectures. In M. Franklin, P. Crowley, H. Hadimioglu, and
P. Onufryk, editors, Network Processor Design Issues and Practices, Volume 1,
chapter 4, pages 55–90. Morgan Kaufmann, October 2002.
