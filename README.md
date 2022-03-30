# A Two-phase Framework with a Bezier Simplex-based Interpolation Method for Computationally Expensive Multi-objective Optimization

This repository provides the code to reproduce results shown in the following paper.

> Ryoji Tanabe, Youhei Akimoto, Ken Kobayashi, Hiroshi Umeki, Shinichi Shirakawa, Naoki Hamada, **A Two-phase Framework with a Bezier Simplex-based Interpolation Method for Computationally Expensive Multi-objective Optimization**, accepted for [GECCO2022](https://gecco-2022.sigevo.org), [pdf](https://arxiv.org/abs/2203.15292)

The code highly depends on the COCO software:

> Nikolaus Hansen, Anne Auger, Raymond Ros, Olaf Mersmann, Tea Tusar, and Dimo Brockhoff, **COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting**, Optimization Methods and Software, 36(1): 114-144 (2021), [link](https://arxiv.org/abs/1603.08785)


# Requirements

This code require Python (=>3.8), numpy, [cocoex](https://github.com/numbbo/coco), [pygmo](https://esa.github.io/pygmo2/index.html), [pybobyqa](https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/index.html#), click, torch, and [torch\_bsf](https://gitlab.com/hmkz/pytorch-bsf). 

This repository also provides code to run optimizers implemented in [optuna](https://github.com/optuna/optuna) and [PlatEMO](https://github.com/BIMK/PlatEMO) on the bi-objective BBOB problems. Each software is optionally needed to run the code.

``gpus=0`` in the following line in tpb.py can cause an error in pytorch-bsf. The argument ``auto_select_gpus=True`` in line 215 in [bezier\_simplex.py](https://gitlab.com/hmkz/pytorch-bsf/-/blob/master/torch_bsf/bezier_simplex.py) should be ``auto_select_gpus=False`` when the option ``gpus`` is set to 0.

```
bs = torch_bsf.fit(params=ts, values=xs, degree=bez_degree, max_epochs=100, gpus=0)
```

# Usage

The following command runs a TPB with a default parameter setting on the 55 bi-objective BBOB problems with 2, 3, 5, 10, and 20 objectives:

```
$ python tpb.py
```
The parameter setting can be given to the code of TPB as follows:

```
$ python tpb.py --budget_multiplier 20 --n_weights 3 --opt_budget_rate 0.9 --optimizer bobyqa --bez_degree 2 --scalarizing_fun ws --interpolator bez
```

# Usage of other code

## MOTPE in optuna

Simply, the following command runs MOTPE on the 55 bi-objective TPB problems:

```
$ python optuna_bbob.py
```

## EMO algorithms in PlatEMO

First, you need to copy all Matlab files in [the example](https://github.com/numbbo/coco/tree/master/code-experiments/build/matlab) to your PlatEMO code in [this directory](https://github.com/BIMK/PlatEMO/tree/master/PlatEMO). Then, you need to copy ``dummyFun.m``, ``dummyFun1.m``, ``dummyFun2.m``, and ``platemo_bbob.py`` to the same directory. The use of the PlatEMO code in the COCO software require a trick as in ``dummyFun.m``, ``dummyFun1.m``, and ``dummyFun2.m``. The following command runs K-RVEA on the 55 bi-objective TPB problems. Of course, you can run the code on the GUI environment.

```
$ matlab -batch platemo_bbob.m
```
