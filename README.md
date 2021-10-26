# Doubly Robust Thompson Sampling with Linear Payoffs

Python3 based implementation of the paper "Doubly Robust Thompson Sampling with linear payoffs".
In this repository, you can generate Figures of the paper!

## Directory tree

```bash
.
|-- experiment.py
|-- plot.py
|-- algorithms.py
|-- figure1.sh
|-- requirements.txt
|-- README.md
```

- `algorithms.py` contains the Thompson Sampling (TS), Balanced Linear Thompson Sampling (BLTS), and the proposed Doubly Robust Thompson Sampling (DRTS).
- `experiment.py` contains the simulation environments and evaluation of cumulative regrets and estimation error.
- `plot.py` plots the results generated by `experiment.py`
- `figure1.sh` contains a quick start that reproduces the result in the paper.
- `requirements.txt` contains the dependencies to run the codes.


## Requirements
- python 3
- numpy
- scipy
- sobol_seq
- matplotlib
- tqdm


## Quick start

First to install the dependencies,
```bash
pip install -r requirements.txt
```

To generate Figure 1 in the paper simply run,

```bash
sh figure1.sh
```
This code will generate the cumulative regrets and estimation error plots of TS, BLTS, and DRTS,  when d=10, 30, and N=3, 10.

If you want to change the settings, use
```bash
python experiment.py -d 5 -N 7 -seed 1324 -T 10000
```
to evaluate performances of the three algorithms for T=10000 rounds when d=5, and N=7, with seed 1324.

After running `experiment.py`, the estimation error and cumulative regrets are saved in `.txt` format.
Then run
```bash
python plot.py -d 5 -N 7
```
to plot the results of d=5, and N=7.


## Example results
We introduce our example results in our paper.

Figure 1. Comparison of cumulative regrets with best hyperparameters.

<img src="./examples/regrets_d20_N10.png" alt="drawing" width="300"/>
<img src="./examples/regrets_d30_N10.png" alt="drawing" width="300"/>

<img src="./examples/regrets_d20_N20.png" alt="drawing" width="300"/>
<img src="./examples/regrets_d30_N20.png" alt="drawing" width="300"/>

Figure 2. Comparison of estimation error with best hyperparameters

<img src="./examples/beta_err_d20_N10.png" alt="drawing" width="300"/>
<img src="./examples/beta_err_d30_N10.png" alt="drawing" width="300"/>
