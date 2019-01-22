# tgamble3 - Assignment 1

Implementation for five learning algorithms. They are for:

-   Decision trees with some form of pruning
-   Neural networks
-   Boosting
-   Support Vector Machines
-   *k*-nearest neighbors

Datasets used are
- https://archive.ics.uci.edu/ml/datasets/iris
- https://archive.ics.uci.edu/ml/datasets/car+evaluation

## Setup

### Miniconda

- Follow the instructions at https://conda.io/docs/user-guide/install/index.html
- Use miniconda to install all dependencies through `environment.yml` (instruction assumes conda is in your path)

```
    conda env create -f environment.yml
```

- Activate the environment created by miniconda

```
    source activate cs7641_1
```

## Usage

```
python main.py --help

usage: main.py [-h] [-d {wine,credit_card}]
               {clean,knn,svm,ann,dt,boosting} ...

optional arguments:
  -h, --help            show this help message and exit
  -d {wine,credit_card}, --dataset {wine,credit_card}
                        Which dataset to run on

subcommands:
  {clean,knn,svm,ann,dt,boosting}
    clean               Clean the stats from original to final and show me
                        information
    knn                 Run k-nearest neighbors
    svm                 Run Support Vector Machines
    ann                 Run neural networks
    dt                  Run decision trees
    boosting            Run boosting

```


## Example

To run for example, the car problem with an ANN, use the following command.

```
python main.py -d wine ann
```