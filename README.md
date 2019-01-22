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

usage: main.py [-h] [-d {cars,iris}]
               {clean,knn,svm,ann,dt,boosting} ...

optional arguments:
  -h, --help            show this help message and exit
  -d {cars, iris}, --dataset {cars,iris}
                        Which data to analzye

strategies:
  {nearest,vector,neural,tree,boost}
    nearest                 K-Nearest Neighbors
    vector                  Support Vector Machines
    neural                  Neural Networks
    tree                    Decision Tree
    boost                   Boosting

```


## Example

To run the car problem with a decision tree

```
python main.py -d cars tree
```