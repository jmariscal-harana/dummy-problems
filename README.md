# dummy-problems
Solving "simple" problems using machine learning to improve my understanding (and have some fun!).

# Installation
## Docker (recommended)
To install the required dependencies and run the code, use a Docker container and attach to it using Visual Studio Code:
```
cd docker
docker compose up
```

## Pip (experimental)
To install as a Python package, run:
```
pip install dummy-problems
```

Note: published following: https://www.digitalocean.com/community/tutorials/how-to-publish-python-packages-to-pypi-using-poetry-on-ubuntu-22-04

# Project structure
I am mostly following the structure from https://python-poetry.org/docs/basic-usage/#project-setup and https://github.com/Lightning-AI/deep-learning-project-template.

There are three key elements: **dataloaders**, **models**, and **visualisation**.

## Dataloaders


## Models


## Evaluation 


## Visualisation
Visualise training/validation results by running:
```
tensorboard --logdir=lightning_logs/
```

# Example 1: generating a synthetic dataset and benchmarking different classifiers.
- Run `notebooks/synthetic_data_generation.ipynb` to generate the "letters" dataset, a dataset of uppercase grayscale images.
- Run `notebooks/classifier_benchmark.ipynb` to train or test different classifiers on the "letters" dataset. 

SVM (best parameters): {'svm__C': 0.001, 'svm__gamma': 0.01, 'svm__kernel': 'rbf'} 
