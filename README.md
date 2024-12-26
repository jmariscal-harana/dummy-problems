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


## Visualisation
Visualise training/validation results by running:
```
tensorboard --logdir=lightning_logs/
```

# Example 1: generating a synthetic dataset and benchmarking different classifiers.
## Dataset generator
Run `notebooks/synthetic_data_generation.ipynb` to generate the "letters" dataset, a dataset of uppercase grayscale images.

Parameters can be easily modified to change the font, the size, the thickness, or the colour; and to increase/reduce the randomness of the dataset. The default settings produce the following split:
- Train: 8 images per letter = 208 images
- Validate: 2 images per letter = 52 images
- Test: 2 images per letter = 52 images

## Benchmarking classifiers
Run `notebooks/classifier_benchmark.ipynb` to train or test different classifiers on the "letters" dataset.

Currently, three different ML models can be benchmarked: a Support Vector Machine (SVM), a Convolutional Neural Network (CNN), and a state-of-the-art Transformer:
- A SVM (classical ML model) has been chosen based on a model comparison for the MNIST dataset (https://yann.lecun.com/exdb/mnist/).
- A simple, custom CNN has been chosen as a tradeoff between model size and performance (SVM vs Transformer).
- A state-of-the-art Transformer (TinyViT) has been chosen based on its avg_top1 score on the "timm" leaderboard (https://huggingface.co/spaces/timm/leaderboard). A tiny model has been chosen based on available compute and original train split size (208 images).

Results
samples per letter (train) = 10
CONVNET
train acc 0.817307710647583
test acc 0.03846153989434242

samples per letter (train) = 100
CONVNET
train acc 0.9437500238418579
test acc 0.4326923191547394

samples per letter (train) = 1000
CONVNET
train acc 0.9970192313194275 
test acc 0.9790384769439697

## Reproducing results
Due to the randomness of the data generator, approximate results can be obtained by running `notebooks/classifier_benchmark.ipynb`.
