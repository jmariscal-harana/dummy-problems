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
## Synthetic data generation
Run `notebooks/synthetic_data_generation.ipynb` to generate the "letters" dataset, a dataset of images of grayscale letters. The current settings produce a dataset of 128x128 images of uppercase grayscale letters with the following data split:
- Train: 8 images per letter = 208 images
- Validate: 2 images per letter = 52 images
- Test: 2 images per letter = 52 images

The last cell allows you to visualise a few random examples from the train and test sets. Parameters can be easily modified to change the font, the size, the thickness, or the colour; and to increase/reduce the randomness of the dataset. 

## Classifier benchmark
Run `notebooks/classifier_benchmark.ipynb` to train or test different classifiers on the "letters" dataset.

Currently, three different ML models can be benchmarked: a Support Vector Machine (SVM), a Convolutional Neural Network (CNN), and a state-of-the-art Transformer:
- A SVM (classical ML model) has been chosen based on a model comparison for the MNIST dataset (https://yann.lecun.com/exdb/mnist/).
- A simple, custom CNN has been chosen as a tradeoff between model size and performance (SVM vs Transformer).
- A state-of-the-art Transformer (TinyViT) has been chosen based on its avg_top1 score on the "timm" leaderboard (https://huggingface.co/spaces/timm/leaderboard). A tiny model has been chosen based on available compute and original train split size (208 images).

### Results
The models have been benchmarked for three dataset sizes (i.e. 10, 100, 1000), where the number represents the number of training + validation samples per letter.


| MODEL       	| DATASET SIZE 	| ACCURACY (TRAIN) 	| ACCURACY (TEST) 	|
|-------------	|--------------	|:-:           	    | :-:           	|
|             	| 10           	|       0.742      	|      0.058      	|
| **SVM**	    | 100          	|       0.961   	|      0.177      	|
|             	| 1000         	|         *      	|        *      	|
|             	| 10           	|       0.817      	|      0.077      	|
| **ConvNet** 	| 100          	|       0.944      	|      0.433      	|
|             	| 1000         	|       0.997      	|      0.979      	|
|             	| 10           	|        1.0     	|      0.962      	|
| **TinyViT** 	| 100          	|       0.997    	|      0.998    	|
|             	| 1000         	|        1.0      	|       1.0      	|

*Not feasible due to computational limitations.

For SVM models, when training on smaller images (e.g. 32x32) and larger datasets (e.g. 1000), the testing accuracy improves drastically. An issue with the original image size is that images are flattened into very high-dimensional but sparse vectors (i.e. 128*128 = 16384 dimensions), which can lead to poor SVM performance due to the curse of dimensionality.

## Reproducing results
Due to the randomness of the data generator, approximate results can be obtained by running `notebooks/synthetic_data_generation.ipynb` to generate a test dataset and `notebooks/classifier_benchmark.ipynb` to test the models. Note that the second notebook has been generated to facilitate the process of reproducing the results. The original code can be found in `dummy_problems/models/core.py`.
