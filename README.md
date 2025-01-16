# dummy-problems
Solving "simple" problems using machine learning to improve my understanding (and have some fun!).

# Installation
## Docker (recommended)
To install the required dependencies and run the code, use a Docker container and attach to it using Visual Studio Code:
```
cd docker
docker compose up
```

> [!NOTE]  
> If you have access to a GPU, uncomment the line `# runtime: nvidia` from the `docker/docker-compose.yaml` file.

## Pip (experimental)
To install as a Python package, run:
```
pip install dummy-problems
```

> [!NOTE]  
> Published following: https://www.digitalocean.com/community/tutorials/how-to-publish-python-packages-to-pypi-using-poetry-on-ubuntu-22-04

# Project structure
I am mostly following the structure from https://python-poetry.org/docs/basic-usage/#project-setup and https://github.com/Lightning-AI/deep-learning-project-template.

There are three key elements: **dataloaders**, **models**, and **visualisation**.

## Dataloaders
Includes the datasets (`LettersDataset` and `PetsDataset`) and dataloaders (`LettersDataModule` and `PetsDataModule`) for loading train/val/test data.

## Models
Includes three different ML models: a Support Vector Machine (SVM), a Convolutional Neural Network (CNN), and a state-of-the-art Transformer:
- A SVM (classical ML model) has been chosen based on a model comparison for the MNIST dataset (https://yann.lecun.com/exdb/mnist/).
- A simple, custom CNN has been chosen as a tradeoff between model size and performance (SVM vs Transformer).
- A state-of-the-art Transformer (TinyViT) has been chosen based on its avg_top1 score on the "timm" leaderboard (https://huggingface.co/spaces/timm/leaderboard). A tiny model has been chosen based on available compute and dataset sizes (<1000 images).

## Visualisation
Visualise training/validation results by running:
```
tensorboard --logdir=lightning_logs/
```

# Example 1: generating a synthetic dataset and benchmarking different classifiers
## Synthetic data generation
Run `notebooks/synthetic_data_generation.ipynb` to generate the "letters" dataset, a dataset of images of grayscale letters. The current settings produce a dataset of 128x128 images of uppercase grayscale letters with the following data split:
- Train: 8 images per letter = 208 images
- Validate: 2 images per letter = 52 images
- Test: 2 images per letter = 52 images

The last cell allows you to visualise a few random examples from the train and test sets. Parameters can be easily modified to change the font, the size, the thickness, or the colour; and to increase/reduce the randomness of the dataset. 

## Classifier benchmark
Run `notebooks/classifier_benchmark.ipynb` to train or test different classifiers on the "letters" dataset.

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

For SVM models, when training on smaller images (e.g. 32x32) and larger datasets (e.g. 1000), the testing accuracy improves drastically. An issue with the original image size is that images are flattened into very high-dimensional (but sparse) vectors (i.e. 128*128 = 16384 dimensions), which can lead to poor SVM performance due to the curse of dimensionality.

The CNN model struggles with smaller dataset sizes, but performs well for the largest dataset. The TinyViT model performs well even for smaller datasets. This difference probably stems from the differences in their initial weights: whereas the CNN is randomly initialised and trained from scratch, TinyViT was pre-trained on ImageNet, so it had already learnt meaningful image features. 

### Reproducing results
Due to the randomness of the data generator, approximate results can be obtained by running `notebooks/synthetic_data_generation.ipynb` to generate a test dataset and `notebooks/classifier_benchmark.ipynb` to test the models. Note that the second notebook has been generated to facilitate the process of reproducing the results. The original code can be found in `dummy_problems/models/core.py`. The confusion matrix and ROC figures are both displayed and saved to `notebooks/`.

> [!NOTE]
> Checkpoints can be downloaded from [Google Drive](https://drive.google.com/drive/folders/12ps_EMCZIZQETBm3mvWmZ7_Y4iw_uRms). To avoid code modifications, save the checkpoints under a newly created `weights/` folder.

# Example 2: train a model to differentiate between different pets
## Dataset structure
```
Chinchilla/
Hamster/
Rabbit/
train_set.txt
test_set.txt
```

## Pet classifier
Based on the result from "Example 1", a TinyViT model has been re-trained with Early Stopping on the above dataset, achieving 97.8% accuracy (weighted average) on the test set. Run `notebooks/classifier_pets.ipynb` to train or test on the "pets" dataset.

### Results
| PET       	    | PRECISION 	| RECALL 	        | F1-SCORE       	| SUPPORT   |
| -:	            | :-:       	| :-:           	| :-:           	| -:        |
| **Chinchilla**    | 1.000         |       0.889   	|      0.941      	| 9         |
| **Hamster** 	    | 1.000         |       1.000     	|      1.000      	| 6         |
| **Rabbit** 	    | 0.969         |       1.000    	|      0.984    	| 31        |
| **accuracy**      |               |                   |      0.978        | 46        |
| **macro avg**     | 0.990         |       0.963       |      0.975        | 46        |
| **weighted avg**  | 0.979         |       0.978       |      0.978        | 46        |
  
### Reproducing results
Run `notebooks/classifier_pets.ipynb` to reproduce the results. The confusion matrix and ROC figures are both displayed and saved to `notebooks/`.

> [!NOTE]
> The `pets_baseline.ckpt` checkpoint can be downloaded from [Google Drive](https://drive.google.com/drive/folders/12ps_EMCZIZQETBm3mvWmZ7_Y4iw_uRms). To avoid code modifications, save the checkpoint under a newly created `weights/` folder.

# Example 3: train a DQN to solve the cart pole problem
Run `notebooks/dqn_cartpole.ipynb` to train a Deep Q-Network from scratch. This notebook is based on code from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html and https://github.com/pythonlessons/CartPole_reinforcement_learning.

> [!NOTE]
> To skip training and test directly, download the `model_500_steps.pth` weights from [Google Drive](https://drive.google.com/drive/folders/12ps_EMCZIZQETBm3mvWmZ7_Y4iw_uRms). To avoid code modifications, save the checkpoint under a newly created `weights/` folder.
