# AI Programming with Python - Create Your Own Image Classifier

This repository contains the project associated with "Create Your Own Image Classifier"
Udacity course, which is part of the "AI Programming with Python" nanodegree.
It's a fork of Udacity's [Starter Kit](https://github.com/udacity/aipnd-project).

This GitHub.com project is located at [cariad-robert-abel/udacity-ai-programming-project](https://github.com/cariad-robert-abel/udacity-ai-programming-project).

## Pre-Requisites

Install this application before using it, for example by running the following
inside a virtual environment:

    pip install .

This will also install the necessary `udacity-project-aipnd-classifier-*` shims.

### Training Data

This project uses the "102 Category Flower Dataset" by Maria-Elena Nilsback and Andrew Zisserman
(see [University of Oxford Visual Geometry Group Page](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/))
as made available by Udacity [here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz).

The dataset is archived using [DVC](https://dvc.org/) and supposed to be extraced into the top-level
`flowers` sub-directory prior to executing the Jupyter notebook resp. train/predic scripts.

## Training

Run training using the `udacity-project-aipnd-classifier-train` command-line utility:

    udacity-project-aipnd-classifier-train [-h] [--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--gpu] [--model-directory MODEL_DIRECTORY] [--arch {vgg16,resnet50,densenet121}] [--epochs EPOCHS] [--learning-rate LEARNING_RATE] [--hidden-units HIDDEN_UNITS] data-directory

where `--gpu` enables GPU acceleration; `--model-directory` is the model output directory; `--arch` selects a classifier
architecture based on `vgg16`, `resnet50`, or `densenet121`; `--epochs` determines the number of training epochs;
`--learning-rate` determines the learning rate (e.g. `4e-4`); and `--hidden-units` determines the number of nodes on the
hidden classification layer. The `data-directory` needs to point to the `flowers` directory.

## Inference

Run inference using the `udacity-project-aipnd-classifier-predict` command-line utility:

    udacity-project-aipnd-classifier-predict [-h] [--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--gpu] [--top-k TOP_K] [--category-names CATEGORY_NAMES] image model

where `--gpu` enables GPU acceleration; `--top-k` selects the *k* top classification results (default: `1`),
`--category-names` points to the category name mapping file (default: installed `cat_to_name.json`);
and `image` points to the image to be classified while `model` points to the model to use for classification.

## License

Original files Copyright 2018 Udacity, Inc.
My additions to documentation and code are [MIT](https://spdx.org/licenses/MIT).
See [LICENSE-Udacity](LICENSE-Udacity) resp. [LICENSE](LICENSE).
