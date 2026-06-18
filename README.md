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

## License

Original files Copyright 2018 Udacity, Inc.
My additions to documentation and code are [MIT](https://spdx.org/licenses/MIT).
See [LICENSE-Udacity](LICENSE-Udacity) resp. [LICENSE](LICENSE).
