#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = (
    'IMAGENET_CROP_SIZE',
    'IMAGENET_RESIZE',
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    'DataLoaderWrapper',
    'ModelWrapper',
)

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import torch
from torch import nn, optim
from torchvision import datasets, models, transforms


IMAGENET_CROP_SIZE = 224
"""Crop size for ImageNet images"""
IMAGENET_RESIZE = 256
"""Resize size prior to cropping for ImageNet images"""
IMAGENET_MEAN = (0.485, 0.456, 0.406)
"""Mean for normalizing ImageNet images"""
IMAGENET_STD = (0.229, 0.224, 0.225)
"""Standard deviation for normalizing ImageNet images"""


logger = logging.getLogger(__name__)


class DataLoaderWrapper:
    """Wrapper for PyTorch DataLoader"""

    def __init__(self, directory: Path, batch_size: int = 32):
        """Initialize the DataLoader Wrapper

        Args:
            directory: Path to the data directory containing 'train', 'valid', and 'test' subdirectories
            batch_size: Number of samples per batch (default: 32)
        """
        self._directory = directory

        data_transforms = {
            'train': transforms.Compose([
                        transforms.RandomRotation(degrees=45),
                        transforms.RandomResizedCrop(IMAGENET_CROP_SIZE),
                        transforms.RandomHorizontalFlip(p=0.25),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ]),
            'test': transforms.Compose([
                        transforms.Resize(IMAGENET_RESIZE),
                        transforms.CenterCrop(IMAGENET_CROP_SIZE),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ]),
            'valid': transforms.Compose([
                        transforms.Resize(IMAGENET_RESIZE),
                        transforms.CenterCrop(IMAGENET_CROP_SIZE),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ]),
        }

        # Load the datasets with ImageFolder
        image_datasets = {subset: datasets.ImageFolder(directory / subset, transform=data_transforms[subset])
                          for subset in data_transforms
                          }

        # Using the image datasets and the trainforms, define the dataloaders
        dataloaders = {subset: torch.utils.data.DataLoader(image_datasets[subset],
                                                           batch_size=batch_size,
                                                           shuffle=(subset == 'train')
                                                           )
                       for subset in data_transforms
                       }

        # store as instance variables
        self._image_datasets = image_datasets
        self._dataloaders = dataloaders

    @property
    def test(self) -> torch.utils.data.DataLoader:
        """DataLoader for the test set"""
        return self._dataloaders['test']

    @property
    def train(self) -> torch.utils.data.DataLoader:
        """DataLoader for the training set"""
        return self._dataloaders['train']

    @property
    def valid(self) -> torch.utils.data.DataLoader:
        """DataLoader for the validation set"""
        return self._dataloaders['valid']

    @property
    def classes(self) -> list[str]:
        """List of class labels in the dataset"""
        return self._image_datasets['train'].classes

    @property
    def class_to_idx(self) -> dict[str, int]:
        """Mapping from class labels to indices"""
        return self._image_datasets['train'].class_to_idx


class ModelWrapper:
    """Wrapper for PyTorch models that includes class-to-index mapping and category names"""

    def __init__(self, arch: Literal['vgg16', 'resnet50', 'densenet121'], out_features: int, hidden_units: int = 4096,
                 dropout: float = 0.33, weights: str | None = 'DEFAULT', class_to_idx: dict[str, str] | None = None,
                 device: torch.device = torch.device('cpu')):
        """Initialize the Model Wrapper

        Args:
            arch:         Model architecture to use (vgg16, resnet50, or densenet121)
            out_features: Number of output features (classes) for the classifier
            hidden_units: Number of nodes in the hidden layer of the classifier (default: 4096)
            dropout:      Dropout probability for the classifier (default: 0.33)
            weights:      Pre-trained weights to use for the model (see torchvision.models documentation
                          for details; default: 'DEFAULT')
            class_to_idx: Optional mapping from class labels to indices (if not provided, will be set to empty dict)
            device:       Device to use for the model (default: CPU)
        """

        match (arch):
            case 'vgg16':
                model = models.vgg16(weights=weights)
                in_features = model.classifier[0].in_features
            case 'resnet50':
                model = models.resnet50(weights=weights)
                in_features = model.fc.in_features
            case 'densenet121':
                model = models.densenet121(weights=weights)
                in_features = model.classifier.in_features
            case _:
                raise ValueError(f'Unsupported architecture: {arch}')

        # freeze feature weights
        for param in model.parameters():
            param.requires_grad = False

        # freeze feature weights, replace classifier
        self._model = ModelWrapper._replace_classifier(model, in_features, hidden_units, out_features,
                                                       dropout=dropout)
        # move to device
        self._model.to(device)

        # store important parameters
        self._arch = arch
        self._epochs = None
        # class to index map
        self._class_to_idx = class_to_idx or dict()
        self._idx_to_class = None

        # store device
        self._device = device

    @property
    def class_to_idx(self) -> dict[str, int]:
        """Mapping from class labels to indices"""
        return self._class_to_idx

    @class_to_idx.setter
    def class_to_idx(self, value: dict[str, int]):
        """Set the class-to-index mapping and reset the index-to-class mapping"""
        self._class_to_idx = value
        self._idx_to_class = None

    @property
    def idx_to_class(self) -> dict[int, str]:
        """Mapping from indices to class labels (inverse of class_to_idx)"""
        if self._idx_to_class is None:
            self._idx_to_class = {idx: cls for cls, idx in self._class_to_idx.items()}
        return self._idx_to_class

    @staticmethod
    def _replace_classifier(model: torch.nn.Module, in_features: int, hidden_units: int,
                            out_features: int, dropout: float = 0.33) -> torch.nn.Module:
        """Replace the classifier of a pre-trained model with a new one

        Args:
            model: Pre-trained model to modify
            in_features: Number of input features for the new classifier
            hidden_units: Number of nodes in the hidden layer of the new classifier
            out_features: Number of output features (classes) for the new classifier

        Returns:
            Modified model with the new classifier
        """
        # build our custom classifier
        classifier = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(in_features, hidden_units)),
            ('relu', torch.nn.ReLU()),
            ('dropout', torch.nn.Dropout(p=dropout)),
            ('fc2', torch.nn.Linear(hidden_units, out_features)),
            ('output', torch.nn.LogSoftmax(dim=1))
        ]))

        # replace existing classifier
        if hasattr(model, 'fc'):
            model.fc = classifier
        else:
            model.classifier = classifier

        # return model for convenience
        return model

    @classmethod
    def from_checkpoint(cls, filename: Path, device: torch.device = torch.device('cpu')):
        """Load model from checkpoint file

        Args:
            filename: Path to the checkpoint file

        Returns:
            ModelWrapper instance containing the loaded model, class-to-index mapping, and category names
        """
        checkpoint = torch.load(filename, map_location=device)

        if (not isinstance(checkpoint, dict)):
            raise ValueError(f'Invalid checkpoint file: {filename} (must be dict[str, Any])')

        mandatory_keys = ('arch', 'state', 'class_to_idx', 'out_features')
        missing_keys = tuple(k for k in mandatory_keys if k not in checkpoint)
        if missing_keys:
            raise ValueError(f'Invalid checkpoint file: {filename} (missing keys: {", ".join(missing_keys)})')

        arch = checkpoint['arch']
        epochs = checkpoint.get('epochs', 'N/A')

        # small log message
        logger.info(f'Loaded model checkpoint {filename} (arch={arch}, epochs={epochs})')

        # load rest of arguments
        state = checkpoint['state']
        class_to_idx = checkpoint['class_to_idx']
        out_features = checkpoint['out_features']
        kwargs = {k: checkpoint[k] for k in ('hidden_units', 'dropout') if k in checkpoint}

        # create wrapper, load initial instances w/o weights to speed things up)
        result = cls(arch, out_features, device=device, weights=None, class_to_idx=class_to_idx, **kwargs)
        # restore state dict (including feature weights)
        result._model.load_state_dict(state)
        # default to inference mode after loading from checkpoint
        result._model.eval()

        return result

    def to_checkpoint(self, filename: Path):
        """Save model to checkpoint file

        Args:
            filename: Path to the checkpoint file to save to
        """

        # classifier attribute depending on VGG/DenseNet vs ResNet
        classifier = self._model.classifier if (hasattr(self._model, 'classifier')) else self._model.fc

        checkpoint = {
            'arch': self._arch,
            'state': self._model.state_dict(),
            'class_to_idx': self._class_to_idx,
            'hidden_units': classifier.fc2.in_features,
            'out_features': classifier.fc2.out_features,
            'dropout': classifier.dropout.p,
        }

        # add optional arguments if they exist
        if (self._epochs is not None):
            checkpoint['epochs'] = self._epochs

        # save to file
        torch.save(checkpoint, filename)

    def _calc_forward_loss(self, criterion: nn.NLLLoss, images: torch.Tensor,
                           labels: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Calculate Forward Pass Loss

        Args:
            criterion: Loss function to use for calculating the loss
            images: Batch of input images
            labels: Batch of corresponding labels

        Returns:
            Tuple containing the loss and accuracy for the batch
        """
        log_ps = self._model.forward(images)
        # loss w.r.t. original label
        loss : torch.Tensor = criterion(log_ps, labels)
        top_labels = log_ps.argmax(dim=1)
        accuracy = (labels == top_labels).float().mean()
        return loss, accuracy.item()

    def train(self, dataloader: DataLoaderWrapper, epochs: int, learning_rate: float) -> None:
        """Train the model

        Args:
            dataloader: DataLoaderWrapper containing the training/validation data
            epochs: Number of epochs to train for
            learning_rate: Learning rate to use for training
        """
        # setup the loss and optimizer
        criterion = nn.NLLLoss()
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)

        # pre-calculate number of batches for status messages
        num_train_batches = len(dataloader.train)
        num_valid_batches = len(dataloader.valid)

        # training loop
        for epoch in range(epochs):

            # switch to train mode
            self._model.train()

            # accumulate training loss across batches
            train_loss = 0.0
            # accumulate training accuracy across batches
            train_accuracy = 0.0

            # go through training data in batches
            for batch, (inputs, labels) in enumerate(dataloader.train, 1):

                # move to GPU if available
                inputs, labels = inputs.to(self._device), labels.to(self._device)

                # zero-out the accumulated gradients
                optimizer.zero_grad()

                # perform a forward pass
                loss, accuracy = self._calc_forward_loss(criterion, inputs, labels)

                # perform backward pass
                loss.backward()
                optimizer.step()

                # status message every 10 batches
                if (1 == batch or 0 == batch % 10):
                    logger.info(' | '.join((
                        f'Epoch {epoch+1}/{epochs}',
                         'Training',
                        f'Batch {batch:3d}/{num_train_batches}',
                        f'Loss: {loss.item():.4f}',
                        f'Accuracy: {accuracy:.4f}',
                    )))

                # accumulate train loss
                train_loss += loss.item()
                # accumulate train accuracy
                train_accuracy += accuracy

            else:
                # switch to evaluation mode
                self._model.eval()

                # accumulate validation loss across batches
                valid_loss = 0.0
                # accumulate validation accuracies across batches
                valid_accuracy = 0.0

                # go through validation data in batches
                for batch, (input, labels) in enumerate(dataloader.valid, 1):

                    # move to GPU if available
                    input, labels = input.to(self._device), labels.to(self._device)

                    with torch.no_grad():
                        loss, accuracy = self._calc_forward_loss(criterion, input, labels)

                        # status message every 10 batches
                        if (1 == batch or 0 == batch % 10):
                            logger.info(' | '.join((
                                f'Epoch {epoch+1}/{epochs}',
                                 'Validation',
                                f'Batch {batch:2d}/{num_valid_batches}',
                                f'Loss: {loss.item():.4f}',
                                f'Accuracy: {accuracy:.4f}',
                            )))

                        # accumulate validation loss
                        valid_loss += loss.item()
                        # accumulate validation accuracy
                        valid_accuracy += accuracy

            logger.info(' | '.join((
                f'Epoch {epoch+1}/{epochs}',
                 'Complete',
                f'Training Loss: {train_loss/num_train_batches:.4f}',
                f'Training Accuracy: {train_accuracy/num_train_batches:.4f}',
                f'Validation Loss: {valid_loss/num_valid_batches:.4f}',
                f'Validation Accuracy: {valid_accuracy/num_valid_batches:.4f}',
                )))

        # store important parameters
        self._epochs = epochs
        self.class_to_idx = dataloader.class_to_idx

    def test(self, dataloader: DataLoaderWrapper):
        """Test the model on the test set

        Args:
            dataloader: DataLoaderWrapper containing the test data

        Returns:
            Tuple containing the average loss and accuracy on the test set
        """
        # pre-calculate number of batches for status messages
        num_test_batches = len(dataloader.test)

        # setup the loss
        criterion = nn.NLLLoss()

        # switch to evaluation mode
        self._model.eval()

        # accumulate test accuracies across batches
        test_accuracy = 0.0

        # go through test data in batches
        for batch, (input, labels) in enumerate(dataloader.test, 1):

            # move to GPU if available
            input, labels = input.to(self._device), labels.to(self._device)

            with torch.no_grad():
                loss, accuracy = self._calc_forward_loss(criterion, input, labels)

                # status message every 10 batches
                if (1 == batch or 0 == batch % 10):
                    logger.info(' | '.join((
                        f'Testing',
                        f'Batch {batch:2d}/{num_test_batches}',
                        f'Loss: {loss.item():.4f}',
                        f'Accuracy: {accuracy:.4f}'
                    )))

                # accumulate test accuracy
                test_accuracy += accuracy

        logger.info(f'Overall Test Accuracy: {test_accuracy / num_test_batches:.4f}')

    def predict(self, image: torch.Tensor, top_k: int = 5) -> tuple[list[float], list[str]]:
        """Predict the top-k Classes for a single Image

        Args:
            image: Image tensor to predict (should be preprocessed and normalized)
            top_k: Number of top classes to return (default: 5)

        Returns:
            Tuple containing the top-k probabilities and corresponding class indices
        """

        # switch to evaluation mode
        self._model.eval()

        # disable autograd for inference
        with torch.no_grad():
            # change dimensions (C, H, W) -> (1, C, H, W) and move to device
            log_ps = self._model.forward(image.unsqueeze(0).to(self._device))
            # compute probs
            ps = torch.exp(log_ps)
            # top-k classes and their probabilities
            probs, classes = ps.topk(k=top_k, dim=1)
        
        # map class indices, return simple lists
        probs = probs.squeeze(dim=0).tolist()
        classes = classes.squeeze(dim=0).tolist()
        return probs, [self.idx_to_class[idx] for idx in classes]
