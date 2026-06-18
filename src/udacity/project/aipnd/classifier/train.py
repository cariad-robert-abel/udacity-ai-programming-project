#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import sysconfig

from pathlib import Path

import torch

from udacity.project.aipnd.classifier.model import DataLoaderWrapper, ModelWrapper


logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__package__)


def main() -> int:

    parser = argparse.ArgumentParser(description='Train Flower Image Classifier')
    parser.add_argument('--loglevel', type=str, choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
                        default='INFO', help='Log Level')
    parser.add_argument('data_directory', metavar='data-directory', type=Path, help='Path to the data directory')
    parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU for inference')
    parser.add_argument('--model-directory', type=Path, default=Path('models'), help='Path to the model output directory')
    parser.add_argument('--arch', type=str, default='vgg16', choices=('vgg16', 'resnet50', 'densenet121'), help='Model architecture')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--hidden-units', type=int, default=4096, help='Number of nodes in the hidden layer of the classifier')

    # parse command-line arguments
    args = parser.parse_args()

    # modify log level
    logger.setLevel(args.loglevel)

    # validate arguments
    if (not args.data_directory.is_dir() or not args.data_directory.exists()):
        logger.error('Invalid data directory: %s', args.data_directory)
        return 1

    for split in ('train', 'valid', 'test'):
        if (not (args.data_directory / split).is_dir() or not (args.data_directory / split).exists()):
            logger.error('Invalid data directory: %s (missing subdirectory: %s)', args.data_directory, split)
            return 1

    if (args.model_directory.exists() and not args.model_directory.is_dir()):
        logger.error('Invalid model directory: %s', args.model_directory)
        return 1

    if (not args.epochs >= 1):
        logger.error('Invalid value for --epochs: %d (must be >= 1)', args.epochs)
        return 1

    if (not args.learning_rate > 0.0 or args.learning_rate > 1e-2):
        logger.error('Invalid value for --learning-rate: %f (must be > 0.0 and <= 0.01)', args.learning_rate)
        return 1

    if (not args.hidden_units >= 1 or not args.hidden_units <= 32768):
        logger.error('Invalid value for --hidden-units: %d (must be >= 1 and <= 32768)', args.hidden_units)
        return 1

    # mirror command-line arguments
    logger.info(f'Training Model arch={args.arch} hidden_units={args.hidden_units} epochs={args.epochs} learning_rate={args.learning_rate}')
    logger.info(f'Using data directory: {args.data_directory}')
    logger.info(f'Using model output directory: {args.model_directory}')

    try:

        # determine device
        device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')
        if (not torch.cuda.is_available() and args.gpu):
            logger.warning('GPU not available, falling back to CPU')
        logger.info(f'Using device: {device}')

        # create dataloader
        dataloader = DataLoaderWrapper(args.data_directory)

        # create model
        model = ModelWrapper(args.arch, len(dataloader.classes), args.hidden_units, device=device)

        # train model
        model.train(dataloader, epochs=args.epochs, learning_rate=args.learning_rate)

        # test model
        model.test(dataloader)

        # create output directory
        args.model_directory.mkdir(parents=True, exist_ok=True)

        model.to_checkpoint(args.model_directory / f'{args.arch}-{args.hidden_units}-{args.epochs}-{args.learning_rate}.pth')

    except Exception as e:
        logger.error(f'An error occurred during training: {e}')
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
