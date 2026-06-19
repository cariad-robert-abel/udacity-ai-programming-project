#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import json
import sys
import sysconfig

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from udacity.project.aipnd.classifier.model import IMAGENET_CROP_SIZE, IMAGENET_RESIZE, IMAGENET_MEAN, IMAGENET_STD
from udacity.project.aipnd.classifier.model import ModelWrapper


logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[%(asctime)s][%(levelname)-8s] %(message)s', datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger(__package__)


def process_image(image: Image.Image) -> torch.Tensor:
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a PyTorch tensor
    '''
    # resize using PIL
    w, h = image.size
    new_w = IMAGENET_RESIZE if w < h else int(round(w * IMAGENET_RESIZE / h))
    new_h = IMAGENET_RESIZE if h < w else int(round(h * IMAGENET_RESIZE / w))
    image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

    # center crop
    left = (new_w - IMAGENET_CROP_SIZE) // 2
    top =  (new_h - IMAGENET_CROP_SIZE) // 2
    right = left + IMAGENET_CROP_SIZE
    bottom = top + IMAGENET_CROP_SIZE
    image = image.crop((left, top, right, bottom))

    # convert PIL [0, 255] to numpy [0.0, 1.0]
    np_image = np.array(image, dtype=np.float32)
    np_image /= 255.0

    # normalize
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    np_image = (np_image - mean) / std

    # return transposed (H, W, C) -> (C, H, W)
    return torch.Tensor(np_image.transpose((2, 0, 1)))


def main() -> int:

    parser = argparse.ArgumentParser(description='Classify Flower Image')
    parser.add_argument('--loglevel', type=str, choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
                        default='INFO', help='Log Level')
    parser.add_argument('image', type=Path, help='Path to the image file')
    parser.add_argument('model', type=Path, help='Path to the model file')
    parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU for inference')
    parser.add_argument('--top-k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category-names', type=Path, default=Path(sysconfig.get_path('data')) / 'cat_to_name.json', help='Path to category names file')

    # parse command-line arguments
    args = parser.parse_args()

    # modify log level
    logger.setLevel(args.loglevel)

    # validate arguments
    if (not args.top_k >= 1):
        logger.error('Invalid value for --top-k: %d (must be >= 1)', args.top_k)
        return 1

    if (not args.image.is_file() or not args.image.exists()):
        logger.error('Invalid image file: %s', args.image)
        return 1

    if (not args.model.is_file() or not args.model.exists()):
        logger.error('Invalid model file: %s', args.model)
        return 1

    if (not args.category_names.is_file() or not args.category_names.exists()):
        logger.error('Invalid category names file: %s', args.category_names)
        return 1

    # load category names
    with args.category_names.open('r', encoding='utf-8') as f:
        cat_to_name = json.load(f)

    if (not isinstance(cat_to_name, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in cat_to_name.items())):
        logger.error('Invalid category names file: %s (must be a JSON object mapping strings to strings)', args.category_names)
        return 1

    # mirror command-line arguments
    logger.info(f'Predicting top-k={args.top_k} category-names={args.category_names}')
    logger.info(f'Using image file: {args.image}')
    logger.info(f'Using model file: {args.model}')

    try:

        # determine device
        device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')
        if (not torch.cuda.is_available() and args.gpu):
            logger.warning('GPU not available, falling back to CPU')
        logger.info(f'Using device: {device}')

        # create model
        model = ModelWrapper.from_checkpoint(args.model, device=device)

        # load and preprocess image
        image = process_image(Image.open(args.image))

        # run inference for top-k classes
        probs, classes = model.predict(image, top_k=args.top_k)

        # print results
        for propb, cls in zip(probs, classes):
            label = f'{cat_to_name.get(cls, "<unknown>")} ({cls})'
            logger.info(f'{label:32s}: {propb:.4f}')

    except Exception as e:
        logger.error(f'An error occurred during prediction: {e}')
        return 1


    return 0

if __name__ == '__main__':
    sys.exit(main())
