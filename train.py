#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Train the model.

Usage:
  train.py <dataset> [<output>] [--ckpt=<ckpt>]

Options:
  -h --help     Show this help.
  <dataset>     Dataset folder
  <output>      Ouput folder. By default: ./outputs/
  <ckpt>        Path to the checkpoints to restore
"""

from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from PIL import Image, ImageEnhance
from docopt import docopt
import tensorflow as tf
import numpy as np
import random
import pickle
import os

from model import ModelTrafficSign
from data_handler import get_data


BATCH_SIZE = 50
DATASET_FOLDER = "dataset/"


def train(dataset, ckpt=None, output=None):
    """
        Train the model
        **input: **
            *dataset: (String) Dataset folder to used
            *ckpt: (String) [Optional] Path to the ckpt file to restore
            *output: (String) [Optional] Path to the output folder to used. ./outputs/ by default
    """

    def preprocessing_function(img):
        """
            Custom preprocessing_function
        """
        img = img * 255
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.5))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.5))

        return np.array(img) / 255

    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(dataset)

    X_train = X_train / 255
    X_valid = X_valid / 255
    X_test = X_test / 255

    train_datagen = ImageDataGenerator()
    train_datagen_augmented = ImageDataGenerator(
        rotation_range=20,
        shear_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocessing_function)
    inference_datagen = ImageDataGenerator()
    train_datagen.fit(X_train)
    train_datagen_augmented.fit(X_train)
    inference_datagen.fit(X_valid)
    inference_datagen.fit(X_test)

    # Utils method to print the current progression
    def plot_progression(b, cost, acc, label): print(
        "[%s] Batch ID = %s, loss = %s, acc = %s" % (label, b, cost, acc))

    # Init model
    model = ModelTrafficSign("TrafficSign", output_folder=output)
    if ckpt is None:
        model.init()
    else:
        model.load(ckpt)

    # Training pipeline
    b = 0
    valid_batch = inference_datagen.flow(X_valid, y_valid, batch_size=BATCH_SIZE)
    best_validation_loss = None
    augmented_factor = 0.99
    decrease_factor = 0.80
    train_batches = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    augmented_train_batches = train_datagen_augmented.flow(X_train, y_train, batch_size=BATCH_SIZE)

    while True:
        next_batch = next(
            augmented_train_batches if random.uniform(0, 1) < augmented_factor else train_batches)
        x_batch, y_batch = next_batch

        ### Training
        cost, acc = model.optimize(x_batch, y_batch)
        ### Validation
        x_batch, y_batch = next(valid_batch, None)
        # Retrieve the cost and acc on this validation batch and save it in tensorboard
        cost_val, acc_val = model.evaluate(x_batch, y_batch, tb_test_save=True)

        if b % 10 == 0: # Plot the last results
            plot_progression(b, cost, acc, "Train")
            plot_progression(b, cost_val, acc_val, "Validation")
        if b % 1000 == 0: # Test the model on all the validation
            print("Evaluate full validation dataset ...")
            loss, acc, _ = model.evaluate_dataset(X_valid, y_valid)
            print("Current loss: %s Best loss: %s" % (loss, best_validation_loss))
            plot_progression(b, loss, acc, "TOTAL Validation")
            if best_validation_loss is None or loss < best_validation_loss:
                best_validation_loss = loss
                model.save()
            augmented_factor = augmented_factor * decrease_factor
            print("Augmented Factor = %s" % augmented_factor)

        b += 1

if __name__ == '__main__':
    arguments = docopt(__doc__)
    train(arguments["<dataset>"], arguments["--ckpt"], arguments["<output>"])
