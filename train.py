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
from docopt import docopt
import tensorflow as tf
import numpy as np
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
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(dataset)

    # I do not used image augmentation for the moment. This is a
    train_datagen = ImageDataGenerator(rescale=(1/255))
    inference_datagen = ImageDataGenerator(rescale=(1/255))
    train_datagen.fit(X_train)
    inference_datagen.fit(X_valid)
    inference_datagen.fit(X_test)

    # Utils method to print the current progression
    def plot_progression(b, cost, acc, label): print(
        "[%s] Batch ID = %s, loss = %s, acc = %s" % (label, b, cost, acc))

    # Init model
    model = ModelTrafficSign("test", output_folder=output)
    model.init()

    b = 0
    valid_batch = inference_datagen.flow(X_valid, y_valid, batch_size=BATCH_SIZE)
    best_validation_loss = None
    # Training pipeline
    for x_batch, y_batch in train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE):
        ### Training
        cost, acc = model.optimize(x_batch, y_batch)
        ### Validation
        x_batch, y_batch = next(valid_batch, None)
        # Retrieve the cost and acc on this validation batch and save it in tensorboard
        cost_val, acc_val = model.evaluate(x_batch, y_batch, tb_test_save=True)
        if b % 10 == 0: # Plot the last results
            plot_progression(b, cost, acc, "Train")
            plot_progression(b, cost_val, acc_val, "Validation")
        if b % 100 == 0: # Test the model on all the validation
            # TODO
            model.save()

        b += 1

if __name__ == '__main__':
    arguments = docopt(__doc__)
    train(arguments["<dataset>"], arguments["--ckpt"], arguments["<output>"])
