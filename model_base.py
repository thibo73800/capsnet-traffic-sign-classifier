#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from collections import Counter
from utils import Utils as U
import json
import numpy as np
from logger import Logger
import time
import pickle
import os

log = Logger("ModelBase")

class Hyperparameters(object):
    """
        Simple class used to store Hyperparameters
    """
    def __init__(self):
        super(Hyperparameters, self).__init__()
        # List used to store list of hyperparameters name
        self.hyp_list = []

    def set_hyp(self, hyp):
        """
            Method used to store hyperparameters inside this class
            **input: **
                *hyp (Dict) Dictionary storing all hyperparameters values
        """
        for key in hyp:
            self.hyp_list.append(key)
            setattr(self, key, hyp[key])

class ModelBase(object):
    """
        Base Model Class
    """

    #  Hyp : Hyperparameters
    DEFAULT_OUTPUT = "outputs"
    DEFAULT_CHECKPOINT_FOLDER = "checkpoints"

    def __init__(self, model_name, hyperparameters_name=None, hyperparameters_content=None, output_folder=None):
        """
            **input:
                *hyperparameters_name: [Optional] (String|None) Path to the hyperparameters file
                                       By default: hyperparameters.json
                *model_name: (Integer) Name of this model
        """
        super(ModelBase, self).__init__()

        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        # Output folder
        if output_folder is None:
            self.output_folder = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), self.DEFAULT_OUTPUT)
        else:
            self.output_folder = output_folder

        hyp_folder = "settings"
        hyp_filename = "hyperparameters.json"
        hyp_path = os.path.join(self.current_dir, os.path.join(hyp_folder, hyp_filename))
        self.checkpoints_folder = os.path.join(self.output_folder, self.DEFAULT_CHECKPOINT_FOLDER)

        # Set hyperparameters path
        if hyperparameters_name is not None:
            hyp_path = os.path.join(
                self.current_dir, os.path.join(hyp_folder, hyperparameters_name))
        hyp_path = hyp_path if hyperparameters_name is None else hyp_path
        # Load hyperparameters content
        if hyperparameters_content is None:
            hyp_content = U.read_json_file(hyp_path)
        else:
            hyp_content = hyperparameters_content
        # Set hyperparameters
        self.h = Hyperparameters()
        self.h.set_hyp(hyp_content)
        # Set model names
        self.name = model_name
        self.model_name = model_name
        self._set_hyperparameters_name()
        # Since hyperparameters had changed, we need to set again each name
        self._set_names()

    def _create_conv(self, prev, shape, padding='VALID', strides=[1, 1, 1, 1], relu=False,
                     max_pooling=False, mp_ksize=[1, 2, 2, 1], mp_strides=[1, 2, 2, 1]):
        """
            Create a convolutional layer with relu and/mor max pooling(Optional)
        """
        conv_w = tf.Variable(tf.truncated_normal(shape=shape, mean = 0, stddev = 0.1,  seed=0))
        conv_b = tf.Variable(tf.zeros(shape[-1]))
        conv   = tf.nn.conv2d(prev, conv_w, strides=strides, padding=padding) + conv_b

        if relu:
            conv = tf.nn.relu(conv)

        if max_pooling:
            conv = tf.nn.max_pool(conv, ksize=mp_ksize, strides=mp_strides, padding='VALID')

        return conv

    def _fc(self, prev, input_size, output_size, relu=False, sigmoid=False, no_bias=False,
            softmax=False):
        """
            Create fully connecter layer with relu(Optional)
        """
        fc_w = tf.Variable(
            tf.truncated_normal(shape=(input_size, output_size), mean = 0., stddev = 0.1))
        fc_b = tf.Variable(tf.zeros(output_size))
        pre_activation = tf.matmul(prev, fc_w)
        activation = None

        if not no_bias:
            pre_activation = pre_activation + fc_b
        if relu:
            activation = tf.nn.relu(pre_activation)
        if sigmoid:
            activation = tf.nn.sigmoid(pre_activation)
        if softmax:
            activation = tf.nn.softmax(pre_activation)

        if activation is None:
            activation = pre_activation

        return activation, pre_activation

    def init_session(self):
        """
            Init tensorflow session
            A saver property is create at the same time
        """
        #  Create session
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        # Init variables
        self.sess.run(tf.global_variables_initializer())
        # Tensorboard
        self.tf_tensorboard = tf.summary.merge_all()
        train_log_name = os.path.join(
            os.path.join(self.output_folder, "tensorboard"), self.name, self.sub_train_log_name)
        test_log_name = os.path.join(
            os.path.join(self.output_folder, "tensorboard"), self.name, self.sub_test_log_name)
        self.train_writer = tf.summary.FileWriter(train_log_name, self.sess.graph)
        self.test_writer = tf.summary.FileWriter(test_log_name)
        self.train_writer_it = 0
        self.test_writer_it = 0

        # Backup tensors
        backup_tensors = {}
        for field in dir(self):
            if "tf_" in field and field.index("tf_") == 0:
                backup_tensors[field] = getattr(self, field).name
        tf.constant(json.dumps(backup_tensors), dtype=tf.string, name="model_base_tensors_backup")
        # Backup hyperparameters
        backup_hyp = {}
        for field in self.h.hyp_list:
            value = getattr(self.h, field)
            d_type = tf.int32 if isinstance(value, int) else tf.float32
            n_cst = tf.constant(value, dtype=d_type, name="hyp/%s" % field)
            backup_hyp[field] = n_cst.name
        tf.constant(json.dumps(backup_hyp), dtype=tf.string, name="model_base_hyp_backup")

    def get_equal_batches(self, data, labels, batch_size):
        """
            This method will return a generator class which could be used to
            get new batches with the same number of rows for each class

            **input:**
                *batch_size (int) Size of each batch
             **return (Python Generator of Batch class)**
        """
        labels = np.array(labels)

        indexs = np.arange(len(data))
        np.random.shuffle(indexs)

        data = data[indexs]
        labels = labels[indexs]

        max_size = Counter(labels).most_common()[-1][1]
        unique_label = np.array(list(set(labels)))
        nb_classes = len(unique_label)

        if batch_size > max_size:
            batch_size = max_size

        batch_per_class = batch_size // nb_classes
        iterations = max_size // batch_per_class

        for it in range(iterations):

            indexes = []

            for label in unique_label:
                n_indexes = np.where(labels==label)[0][it * batch_per_class: (it + 1) * batch_per_class]
                n_indexes = n_indexes.tolist()
                indexes += n_indexes

            indexes = np.array(indexes)

            x = data[indexes]
            y = labels[indexes]

            yield x, y


    def get_batches(self, data_list, batch_size, shuffle=True):
        """
            This method will return a generator class which could be used to
            get new batches.

            **input:**
                *batch_size (int) Size of each batch
             **return (Python Generator of Batch class)**
        """
        if shuffle:
            indexs = np.arange(len(data_list[0]))
            np.random.shuffle(indexs)

            for d, data in enumerate(data_list):
                data_list[d] = np.array(data_list[d])
                data_list[d] = data_list[d][indexs]

        iterations = len(data_list[0]) // batch_size
        for iteration in range(iterations):
            yield (dt[iteration * batch_size: (iteration + 1) * batch_size] for dt in data_list)

    def save(self, name=None):
        """
            Save the model
        """
        log.info("Saving model ...")

        if name is None:
            name = self.model_name

        if not os.path.exists(self.checkpoints_folder):
            os.makedirs(self.checkpoints_folder)

        save_path = self.saver.save(
            self.sess, os.path.join(self.checkpoints_folder, name))

        log.info("Model successfully saved here: %s" % save_path)

    def _set_hyperparameters_name(self):
        """
            Convert hyperparameters dict to a string
            This string will be used to set the models names
        """
        # Generate a little name for each hyperparameters
        hyperparameters_names = [("".join([p[0] for p in hyp.split("_")]), getattr(self.h, hyp))
                                 for hyp in self.h.hyp_list]
        self.hyperparameters_name = ""
        for index_hyperparameter, hyperparameter in enumerate(hyperparameters_names):
            short_name, value = hyperparameter
            prepend = "" if index_hyperparameter == 0 else "_"
            self.hyperparameters_name += "%s%s_%s" % (prepend, short_name, value)

    def _set_names(self):
        """
            Set all model names
        """
        name_time = "%s--%s" % (self.model_name, time.time())
        # model_name is used to set the ckpt name
        self.model_name = "%s--%s" % (self.hyperparameters_name, name_time)
        # sub_train_log_name is used to set the name of the training part in tensorboard
        self.sub_train_log_name = "%s-train--%s" % (self.hyperparameters_name, name_time)
        # sub_test_log_name is used to set the name of the testing part in tensorboard
        self.sub_test_log_name = "%s-test--%s" % (self.hyperparameters_name, name_time)

    def dump_batch(self, folder, data):
        """
            Save batches
            Mainly used for Reinforcement Learning
        """
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder)
        # Create folder if not exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        pickle.dump(data, open(os.path.join(folder, str(time.time())), "wb" ))


    def load(self, ckpt):
        """
            Load a model
        """
        log.info("Loading ckpt ...")
        #loaded_graph = tf.Graph()
        #tf.reset_default_graph()
        #g = tf.Graph()
        #with g.as_default():
        self.sess = tf.Session()
        # Load the graph
        loader = tf.train.import_meta_graph(ckpt + '.meta')
        loader.restore(self.sess, ckpt)

        g = tf.get_default_graph()

        # Search for the backup tensor
        tensor_names = [
            n.name for n in g.as_graph_def().node if "model_base_tensors_backup" in n.name]

        # Search for the backup hyp
        hyp_names = [
            n.name for n in g.as_graph_def().node if "model_base_hyp_backup" in n.name]

        # Get the tensor string
        #tensors = g.get_tensor_by_name(names[0])
        tensors = g.get_operation_by_name(tensor_names[0]).outputs
        hyps = g.get_operation_by_name(hyp_names[0]).outputs

        #self.sess.run(tf.global_variables_initializer())

        tensors = self.sess.run(tensors)[0]
        tensors = json.loads(tensors)
        for tensor in tensors:
            try:
                n_tensor = g.get_tensor_by_name(tensors[tensor])
            except Exception as e:
                n_tensor = g.get_operation_by_name(tensors[tensor])
            setattr(self, tensor, n_tensor)

        hyps = self.sess.run(hyps)[0]
        hyps = json.loads(hyps)
        for hyp in hyps:
            n_hyp = g.get_tensor_by_name(hyps[hyp])
            setattr(self.h, hyp, self.sess.run(n_hyp))

        log.info("Ckpt ready")

        # Tensorboard
        self.tf_tensorboard = tf.summary.merge_all()
        train_log_name = os.path.join(
            os.path.join(self.output_folder, "tensorboard"), self.name, self.sub_train_log_name)
        test_log_name = os.path.join(
            os.path.join(self.output_folder, "tensorboard"), self.name, self.sub_test_log_name)
        self.train_writer = tf.summary.FileWriter(train_log_name, self.sess.graph)
        self.test_writer = tf.summary.FileWriter(test_log_name)
        self.train_writer_it = 0
        self.test_writer_it = 0

        self.model_name = ckpt.split("/")[-1]
        self.saver = tf.train.Saver()


if __name__ == '__main__':
    base_model = BaseModel("test")
