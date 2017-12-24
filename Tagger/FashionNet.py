import json
import numpy as np
import tensorflow as tf

from Tagger.image_formatter import preprocess

import Tagger.FashionNet_Layers as L

config = json.load(open('config.json'))
labels = json.load(open('labels.json'))


class FashionNet(object):

    classifications = {label: len(labels[label]) for label in labels.keys()}

    def __init__(self):

        self.iteration = tf.Variable(0, name='iteration', trainable=False, dtype=tf.int32)
        self.iteration_step_op = tf.Variable.assign_add(self.iteration, 1)

        self.stimulus = tf.placeholder(tf.float32, [None, 224, 224, 3])

        # convolutional weights should be a member of all classifier variable collections
        all_coll = [tf.GraphKeys.GLOBAL_VARIABLES, *self.classifications]

        # block 1 -- outputs 112x112x64
        net = L.conv(self.stimulus, name="conv1_1", kh=3, kw=3, n_out=64, collections=all_coll)
        net = L.conv(net, name="conv1_2", kh=3, kw=3, n_out=64, collections=all_coll)
        net = L.pool(net, name="pool1", kh=2, kw=2, dw=2, dh=2)

        # block 2 -- outputs 56x56x128
        net = L.conv(net, name="conv2_1", kh=3, kw=3, n_out=128, collections=all_coll)
        net = L.conv(net, name="conv2_2", kh=3, kw=3, n_out=128, collections=all_coll)
        net = L.pool(net, name="pool2", kh=2, kw=2, dh=2, dw=2)

        # # block 3 -- outputs 28x28x256
        net = L.conv(net, name="conv3_1", kh=3, kw=3, n_out=256, collections=all_coll)
        net = L.conv(net, name="conv3_2", kh=3, kw=3, n_out=256, collections=all_coll)
        net = L.pool(net, name="pool3", kh=2, kw=2, dh=2, dw=2)

        # block 4 -- outputs 14x14x512
        net = L.conv(net, name="conv4_1", kh=3, kw=3, n_out=512, collections=all_coll)
        net = L.conv(net, name="conv4_2", kh=3, kw=3, n_out=512, collections=all_coll)
        net = L.conv(net, name="conv4_3", kh=3, kw=3, n_out=512, collections=all_coll)
        net = L.pool(net, name="pool4", kh=2, kw=2, dh=2, dw=2)

        # block 5 -- outputs 7x7x512
        net = L.conv(net, name="conv5_1", kh=3, kw=3, n_out=512, collections=all_coll)
        net = L.conv(net, name="conv5_2", kh=3, kw=3, n_out=512, collections=all_coll)
        net = L.conv(net, name="conv5_3", kh=3, kw=3, n_out=512, collections=all_coll)
        net = L.pool(net, name="pool5", kh=2, kw=2, dw=2, dh=2)

        # flatten before loading into feedforward classifier
        flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
        net = tf.reshape(net, [-1, flattened_shape], name="flatten")

        self.expected = {}
        self.predict = {}
        self.loss = {}
        self.train_op = {}

        for label in labels.keys():
            self.expected[label] = tf.placeholder(tf.float32)
            self.predict[label] = tf.nn.softmax(self.classifier(net, label))

            loss_classify = L.loss(self.predict[label], self.expected[label])
            loss_weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(i) for i in tf.get_collection(label)]))
            self.loss[label] = loss_classify + config['weight_decay'] * loss_weight_decay

            optimizer = tf.train.GradientDescentOptimizer(config['step_size'])
            self.train_op[label] = optimizer.minimize(self.loss[label], var_list=tf.get_collection(label))

    def classifier(self, net, label):

        collections = [label, tf.GraphKeys.GLOBAL_VARIABLES]

        net = L.fully_connected(net, name=label+"_1", n_out=2048, collections=collections)
        net = tf.nn.dropout(net, config['keep_prob'])
        net = L.fully_connected(net, name=label+"_2", n_out=1024, collections=collections)
        net = tf.nn.dropout(net, config['keep_prob'])
        net = L.fully_connected(net, name=label+"_3", n_out=self.classifications[label], collections=collections)
        return net

    def get_attributes(self, image):
        predictions = {}

        for label, predict in self.predict.items():
            predictions[label] = labels[label][np.argmax(predict(np.array(preprocess(image))[None])[0])]

        return predictions
