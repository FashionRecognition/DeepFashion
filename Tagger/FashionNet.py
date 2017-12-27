import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import xavier_initializer

from Tagger.image_formatter import preprocess


class FashionNet(object):
    classifications = {}

    def __init__(self, config, labels, dimensions):
        self.config = config
        self.labels = labels

        self.classifications = {label: len(self.labels[label]) for label in self.labels.keys()}

        self.iteration = tf.Variable(0, name='iteration', trainable=False, dtype=tf.int32)
        self.iteration_step_op = tf.Variable.assign_add(self.iteration, 1)

        self.stimulus = tf.placeholder(tf.float32, [None, *dimensions, 3])

        # convolutional weights should be a member of all classifier variable collections
        all_coll = [tf.GraphKeys.GLOBAL_VARIABLES, *self.classifications.keys()]

        # If dimensions are 192x256, outputs are given

        # block 1 -- outputs 96x128x64
        net = conv(self.stimulus, name="conv1_1", kh=3, kw=3, n_out=64, collections=all_coll)
        net = conv(net, name="conv1_2", kh=3, kw=3, n_out=64, collections=all_coll)
        net = pool(net, name="pool1", kh=2, kw=2, dw=2, dh=2)

        # block 2 -- outputs 48x64x128
        net = conv(net, name="conv2_1", kh=3, kw=3, n_out=128, collections=all_coll)
        net = conv(net, name="conv2_2", kh=3, kw=3, n_out=128, collections=all_coll)
        net = pool(net, name="pool2", kh=2, kw=2, dh=2, dw=2)

        # # block 3 -- outputs 24x32x256
        net = conv(net, name="conv3_1", kh=3, kw=3, n_out=256, collections=all_coll)
        net = conv(net, name="conv3_2", kh=3, kw=3, n_out=256, collections=all_coll)
        net = pool(net, name="pool3", kh=2, kw=2, dh=2, dw=2)

        # block 4 -- outputs 12x16x512
        net = conv(net, name="conv4_1", kh=3, kw=3, n_out=512, collections=all_coll)
        net = conv(net, name="conv4_2", kh=3, kw=3, n_out=512, collections=all_coll)
        net = conv(net, name="conv4_3", kh=3, kw=3, n_out=512, collections=all_coll)
        net = pool(net, name="pool4", kh=2, kw=2, dh=2, dw=2)

        # block 5 -- outputs 6x8x512
        net = conv(net, name="conv5_1", kh=3, kw=3, n_out=512, collections=all_coll)
        net = conv(net, name="conv5_2", kh=3, kw=3, n_out=512, collections=all_coll)
        net = conv(net, name="conv5_3", kh=3, kw=3, n_out=512, collections=all_coll)
        net = pool(net, name="pool5", kh=2, kw=2, dw=2, dh=2)

        # flatten before loading into feedforward classifier
        flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
        net = tf.reshape(net, [-1, flattened_shape], name="flatten")

        self.expected = {}
        self.predict = {}
        self.loss = {}
        self.train_op = {}

        for label in self.labels.keys():
            self.expected[label] = tf.placeholder(tf.float32)
            self.predict[label] = tf.nn.softmax(self.classifier(net, label))

            # Compute loss for a set of samples, then collapse over the batch
            xentropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.predict[label],
                labels=self.expected[label],
                name='xentropy_' + label)
            loss_classify = tf.reduce_mean(xentropy, name='loss_' + label)

            # Weight decay
            loss_weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(i) for i in tf.get_collection(label)]))
            self.loss[label] = loss_classify + config['weight_decay'] * loss_weight_decay

            # Compute gradients
            optimizer = tf.train.GradientDescentOptimizer(config['step_size'], name='optimizer_' + label)
            gradients = optimizer.compute_gradients(self.loss[label], var_list=tf.get_collection(label))

            # Training uses gradient clipping via norm
            gradients_clipped = [(tf.clip_by_norm(grad, config['gradient_norm']), var) for grad, var in gradients]
            self.train_op[label] = optimizer.apply_gradients(gradients_clipped)

    def classifier(self, net, label):

        collections = [label, tf.GraphKeys.GLOBAL_VARIABLES]

        net = fully_connected(net, name=label+"_1", n_out=2048, collections=collections)
        net = tf.nn.dropout(net, self.config['keep_prob'])
        net = fully_connected(net, name=label+"_2", n_out=1024, collections=collections)
        net = tf.nn.dropout(net, self.config['keep_prob'])
        net = fully_connected(net, name=label+"_3", n_out=self.classifications[label], collections=collections)
        return net

    def get_attributes(self, image):
        predictions = {}

        for label, predict in self.predict.items():
            predictions[label] = self.labels[label][np.argmax(predict(np.array(preprocess(image))[None])[0])]

        return predictions


def conv(input_tensor, name, kw, kh, n_out, dw=1, dh=1, activation_fn=tf.nn.relu, collections=None):
    if not collections:
        collections = [tf.GraphKeys.GLOBAL_VARIABLES]

    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [kh, kw, n_in, n_out],
                                  tf.float32, xavier_initializer(), collections=collections)
        biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0), collections=collections)
        convolution = tf.nn.conv2d(input_tensor, weights, (1, dh, dw, 1), padding='SAME')

        return activation_fn(tf.nn.bias_add(convolution, biases))


def fully_connected(input_tensor, name, n_out, activation_fn=tf.nn.relu, collections=None):
    if not collections:
        collections = [tf.GraphKeys.GLOBAL_VARIABLES]

    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [n_in, n_out], tf.float32, xavier_initializer(), collections=collections)
        biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0), collections=collections)
        logits = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
        return activation_fn(logits)


def pool(input_tensor, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_tensor,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='VALID',
                          name=name)
