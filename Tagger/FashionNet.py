import tensorflow as tf
import numpy as np
import Tagger.layer_VGG16 as L


class FashionNet(object):

    def __init__(self, pretrained=False):
        n_classes = 1000
        weight_decay = 0.0005

        # define the graph

        self.stimulus = tf.placeholder(tf.float32, [1, 224, 224, 3])
        self.expected = tf.placeholder(tf.float32)

        # block 1 -- outputs 112x112x64
        net = L.conv(self.stimulus, name="conv1_1", kh=3, kw=3, n_out=64)
        net = L.conv(net, name="conv1_2", kh=3, kw=3, n_out=64)
        net = L.pool(net, name="pool1", kh=2, kw=2, dw=2, dh=2)

        # block 2 -- outputs 56x56x128
        net = L.conv(net, name="conv2_1", kh=3, kw=3, n_out=128)
        net = L.conv(net, name="conv2_2", kh=3, kw=3, n_out=128)
        net = L.pool(net, name="pool2", kh=2, kw=2, dh=2, dw=2)

        # # block 3 -- outputs 28x28x256
        net = L.conv(net, name="conv3_1", kh=3, kw=3, n_out=256)
        net = L.conv(net, name="conv3_2", kh=3, kw=3, n_out=256)
        net = L.pool(net, name="pool3", kh=2, kw=2, dh=2, dw=2)

        # block 4 -- outputs 14x14x512
        net = L.conv(net, name="conv4_1", kh=3, kw=3, n_out=512)
        net = L.conv(net, name="conv4_2", kh=3, kw=3, n_out=512)
        net = L.conv(net, name="conv4_3", kh=3, kw=3, n_out=512)
        net = L.pool(net, name="pool4", kh=2, kw=2, dh=2, dw=2)

        # block 5 -- outputs 7x7x512
        net = L.conv(net, name="conv5_1", kh=3, kw=3, n_out=512)
        net = L.conv(net, name="conv5_2", kh=3, kw=3, n_out=512)
        net = L.conv(net, name="conv5_3", kh=3, kw=3, n_out=512)
        net = L.pool(net, name="pool5", kh=2, kw=2, dw=2, dh=2)

        # flatten
        flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
        net = tf.reshape(net, [-1, flattened_shape], name="flatten")

        # fully connected
        keep_prob = 0.5
        net = L.fully_connected(net, name="fc6", n_out=4096)
        net = tf.nn.dropout(net, keep_prob)
        net = L.fully_connected(net, name="fc7", n_out=4096)
        net = tf.nn.dropout(net, keep_prob)
        net = L.fully_connected(net, name="fc8", n_out=n_classes)

        logits = net
        self.predict = tf.nn.softmax(logits)

        loss_classify = L.loss(logits, self.expected)
        loss_weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(i) for i in tf.get_collection('variables')]))
        self.loss = loss_classify + weight_decay * loss_weight_decay

        optimizer = tf.train.GradientDescentOptimizer(0.01)
        self.train_op = optimizer.minimize(self.loss)

    def save(self):
        pass

    def get_attributes(self, image):
        # TODO: Call the network.
        im = np.array(image)

        # Run the model, remap to labelnames, then return
        return ['fur', 'light']
