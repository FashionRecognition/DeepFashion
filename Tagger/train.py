import os
import sys
import random

import tensorflow as tf
from pymongo import MongoClient

import numpy as np

from PIL import Image
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt

from Tagger.FashionNet import FashionNet
from Tagger.image_formatter import preprocess

mongo_client = MongoClient(host='localhost', port=27017)  # Default port
db = mongo_client.deep_fashion

network = FashionNet()
save_all = tf.train.Saver(max_to_keep=4,)
save_path = os.path.dirname(os.path.realpath(__file__)) + '/saved/'

# batch size of 16 is stable for 4GB of graphics card memory, comment the log level to tweak on your own system
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
batch_size = 16

# when loss is less than epsilon, consider the model converged. Wouldn't this be nice?
epsilon = 5

# How often to check for convergence and print
frequency = 10
history = {label: [] for label in FashionNet.labels.keys()}

# 538 color palette
plt.style.use('fivethirtyeight')
matplotlib.rcParams.update({'font.size': 12})
colors = np.array([[48, 162, 218],
                   [252, 79, 48],
                   [229, 174, 56],
                   [109, 144, 79],
                   [139, 139, 139]]).astype(float) / 256

# Once it plots, the main thread returns to network administration
# This means the plots aren't very responsive- to fix this split it off to a different thread. It still plots though
fig = plt.figure(figsize=(5, 3))
plt.title('Loss')
plt.show(block=False)

with tf.Session() as sess:

    if os.path.exists(save_path):
        save_all.restore(sess, tf.train.latest_checkpoint(save_path))
    else:
        sess.run(tf.global_variables_initializer())


    def prepare(data, label):
        # Convert list of dictionaries into two lists of values
        binaries, tags = [i for i in zip(*[(stim['image'], stim[label]) for stim in data])]

        # Convert a list of binaries to a 4D stimulus array
        stimulus = np.stack([np.array(preprocess(Image.open(BytesIO(binary)))) for binary in binaries])

        # Convert a text label to a onehot encoding
        expected = np.stack([np.eye(FashionNet.classifications[label])[FashionNet.labels[label].index(tag)]
                             for tag in tags])

        return stimulus, expected


    def checkpoint(iteration):
        print()

        total_loss = 0
        # Sample from records with all attributes
        query = [{"$match": {label: {"$exists": True} for label in FashionNet.labels.keys()}},
                 {"$project": {'image': 1, **{label: 1 for label in FashionNet.labels.keys()}}},
                 {"$sample": {"size": batch_size}}]
        # Exhaust the generator into a list b
        data = list(db.ebay.aggregate(query))

        for label in FashionNet.labels.keys():

            try:
                # Reformat sampled data into input/output values
                stimulus, expected = prepare(data, label)

                loss = sess.run(network.loss[label], feed_dict={
                    network.stimulus: stimulus,
                    network.expected[label]: expected
                })
            except TypeError:
                print("Invalid image in sample, re-attempting checkpoint.")
                checkpoint(iteration)

            history[label].append((iteration, loss))
            print("\t" + label + ": " + str(loss))

            total_loss += loss

        # Plotting
        plt.gca().clear()
        plt.title('Loss')
        for index, label in enumerate(FashionNet.labels.keys()):
            plt.plot(*zip(*history[label]), marker='.', color=colors[index], label=label)
            plt.legend(loc='upper left', prop={'size': 6})

        # Draw
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(0.00001)

        # Store the graph
        save_all.save(sess, save_path + 'model.ckpt')

        # If converged, return false to end the program
        return total_loss > epsilon

    iteration = sess.run(network.iteration)

    print("Iteration: " + str(iteration), end="")

    while iteration % frequency != 0 or checkpoint(iteration):
        iteration = sess.run(network.iteration_step_op)

        label = random.choice(list(FashionNet.labels.keys()))

        query = [{"$match": {label: {"$exists": True}}},
                 {"$project": {'image': 1, label: 1}},
                 {"$sample": {"size": batch_size}}]
        data = db.ebay.aggregate(query)

        try:
            # Batch gradient update
            stimulus, expected = prepare(data, label)
            sess.run(network.train_op[label], feed_dict={
                network.stimulus: stimulus,
                network.expected[label]: expected
            })
        except ValueError:
            # There seem to be some faulty images? Just ignore them when they come up
            continue

        # Show status
        sys.stdout.write("\r\x1b[KIteration: " + str(iteration))
        sys.stdout.flush()
