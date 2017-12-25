import os
import sys
import json
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

# batch size of 16 is stable for 4GB of graphics card memory, comment the log level to tweak on your own system
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 538 color palette
plt.style.use('fivethirtyeight')
matplotlib.rcParams.update({'font.size': 12})
colors = np.array([[48, 162, 218],
                   [252, 79, 48],
                   [229, 174, 56],
                   [109, 144, 79],
                   [139, 139, 139]]).astype(float) / 256

mongo_client = MongoClient(host='localhost', port=27017)  # Default port
db = mongo_client.deep_fashion

# To create a new model, edit the default_config.json, then run with your 'model_name'
model_name = 'first_gradient_descent'
save_path = os.path.dirname(os.path.realpath(__file__)) + '\\saved\\' + model_name + '\\'
save = True

labels = json.load(open('labels.json'))

with tf.Session() as sess:
    # Load configs
    if os.path.exists(save_path):
        config = json.load(open('saved\\' + model_name + '\\config.json'))
        history = json.load(open('saved\\' + model_name + '\\history.json'))
    else:
        config = json.load(open('default_config.json'))
        history = {label: [] for label in labels.keys()}

    # Network construction
    network = FashionNet(config, labels)
    save_all = tf.train.Saver(max_to_keep=4, save_relative_paths=True)

    # Variable initialization
    if os.path.exists(save_path):
        save_all.restore(sess, tf.train.latest_checkpoint(save_path))
    else:
        # Set up config and history files for the first time
        if save:
            os.makedirs(save_path)
            with open(save_path + 'config.json', 'w') as outfile:
                json.dump(config, outfile)
            with open(save_path + 'history.json', 'w') as outfile:
                json.dump(history, outfile)

        sess.run(tf.global_variables_initializer())

    # Once it plots, the main thread returns to network administration
    # This means the plots aren't very responsive- to fix this split it off to a different thread. It still plots though
    if config['plot']:
        fig = plt.figure(figsize=(5, 3))
        plt.title('Loss')
        plt.show(block=False)


    def prepare(data, label):
        # Convert list of dictionaries into two lists of values
        binaries, tags = [i for i in zip(*[(stim['image'], stim[label]) for stim in data])]

        # Convert a list of binaries to a 4D stimulus array
        stim = np.stack([np.array(preprocess(Image.open(BytesIO(binary)))) for binary in binaries])

        # Convert a text label to a onehot encoding
        exp = np.stack([np.eye(FashionNet.classifications[label])[labels[label].index(tag)]
                        for tag in tags])

        return stim, exp


    # Check losses, update plot, save model, check for convergence
    def checkpoint(check_iteration):
        print()

        total_loss = 0
        # Take first n records with all attributes
        query = [{"$match": {label: {"$exists": True} for label in labels.keys()}},
                 {"$project": {'image': 1, **{label: 1 for label in labels.keys()}}},
                 {"$limit": config['batch_size']}]
        # Exhaust the generator into a list b
        samples = list(db.ebay.aggregate(query))

        for label in labels.keys():

            # Reformat sampled data into input/output values
            stimulus, expected = prepare(samples, label)

            loss = sess.run(network.loss[label], feed_dict={
                network.stimulus: stimulus,
                network.expected[label]: expected
            })

            if config['plot']:
                history[label].append((int(check_iteration), float(loss)))

            print("\t" + label + ": " + str(loss))
            total_loss += loss

        # Store the graph
        if save:
            save_all.save(sess, save_path + 'model.ckpt')

        # Plotting
        if config['plot']:
            with open(save_path + 'history.json', 'w') as histfile:
                json.dump(history, histfile)

            plt.gca().clear()
            plt.title('Loss')
            for index, label in enumerate(labels.keys()):
                plt.plot(*zip(*history[label]), marker='.', color=colors[index], label=label)
                plt.legend(loc='upper left', prop={'size': 6})

            # Draw
            backend = plt.rcParams['backend']
            if backend in matplotlib.rcsetup.interactive_bk:
                fig_manager = getattr(matplotlib, '_pylab_helpers').Gcf.get_active()
                if fig_manager is not None:
                    canvas = fig_manager.canvas
                    if canvas.figure.stale:
                        canvas.draw()
                    canvas.start_event_loop(0.00001)

        # If converged, return false to end the program. Wouldn't this be nice?
        return total_loss > config['epsilon']


    # Sample, compute gradients, and update weights
    def update():
        label = random.choice(list(labels.keys()))

        query = [{"$match": {label: {"$exists": True}}},
                 {"$project": {'image': 1, label: 1}},
                 {"$sample": {"size": config['batch_size']}}]
        data = db.ebay.aggregate(query)

        try:
            # Batch gradient update
            stimulus, expected = prepare(data, label)
            sess.run(network.train_op[label], feed_dict={
                network.stimulus: stimulus,
                network.expected[label]: expected
            })
        except tf.errors.ResourceExhaustedError:
            print("Resources Exhausted. Attempting to continue.")


    iteration = sess.run(network.iteration)

    print("Iteration: " + str(iteration), end="")

    while iteration % config['frequency'] != 0 or checkpoint(iteration):
        iteration = sess.run(network.iteration_step_op)

        update()

        # Show status
        sys.stdout.write("\r\x1b[KIteration: " + str(iteration))
        sys.stdout.flush()
