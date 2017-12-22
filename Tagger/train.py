import random

import tensorflow as tf
from pymongo import MongoClient

import numpy as np

from PIL import Image
from io import BytesIO

from Tagger.FashionNet import FashionNet
from Tagger.image_formatter import preprocess

mongo_client = MongoClient(host='localhost', port=27017)  # Default port
db = mongo_client.deep_fashion

network = FashionNet()
batch_size = 5


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    while True:
        label = random.choice(list(FashionNet.labels.keys()))

        query = [{"$match": {label: {"$exists": True}}},
                 {"$project": {'image': 1, label: 1}},
                 {"$sample": {"size": batch_size}}]

        data = db.ebay.aggregate(query)
        binaries, tags = [i for i in zip(*[(stim['image'], stim[label]) for stim in data])]

        # Convert a list of binaries to a 4D stimulus array
        stimulus = np.stack([np.array(preprocess(Image.open(BytesIO(binary)))) for binary in binaries])

        # Convert a text label to a onehot encoding
        expected = np.stack([np.eye(FashionNet.classifications[label])[FashionNet.labels[label].index(tag)]
                             for tag in tags])

        # Compute loss for the batch
        l = sess.run([network.loss[label]], feed_dict={
            network.stimulus: stimulus,
            network.expected[label]: expected
        })

        print(l)
