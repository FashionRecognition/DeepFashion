import numpy as np
from PIL import Image

from io import BytesIO
from pymongo import MongoClient
import json

from Tagger.image_formatter import preprocess

mongo_client = MongoClient(host='localhost', port=27017)  # Default port
db = mongo_client.deep_fashion

config = json.load(open('default_config.json'))


def find_normalizers():
    samples = 5000

    image_list = []
    for listing in db.ebay.aggregate([{"$sample": {"size": samples}}]):
        image_list.append(np.array(preprocess(Image.open(BytesIO(listing['image'])), config['image_shape'], mask=False, letterboxing=False)))

    images = np.array(image_list)
    print(images.shape)
    print(np.max(images.std(axis=0)))

    np.save('pixel_mean.npy', images.mean(axis=0))
    np.save('pixel_deviation.npy', images.std(axis=0))


def normalize_sample(quantity):
    # Use the mean and deviance to produce normalized data for an arbitrary image
    pixel_mean = np.load('pixel_mean.npy')
    pixel_deviance = np.load('pixel_deviation.npy')

    for record in list(db.ebay.aggregate([{"$sample": {"size": quantity}}])):
        # print(record['title'])

        im = preprocess(Image.open(BytesIO(record['image'])), config['image_shape'], mask=False, letterboxing=False)
        standardized = (im - pixel_mean) / pixel_deviance

        print(standardized)


# Only needs to be run once
# find_normalizers()


# Validate that it works correctly
normalize_sample(20)
