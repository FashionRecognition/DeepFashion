import tensorflow as tf
import numpy as np


class FashionNet(object):

    def __init__(self, pretrained=True):
        pass

    def save(self):
        pass

    def get_attributes(self, image):
        im = np.array(image)

        # Run the model, remap to labelnames, then return
        return ['fur', 'light']
