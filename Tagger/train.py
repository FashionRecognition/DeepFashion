import numpy as np
import random
import re
from PIL import Image
import tensorflow as tf

from Tagger.FashionNet import FashionNet

read_dictionary = np.load('./sources/df_women_partition.npy').item()
print(read_dictionary['img/1981_Graphic_Ringer_Tee/img_00000003.jpg'])  # displays "world"

network = FashionNet()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    trainDict = np.load('./sources/df_women_training.npy').item()
    for key, value in sorted(trainDict.items(), key=lambda x: random.random()):

        x = np.array(Image.open(r'.\processed\\' + re.sub('[jpg]{3}$', 'png', key)))
        xLabel = trainDict["img/1981_Graphic_Ringer_Tee/img_00000003.jpg"]

        l = sess.run([network.loss], feed_dict={
            network.stimulus: x.reshape((1, 224, 224, 3)),
            network.expected: xLabel
        })

        print(l)
