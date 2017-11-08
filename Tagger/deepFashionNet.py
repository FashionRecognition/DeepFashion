import tensorflow as tf
import re
import zipfile
from PIL import Image
import numpy as np
import os
import layers as L
import pandas as pd
from sklearn.utils import shuffle
import random
from random import shuffle

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
c = 0;
list = []
trainImages = []
d = set();
tok = []
trainDict = {}
tempDict = {}
validationImages = []
testImages = []
n_classes = 1000;

# batch_size: 256

# num_classes: 10

learning_rate = 0.01

data_dims = [32, 32, 3]

weight_decay = 0.0005

# with open(r'C:\Users\Varun\ADM2017\list_eval_partition.txt') as input_file:
#     for line in input_file:
#         tok = line.split()
#         if (tok[1] == "train"):
#             trainImages.append(tok[0])
#         if (tok[1] == "val"):
#             validationImages.append(tok[0])
#         if (tok[1] == "test"):
#             testImages.append(tok[0])
#     print("Length of the training set ")
#     print(len(trainImages))
#
#     print("Length of validation set")
#     print(len(validationImages))
#     print("Length of Test Set ")
#     print(len(testImages))
# tt = []
# ccc = 0
# labelList = []
#with open(r'C:\Users\Varun\ADM2017\list_attr_img.txt') as input_file:
 # count=0;
 #for line in input_file:
#     count+=1;
#    tok=line.split(" ")
#   tempDict[tok[0]]=np.clip(np.array(re.split(' +',line)[1:]).astype(int),0,1)
#   np.save('my_file.npy', tempDict)

# print(df.head(5))
#tempDict = np.load('my_file.npy').item()
#print(read_dictionary['hello'])  # displays "world"
#print("Length of the Training images only Dictionary########################")
#print(len(tempDict))

# for image in trainImages:
#     trainDict[image] = tempDict.get(image)
# np.save('final_dict.npy', trainDict)
# print("Final_dict Dictionary is Saved  ")
# print("Length of the Final Training Dictionary")
# print(len(trainDict))



c = 0

# defining the graph
rgb_mean = np.array([116.779, 123.68, 103.939], dtype=np.float32)

mu = tf.constant(rgb_mean, name="rgb_mean")

keep_prob = 0.5
# subtract image mean

X = tf.placeholder(tf.float32, [1, 224, 224, 3]);
Y = tf.placeholder(tf.float32);

x = tf.placeholder(tf.float32)  # , [224, 224, 3]);
y = tf.placeholder(tf.float32);
net = tf.subtract(X, mu, name="input_mean_centered")

# block 1 -- outputs 112x112x64

net = L.conv(net, name="conv1_1", kh=3, kw=3, n_out=64)

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

net = L.fully_connected(net, name="fc6", n_out=4096)

net = tf.nn.dropout(net, keep_prob)

net = L.fully_connected(net, name="fc7", n_out=4096)

net = tf.nn.dropout(net, keep_prob)

net = L.fully_connected(net, name="fc8", n_out=n_classes)

logits = net

probs = tf.nn.softmax(logits)

#print(len(logits))

loss_classify = L.loss(logits, Y)

loss_weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(i) for i in tf.get_collection('variables')]))

loss = loss_classify + weight_decay * loss_weight_decay

# error_top5 = L.topK_error(probs,Y, K=5)

# error_top1 = L.topK_error(probs,Y, K=1)

optimizer = tf.train.GradientDescentOptimizer(0.01);

train_op = optimizer.minimize(loss)
init_op = tf.global_variables_initializer()

file = 'file'

read_dictionary = np.load('my_file.npy').item()
print(read_dictionary['img/1981_Graphic_Ringer_Tee/img_00000003.jpg'])  # displays "world"

with tf.Session() as sess:
    sess.run(init_op)
    path = r'C:\Users\Varun\Fashion\Fashion_Subset_Processed224'
    c = 0;
    trainDict = np.load('final_dict.npy').item()
    for key, value in sorted(trainDict.items(), key=lambda x: random.random()):

        key = re.sub('[jpg]{3}$', 'png', key)
        key = path + r"\\" + key;
        print(key)
        x = np.array(Image.open(key))
        print(x)

        # name.replace(path, "")
        # print(name)
        # print(x.shape)
        xLabel = trainDict["img/1981_Graphic_Ringer_Tee/img_00000003.jpg"];
        l = sess.run([loss], feed_dict={X: x.reshape((1, 224, 224, 3)), Y: xLabel});

        print(l)

