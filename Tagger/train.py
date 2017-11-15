import re
import zipfile

c = 0;
list = []
trainImages = []
d = set()
tok = []
trainDict = {}
tempDict = {}
validationImages = []
testImages = []

with open('list_eval_partition.txt') as input_file:
    for line in input_file:
        tok = line.split()
        if (tok[1] == "train"):
            trainImages.append(tok[0])
        if (tok[1] == "val"):
            validationImages.append(tok[0])
        if (tok[1] == "test"):
            testImages.append(tok[0])
    print("Length of the training set ")
    print(len(trainImages))

    print("Length of validation set")
    print(len(validationImages))
    print("Length of Test Set ")
    print(len(testImages))

with open('list_attr_img.txt') as input_file:
    for line in input_file:
        tok = line.split()
        tempDict[tok[0]] = tok[1]
    print("Length of the Training images only Dictionary")
    print(len(tempDict))

    for image in trainImages:
        trainDict[image] = tempDict.get(image)

    print("Length of the Final Training Dictionary")
    print(len(trainDict))

    # for i in range(5):
    # print(trainDict.values())
