from pymongo import MongoClient
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import sys

# openCV package is distributed as a binary, so references won't resolve
import cv2

mongo_client = MongoClient(host='localhost', port=27017)  # Default port
db = mongo_client.deep_fashion

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.dpi'] = 200


# A training image should have standard 224x224 dimensions
def preprocess(im):
    dimensions = min(im.size)

    # Width is smaller dimension
    if im.size[0] == dimensions:
        xoff = 0
        yoff = (im.size[0] - dimensions) / 2
    else:
        xoff = (im.size[1])
        yoff = 0

    # Reduce size
    im = im.crop((xoff, yoff, dimensions, dimensions))
    im = im.resize((224, 224), Image.NEAREST)

    # Normalize the image
    # colors = np.array(im).astype(float)
    # colors -= 128
    # colors /= 256
    # print(colors)
    # colors = (colors - np.mean(colors, axis=2)[..., None]) / np.std(colors, axis=2)[..., None]
    # colors *= 256
    # colors += 128
    # colors = np.clip(colors, 0, 255)

    # Image.fromarray(colors, 'RGB').save(fullname_png)
    return im


def canny_mask(img):
    # Operate on greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection, thresholded between 10 and 200
    edges = cv2.Canny(gray, 10, 200)

    # Dilation then erosion to clean noising in edges, called closing. No kernel is used
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, None)

    # At this stage, the image is b/w, with white along edges, black within regions

    # ~~~ Now find regions, which are enclosed by contours:
    # Retrieval mode - consider landlocked contours children. TREE preserves hierarchy, LIST flattens
    # Chain approx - how many points used along edge of region.
    #                NONE is all, SIMPLE works well for straight edges, Teh-Chin for lossy fitting of a curvy contour
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # contours is a list of regions, each region is a list of boundary coordinates
    # hierarchy is a list of region metadata: [next sibling ID, previous sibling ID, child ID, parent ID]

    # Additional edge plot for seeing top n classifications
    # contours = sorted(contours, key=lambda cont: cv2.contourArea(cont), reverse=True)[:5]
    #
    # tempmask = np.zeros(edges.shape)
    # for idx, contour in enumerate(contours):
    #     cv2.fillPoly(tempmask, contour, 1 - 1. / len(contours) * (idx + 1))
    # cv2.imshow('mask', tempmask)

    # Catch a few cases where segmentation breaks down
    if not contours:
        return img

    # Pick the contour with the greatest area, tends to represent the clothing item
    max_contour = max(contours, key=lambda cont: cv2.contourArea(cont))
    if not (.20 < cv2.contourArea(max_contour) / np.prod(img.shape[:2]) < .80):
        return img

    # Create empty mask, draw filled polygon on it corresponding to largest contour
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour, 255)

    # Smooth mask, then blur it
    mask = cv2.dilate(mask, None, iterations=10)
    mask = cv2.erode(mask, None, iterations=10)

    # Catch another case where segmentation breaks down
    border_size = np.sum(img.shape[:2] * 2) - 2
    border_coverage = border_size - (np.sum(mask[-1:] + mask[:1]) + np.sum(mask[:, -1:] + mask[:, :1])) / 255
    if (border_coverage / border_size) < .6:
        return img

    # Checks have passed, now apply the mask to the image
    blur_radius = 5
    mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

    mask = np.dstack([mask] * 3).astype('float32') / 255.0
    return (mask * img.astype('float32')).astype('uint8')


def canny_is_quality(img):

    # Operate on greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 10, 200)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, None)

    # At this stage, the image is b/w, with white along edges, black within regions

    # ~~~ Now find regions, which are enclosed by contours:
    # Retrieval mode - consider landlocked contours children. TREE preserves hierarchy, LIST flattens
    # Chain approx - how many points used along edge of region.
    #                NONE is all, SIMPLE works well for straight edges, Teh-Chin for lossy fitting of a curvy contour
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # contours is a list of regions, each region is a list of boundary coordinates
    # hierarchy is a list of region metadata: [next sibling ID, previous sibling ID, child ID, parent ID]

    if not contours:
        return False

    max_contour = max(contours, key=lambda cont: cv2.contourArea(cont))
    if not (.23 < cv2.contourArea(max_contour) / np.prod(img.shape[:2]) < .75):
        return False

    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour, 255)

    mask = cv2.dilate(mask, None, iterations=10)
    mask = cv2.erode(mask, None, iterations=10)

    border_size = np.sum(img.shape[:2] * 2) - 2
    border_coverage = border_size - (np.sum(mask[-1:] + mask[:1]) + np.sum(mask[:, -1:] + mask[:, :1])) / 255

    if (border_coverage / border_size) < .7:
        return False
    return True


if __name__ == '__main__':

    def test_canny():
        while True:
            # Proof of concept, this can be made more efficient
            record = list(db.ebay.aggregate([{"$sample": {"size": 1}}]))[0]
            print(record['title'])
            pil_image = Image.open(BytesIO(record['image']))
            img = canny_mask(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))

            cv2.imshow("masked", img)
            cv2.waitKey()

    def quality_check():
        count = 0
        for record in db.ebay.find({}):
            count += 1
            sys.stdout.write("\r\x1b[KProcessing: " + str(count))
            sys.stdout.flush()

            # Proof of concept, this can be made more efficient
            pil_image = Image.open(BytesIO(record['image']))
            if canny_mask(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)):
                db.ebay.update({'image_url': record['image_url']}, {"$set": {"quality": True}})


    # Tag database with the results of the canny segmentation
    # quality_check()

    # Mask some images with canny
    test_canny()
