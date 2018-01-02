from pymongo import MongoClient
from io import BytesIO

import numpy as np
from PIL import Image

# openCV package is distributed as a binary, so references won't resolve
import cv2


# Prep a training image with standard dimensions and no backing
def preprocess(im, target_dims, debug=False):
    im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    mask, (y, x, h, w) = canny_mask(im, debug)

    # Apply mask, then slice to bounding box of mask
    mask = np.dstack([mask] * 3).astype('float32') / 255.0
    im = (mask * im.astype('float32')).astype('uint8')[x:x+w, y:y+h]

    # Convert back to PIL
    im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 'RGB')

    width, height = im.size
    aspect_ratio = width / height
    target_ratio = target_dims[0] / target_dims[1]

    if aspect_ratio != target_ratio:
        # If image is wider than target dims, add a black bar to the bottom
        if aspect_ratio > target_ratio:
            background = Image.new('RGB', (width, target_dims[1]), (0, 0, 0))
            background.paste(im, (0, 0))

            im = background

        # Image is thinner than target dims, crop out top/bottom bars
        else:
            mult = width / target_dims[0]
            # The top of the image is more important, so we won't crop to center
            # y_off = int((height - target_dims[1] * mult) / 2)
            im = im.crop((0, 0, target_dims[0] * mult, target_dims[1] * mult))

    # Reduce size
    im = im.resize(target_dims, Image.NEAREST)

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


def canny_mask(img, debug=False):
    if debug:
        cv2.imshow("original", img)
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
    if debug:
        contours = sorted(contours, key=lambda cont: cv2.contourArea(cont), reverse=True)[:5]

        tempmask = np.zeros(edges.shape)
        for idx, contour in enumerate(contours):
            cv2.fillPoly(tempmask, [contour], 1 - 1. / len(contours) * (idx + 1))
        cv2.imshow('mask', tempmask)

    # Catch a few cases where segmentation breaks down
    if not contours:
        if debug:
            raise ValueError("No contours detected")
        else:
            return np.ones(img.shape[:2]) * 255, (0, 0, *img.shape[:2])

    # Pick the contour with the greatest area, tends to represent the clothing item
    max_contour = max(contours, key=lambda cont: cv2.contourArea(cont))
    if not (.20 < cv2.contourArea(max_contour) / np.prod(img.shape[:2]) < .80):
        if debug:
            raise ValueError("Detected poor area coverage")
        else:
            return np.ones(img.shape[:2]) * 255, (0, 0, *img.shape[:2])

    # Create empty mask, draw filled polygon on it corresponding to largest contour
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillPoly(mask, [max_contour], 255)

    # Smooth mask, then blur it
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=2)

    # Catch another case where segmentation breaks down
    border_size = np.sum(img.shape[:2] * 2) - 2
    border_coverage = border_size - (np.sum(mask[-1:] + mask[:1]) + np.sum(mask[:, -1:] + mask[:, :1])) / 255
    if (border_coverage / border_size) < .6:
        if debug:
            raise ValueError("Detected poor border coverage")
        else:
            return np.ones(img.shape[:2]) * 255, (0, 0, *img.shape[:2])

    # First remove some fine details from the mask
    blur_radius = 25
    macro = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
    macro[macro < 128] = 0
    macro[macro > 0] = 1

    blur_radius = 5
    mask = cv2.GaussianBlur(mask * macro, (blur_radius, blur_radius), 0)

    # Find bounding box of mask
    nonzero = cv2.findNonZero(mask.astype(np.uint8))
    return mask, cv2.boundingRect(nonzero)


if __name__ == '__main__':

    mongo_client = MongoClient(host='localhost', port=27017)  # Default port
    db = mongo_client.deep_fashion

    def test_canny():
        while True:
            # Proof of concept, this can be made more efficient
            record = list(db.ebay.aggregate([{"$sample": {"size": 1}}]))[0]
            print(record['title'])
            pil_image = Image.open(BytesIO(record['image']))

            try:
                preprocess(pil_image, (192, 256), debug=True)
                cv2.waitKey()
            except ValueError as err:
                print(err)

    # Mask some images with canny
    test_canny()
