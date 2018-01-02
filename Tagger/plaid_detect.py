import matplotlib.pyplot as plt

from pymongo import MongoClient
import numpy as np
from PIL import Image
from io import BytesIO
import cv2

from Tagger.image_formatter import canny_mask

mongo_client = MongoClient(host='localhost', port=27017)  # Default port
db = mongo_client.deep_fashion


def plaid_detection(image):
    processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    lines = cv2.HoughLinesP(thresh.astype(np.uint8), 1.2, np.pi / 180, 200)

    thresh_rgb = np.transpose(np.array([thresh] * 3), axes=(1, 2, 0)).copy()

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(thresh_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

    fig, axes = plt.subplots(nrows=2, figsize=(7, 8))
    ax0, ax1 = axes
    plt.gray()

    ax0.imshow(image)
    ax0.set_title('Image')

    ax1.imshow(thresh_rgb)
    ax1.set_title('Local thresholding')

    for ax in axes:
        ax.axis('off')

    plt.show()


if __name__ == '__main__':

    mongo_client = MongoClient(host='localhost', port=27017)  # Default port
    db = mongo_client.deep_fashion

    def test_canny():
        while True:
            # Proof of concept, this can be made more efficient
            record = list(db.ebay.aggregate([{"$match": {"pattern": "plaid"}},
                                             {"$sample": {"size": 1}}]))[0]
            print(record['title'])
            pil_image = Image.open(BytesIO(record['image']))

            try:
                # Compute the mask and bounds
                mask, (x, y, w, h) = canny_mask(np.array(pil_image))

                # Apply mask, then slice to bounding box of mask
                mask = np.dstack([mask] * 3).astype('float32') / 255.0
                im = (mask * np.array(pil_image).astype('float32')).astype('uint8')[x:x + w, y:y + h]

                # Convert back to PIL, run plaid detection
                plaid_detection(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            except ValueError as err:
                print(err)

    # Mask some images with canny
    test_canny()
