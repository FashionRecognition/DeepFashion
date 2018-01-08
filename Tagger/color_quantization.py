from pymongo import MongoClient
import numpy as np
from PIL import Image
import cv2
from io import BytesIO

import json

from Tagger.image_formatter import canny_mask

if __name__ == '__main__':
    k_colors = 3
    mongo_client = MongoClient(host='localhost', port=27017)  # Default port
    db = mongo_client.deep_fashion


    def quantize():
        while True:
            record = list(db.ebay.aggregate([{"$match": {"pattern": "plaid"}},
                                             {"$sample": {"size": 1}}]))[0]
            print(record['title'])

            pil_image = Image.open(BytesIO(record['image']))
            cv2_image = np.array(pil_image)[..., ::-1]

            cv2.imshow("original", cv2_image)
            mask, (y, x, h, w) = canny_mask(cv2_image)

            mask = cv2.erode(mask, None, iterations=10)

            # Apply mask, then slice to bounding box of mask
            mask = np.dstack([mask] * 3).astype('float32') / 255.0
            arr_image = (mask * np.array(pil_image).astype('float32')).astype('uint8')[x:x + w, y:y + h]

            pil_image = Image.fromarray(arr_image, 'RGB').convert('P', palette=Image.ADAPTIVE, colors=k_colors)
            colors = [pil_image.getpalette()[off * 3:(off + 1) * 3] for off in range(k_colors)]

            for color in colors:
                text_color = color_name(color)

                if text_color != 'black':
                    print(color)
                    print(''.join([hex(channel)[-2:] for channel in color]))
                    print(text_color)
            print('\n\n')

            cv2.imshow("palettized", np.array(pil_image.convert("RGB"))[..., ::-1])
            # pil_image.show()
            cv2.waitKey()


    def color_name(rgb_source):

        with open('colors.json', 'r') as colorfile:
            colors_json = json.load(colorfile)

        best_color = ''
        min_distance = 100000

        # Compute distance from each color
        for color, hexval in colors_json.items():
            rgb_label = np.array([int(hexval[:2], 16), int(hexval[2:4], 16), int(hexval[4:], 16)])
            distance = ((rgb_source - rgb_label)**2).sum()

            # Check for acceptance
            if distance < min_distance:
                min_distance = distance
                best_color = color

        return best_color

    quantize()
