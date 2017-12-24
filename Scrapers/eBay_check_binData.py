from pymongo import MongoClient
from io import BytesIO
from PIL import Image
import numpy as np

import sys

from Tagger.image_formatter import preprocess

# Run once after scraping to remove records with invalid binary data

mongo_client = MongoClient(host='localhost', port=27017)  # Default port
db = mongo_client.deep_fashion


count = 0
for document in db.ebay.find({}):
    try:
        shape = np.shape(np.array(preprocess(Image.open(BytesIO(document['image'])))))
        if shape != (224, 224, 3):
            count += 1
            # Show status
            sys.stdout.write("\r\x1b[KRemoved: " + str(count))
            sys.stdout.flush()

            db.ebay.remove({'image_url': document['image_url']})

    except Exception as err:
        print("\nException: " + document['image_url'])
        print(err)

        count += 1
        db.ebay.remove({'image_url': document['image_url']})
