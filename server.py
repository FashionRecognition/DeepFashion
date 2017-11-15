from flask import Flask
from flask import request

from PIL import Image
from pymongo import MongoClient
from bson import json_util
from json import dumps

from Tagger.FashionNet import FashionNet

# Initialize globals
app = Flask(__name__)
network = FashionNet()

# Set up mongo
# If running locally, be sure to create a database and collection--
#   Run the following commands in the mongo.exe shell:
#       use deep_fashion
#       db.createCollection("listings")
#   Optional, inserts one record manually:
#       db.listings.insert({"attributes": ["fur", "light"], "url": "https://shop.nordstrom.com/s/4810512"})

mongo_client = MongoClient(host='localhost', port=27017)  # Default port
db = mongo_client.deep_fashion

# To send a POST with image in postman:
# https://stackoverflow.com/questions/39660074/post-image-data-using-postman

# Remember to create a text file with the api key sent in the post.
with open('api_key.txt', 'r') as keyfile:
    api_key = keyfile.readline().strip()


@app.route('/listings/', methods=['POST'])
def listings():

    # Ensure key is valid
    if request.values['api_key'] != api_key:
        return str({'error': 'invalid api key'})

    # Get attributes from image
    img = Image.open(request.files['image'])
    attributes = network.get_attributes(img)

    # Construct query
    query = {
        'attributes': {
            '$all': attributes
        }
    }

    # actually query Mongo
    # https://docs.mongodb.com/manual/reference/operator/query/
    records = list(db.listings.find(query))

    # Convert to json string and return
    return dumps(records, default=json_util.default)


if __name__ == '__main__':
    app.run(port=8888)
