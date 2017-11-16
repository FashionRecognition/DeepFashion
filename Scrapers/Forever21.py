import json
import urllib.request
from bs4 import BeautifulSoup
from pymongo import MongoClient

mongo_client = MongoClient(host='localhost', port=27017)  # Default port
db = mongo_client.deep_fashion

category_url = "https://www.forever21.com/us/shop/Catalog/Category/f21/top_blouses"


def parse_category(url):
    # Load the html webpage into a text variable
    webpage = urllib.request.urlopen(url).read().decode("utf8")

    # Iterate through each line, by splitting the string into a list by newline
    for line in webpage.split('\n'):

        # Ignore the line if it isn't the one that assigns to the cData javascript variable
        if 'var cData = ' not in line:
            continue

        # Parse the json that would have been assigned in javascript
        listings = json.loads(line.split('var cData = ')[1][:-2])

        # The JSON stores all the product listings under the CatalogProducts key
        for product in listings['CatalogProducts']:

            # Parse the product page for product in the listing
            parse_product(product['ProductShareLinkUrl'])


def parse_product(url):
    webpage = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(webpage, "lxml")

    for line in webpage.decode("utf8").split('\n'):

        # Ignore the line if it isn't the one that assigns to the cData javascript variable
        if 'var pData = ' not in line:
            continue

        # Parse the json that would have been assigned in javascript
        product = json.loads(line.split('var pData = ')[1][:-2])

        listing = {
            'url': url,
            'vendor': 'Forever21',
            'name': product['DisplayName'],
            'price': product['ListPrice'],
            'colors': [variant['ColorName'] for variant in product['Variants']],
            'gender': product['ProductSizeChart'],
            'image': soup.find("meta", property="og:image")['content']
        }

        # Insert if url is not already listed
        if not list(db.products.find({'url': url})):
            db.products.insert_one(listing)
            print(listing)
        else:
            print("Product already catalogued: " + listing['name'])

parse_category(category_url)
