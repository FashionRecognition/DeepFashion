import requests
import sys
import time

from multiprocessing import Queue, Process, cpu_count

import bs4

from pymongo import MongoClient

from io import BytesIO
from bson.binary import Binary
# from PIL import Image

mongo_client = MongoClient(host='localhost', port=27017)  # Default port
db = mongo_client.deep_fashion

listing_url = 'https://www.ebay.com/sch/i.html?_nkw=used+womens+clothes&_sop=10&_pgn='


def scrape_page(page_number):
    soup = bs4.BeautifulSoup(requests.get(listing_url + str(page_number)).content, "lxml")

    # Iterate through each product listing
    for listing in soup.find("ul", {"id": "GalleryViewInner"}):

        # Some elements are not for products. Just ignore the error
        try:
            # Isolate the image div
            element = bs4.BeautifulSoup(str(listing), "lxml").div.div.div.a.img

            # Check if record is already in database
            if not list(db.ebay.find({'image_url': element['src']})):

                # Example conversion from url > request > bytestream > binary > bytestream > image
                # The binary stage is stored in mongodb
                # Image.open(BytesIO(Binary(BytesIO(requests.get(image_url).content).getvalue()))).show()

                # Decode binary via the following:
                # Image.open(BytesIO(listing['image']))

                listing = {'image_url': element['src'],
                           'title': element['alt'],
                           'image': Binary(BytesIO(requests.get(element['src']).content).getvalue())}

                db.ebay.insert_one(listing)

        except AttributeError:
            pass


def scrape():
    page_queue = Queue()

    # The first 10,000 results are accessible on ebay. That makes 208 pages
    for page in reversed(range(208)):
        page_queue.put(page)

    pool = [Process(target=process_wrapper, args=(page_queue,), name=str(proc))
            for proc in range(cpu_count())]

    for proc in pool:
        proc.start()

    while any([proc.is_alive() for proc in pool]):

        # Show status
        sys.stdout.write("\r\x1b[KCollecting: " + str(208 - page_queue.qsize()) + '/' + str(208))
        sys.stdout.flush()
        time.sleep(0.5)

    # Once the pool of pages to scrape has been exhausted, each thread will die
    # Once the threads are dead, this terminates all threads and the program is complete
    for proc in pool:
        proc.terminate()


def process_wrapper(page_queue):
    # Take elements from the queue until the queue is exhausted
    while not page_queue.empty():
        page_id = page_queue.get()

        success = False
        while not success:
            try:
                scrape_page(page_id)
                success = True
            except Exception as err:
                print(err)
                print("Re-attempting page " + str(page_id))

            # Be nice to their servers
            time.sleep(1)

# Only call scrape when invoked from main. This prevents forked processes from calling it
if __name__ == '__main__':
    scrape()
