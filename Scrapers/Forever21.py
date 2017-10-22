import re

from scrapy.spiders import CrawlSpider
from scrapy.http import Request
from scrapy.selector import HtmlXPathSelector

from Scrapers.Listing import Listing

from pymongo import MongoClient

import logging

client = MongoClient('localhost', 27017)
db = client.aggregation

logging.getLogger('scrapy').setLevel(logging.WARNING)


# This is a partially converted skeleton for scraping Forever21.


class Forever21Spider(CrawlSpider):
    name = "forever21"
    store_url = "https://www.forever21.com"
    start_urls = [
        store_url + "/us/shop/Catalog/Product/21men/mens-tops/2000216764",
    ]

    # This extracts a float number from a string
    re_float = re.compile('[-+]?[0-9]*\\.?[0-9]+')

    xpaths = {
        'parse_product': '//div[@id="products"]',
        'parse_category_next': '//div[@id="h1Title"]/',

        'parse_product_name': 'id(\'h1Title\')/text()[1]',
        'parse_product_price': 'id(\'ItemPrice\')/child::node()[1]',
        'parse_product_color': 'id(\'selectedColorName\')/child::node()'

    }

    def extract_xpath(self, hxs, name_xpath):
        xpath = self.xpaths[name_xpath]
        return hxs.select(xpath).extract()

    # def parse(self, response):
    #     hxs = HtmlXPathSelector(response)
    #     products = hxs.select(self.xpaths['parse_product']).extract()
    #     for product in products:
    #         category_page = self.store_url + product
    #         yield Request(category_page, callback=self.parse_product)
    #
    #     next_page = hxs.select(self.xpaths['parse_category_next']).extract()
    #     if next_page:
    #         next_page = self.store_url + str(next_page[0])
    #         yield Request(next_page, callback=self.parse)

    def parse(self, response):
        if response.url == self.store_url + '/':
            return

        hxs = HtmlXPathSelector(response)
        item = Listing()

        # Source
        item.source = self.store_url

        # Product Name
        item.name = self.extract_xpath(hxs, 'parse_product_name')[0]
        print(self.extract_xpath(hxs, 'parse_product_price'))
        item.price = self.extract_xpath(hxs, 'parse_product_price')[0]

        print(item.price)

        # Actually insert to database
        db.records.insert_one(item.get_json())
