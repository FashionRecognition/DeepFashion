import re

from scrapy.spiders import CrawlSpider
from scrapy.http import Request
from scrapy.selector import HtmlXPathSelector

from Scrapers.Listing import Listing

from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.aggregation


# This is a partially converted skeleton for scraping Forever21.


class Forever21SpiderSmall(CrawlSpider):
    name = "forever21"
    store_url = "https://www.forever21.com"
    start_urls = [
        store_url + "/us/shop/Catalog/Category/21men/mens-tops",
    ]

    # This extracts a float number from a string
    re_float = re.compile('[-+]?[0-9]*\\.?[0-9]+')

    xpaths = {
        'parse_product': '//div[@id="products"]',
        'parse_category_next': '//div[@id="h1Title"]/',
    }

    AVAIL_CHOICES = {
    }

    def extract_xpath(self, hxs, name_xpath):
        xpath = self.xpaths[name_xpath]
        return hxs.select(xpath).extract()

    def parse(self, response):
        hxs = HtmlXPathSelector(response)
        products = hxs.select(self.xpaths['parse_product']).extract()
        print(products)
        for product in products:
            pass
