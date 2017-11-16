import re

from scrapy.spiders import CrawlSpider
from scrapy.http import Request
from scrapy import Selector

from pymongo import MongoClient

from scrapy.crawler import CrawlerProcess

client = MongoClient('localhost', 27017)
db = client.aggregation


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
        'parse_product_price': 'id(\'ItemPrice\')/child::node()',
        'parse_product_color': 'id(\'selectedColorName\')/child::node()'
    }

    def extract_xpath(self, hxs, name_xpath):
        xpath = self.xpaths[name_xpath]
        return hxs.xpath(xpath).extract()

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
        hxs = Selector(response)

        listing = {
            'url': self.store_url,
            'name': self.extract_xpath(hxs, 'parse_product_name')[0],
            'price': Request(response.url, callback=self.parse_price)
        }

        # Actually insert to database
        db.records.insert_one(listing)

    def parse_price(self, response):
        selection = Selector(response)
        print(self.extract_xpath(selection, 'parse_product_price'))
        return self.extract_xpath(selection, 'parse_product_price')


if __name__ == "__main__":
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })

    process.crawl(Forever21Spider)
    process.start()
