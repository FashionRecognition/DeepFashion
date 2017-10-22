import re

from scrapy.spiders import CrawlSpider
from scrapy.http import Request
from scrapy.selector import HtmlXPathSelector

from Scrapers.Listing import Listing

from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.aggregation


# This is a partially converted skeleton for scraping Forever21.


class Forever21Spider(CrawlSpider):
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

        'parse_product_product_name': '//span[@itemprop="name"]/text()',
        'parse_product_product_number_deal': '//div[@id="pd_deal"]//input[@name="pID"]/@value',
        'parse_product_product_number_img': '//div[@id="pd_img"]//input[@name="pID"]/@value',
        'parse_product_product_number_img_add2wl': '//span[@class="add2wl"]/@onclick',
        'parse_product_product_number_share': '//div[@id="pd_deal"]//li[@class="email"]/a/@href',
        'parse_product_description': '//div[@id="pd_desc"]//text()',
        'parse_product_categories': '//div[@id="pd_bcr"]/ul//li/*/text()',
        'parse_product_image_url': '//img[@id="pd_imgtag"]/@src',
        'parse_product_availability': '//div[@id="pd_deal"]//p[@class="stock"]/strong/text()',
        'parse_product_sale_price': '//span[@itemprop="price"]//text()',
        'parse_product_on_sale_save': '//div[@class="midcart_widget"]//li[@class="save"]',
        'parse_product_on_sale_img': '//div[@id="pd_img"]//div[contains(@class, "nl-promo")]',

    }

    AVAIL_CHOICES = {
        # Add more entries here:
        'name': Listing.name,
        'url': Listing.source
    }

    def extract_xpath(self, hxs, name_xpath):
        xpath = self.xpaths[name_xpath]
        return hxs.select(xpath).extract()

    def parse(self, response):
        hxs = HtmlXPathSelector(response)
        products = hxs.select(self.xpaths['parse_product']).extract()
        print(products)
        for product in products:
            category_page = self.store_url + product
            yield Request(category_page, callback=self.parse_product)
        #
        # next_page = hxs.select(self.xpaths['parse_category_next']).extract()
        # if next_page:
        #     next_page = self.store_url + str(next_page[0])
        #     yield Request(next_page, callback=self.parse)

    def parse_product(self, response):
        if response.url == self.store_url + '/':
            return

        hxs = HtmlXPathSelector(response)
        item = Listing()

        # Source
        item['source'] = self.store_url

        # Product Name
        tmp = self.extract_xpath(hxs, 'parse_product_product_name')
        if len(tmp) != 1:
            raise ValueError('No Product Name')
        item['product_name'] = tmp[0]

        # Actually insert to database
        db.records.insert_one(item.get_json())
