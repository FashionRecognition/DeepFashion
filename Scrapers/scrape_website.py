from scrapy.selector import HtmlXPathSelector
from scrapy.http import Request
from scrapy.contrib.spiders import CrawlSpider

from ..Listing import Listing

import re

# This is a partially converted skeleton for scraping Forever21.


class Forever21Spider(CrawlSpider):
    name = "forever21"
    store_url = "https://www.forever21.com/us/shop"
    start_urls = [
        store_url,
    ]

    # This extracts a float number from a string
    re_float = re.compile('[-+]?[0-9]*\\.?[0-9]+')

    xpaths = {
        'parse_categories': '//div[@id="mlc"]//ul[@class="cat"]//li//a/@href',

        'parse_category_sub': '//div[@id="mlc"]//ul[@class="sub"]//li//a/@href',
        'parse_category_products': '//div[@class="prod"]//h2[@itemprop]/a/@href',
        'parse_category_next': '//div[@class="pag "]/ul/li[@class="textnav"]/a[@rel="next"]/@href',

        'parse_sub_category_products': '//div[@class="prod"]//h2[@itemprop]/a/@href',
        'parse_sub_category_next': '//div[@class="pag "]/ul/li[@class="textnav"]/a[@rel="next"]/@href',

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
        categories = hxs.select(self.xpaths['parse_categories']).extract()
        for category_page in categories:
            category_page = self.store_url + category_page
            yield Request(category_page, callback=self.parse_category)

    def parse_category(self, response):
        hxs = HtmlXPathSelector(response)
        categories = hxs.select(self.xpaths['parse_category_sub']).extract()
        for category_page in categories:
            category_page = self.store_url + category_page
            yield Request(category_page, callback=self.parse_sub_category)

        products = hxs.select(self.xpaths['parse_category_products']).extract()
        for product in products:
            product_page = self.store_url + product
            yield Request(product_page, callback=self.parse_product)

        next_page = hxs.select(self.xpaths['parse_category_next']).extract()
        if next_page:
            next_page = self.store_url + str(next_page[0])
            yield Request(next_page, callback=self.parse_category)

    def parse_sub_category(self, response):
        hxs = HtmlXPathSelector(response)
        products = hxs.select(self.xpaths['parse_sub_category_products']).extract()
        for product in products:
            product_page = self.store_url + product
            yield Request(product_page, callback=self.parse_product)

        next_page = hxs.select(self.xpaths['parse_sub_category_next']).extract()
        if next_page:
            next_page = self.store_url + str(next_page[0])
            yield Request(next_page, callback=self.parse_category)

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

        # Product Number
        tmp = self.extract_xpath(hxs, 'parse_product_product_number_deal')  # In Stock
        if len(tmp) == 0:
            tmp = self.extract_xpath(hxs, 'parse_product_product_number_img')  # Out of Stock
            if len(tmp) == 0:
                tmp = self.extract_xpath(hxs, 'parse_product_product_number_img_add2wl')  # Out of Stock and No Image
                if len(tmp) == 0:
                    tmp = self.extract_xpath(hxs,
                                             'parse_product_product_number_share')  # Out of Stock, No Image and no Add2wl
                    if len(tmp) == 0:
                        raise ValueError('No Product Number')

                    re_share = re.compile('m=tell&p=(\d+)')
                    ms = re_share.search(tmp[0])
                    tmp = ms.groups()
                    if len(tmp) == 0:
                        raise ValueError('No Product Number')
                else:
                    re_add2wl = re.compile('pid=(\d+)')
                    ms = re_add2wl.search(tmp[0])
                    tmp = ms.groups()
                    if len(tmp) == 0:
                        raise ValueError('No Product Number')

        item['product_number'] = tmp[0]

        # Description
        tmp = self.extract_xpath(hxs, 'parse_product_description')
        if len(tmp) is 0:
            raise ValueError('No Description')
        else:
            item['description'] = '\n'.join(map(lambda s: s.strip(), tmp)).strip()

        # Category Name
        tmp = self.extract_xpath(hxs, 'parse_product_categories')
        if len(tmp) <= 0:
            raise ValueError('No Categories')

        cg_paths = []
        cg_path = []

        for c in tmp:
            c = c.strip()
            if c == '':
                continue
            elif c == 'Home':
                cg_path = Listing.CG_PATH_SEP.join(cg_path)
                if cg_path != '':
                    cg_paths.append(cg_path)
                cg_path = ['Home']
            else:
                cg_path.append(c)

        cg_paths.append(Listing.CG_PATH_SEP.join(cg_path))
        item['category_name'] = Listing.CG_PATHS_SEP.join(cg_paths)

        # Product URL
        item['product_url'] = response.url

        # Image URL
        tmp = self.extract_xpath(hxs, 'parse_product_image_url')
        if len(tmp) is 0:
            raise ValueError('No Image URL')
        else:
            item['image_url'] = self.extract_xpath(hxs, 'parse_product_image_url')[0]

        # Product Condition
        item['product_condition'] = Listing.PC_NEW

        # Availability
        tmp = self.extract_xpath(hxs, 'parse_product_availability')
        if len(tmp) != 1:
            raise ValueError('No Availability')
        else:
            tmp = self.AVAIL_CHOICES.get(re.sub('\s+', '', tmp[0].lower()))
            if not tmp:
                raise ValueError('No such Availability')
            item['availability'] = tmp

        # Sale Price
        tmp = self.extract_xpath(hxs, 'parse_product_sale_price')
        tmp = re.sub('[$|\s]', '', ''.join(tmp))
        item['sale_price'] = float(tmp)

        # On Sale
        item['on_sale'] = 0
        if len(self.extract_xpath(hxs, 'parse_product_on_sale_img')) > 0 or len(
                self.extract_xpath(hxs, 'parse_product_on_sale_save')) > 0:
            item['on_sale'] = 1

        # Currency
        item['currency'] = 'AUD'

        # Manufacturer
        item['manufacturer'] = ''

        # Shipping Cost
        # Generate a Request to get the Shipping Cost
        request = Request(self.SC_URL % (item['product_number']), callback=self.parse_shipping_cost, dont_filter=True)
        request.meta['item'] = item

        return request
