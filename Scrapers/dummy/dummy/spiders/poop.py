import scrapy


class DummyScrape(scrapy.Spider):
    name = 'dummy'

    start_urls = [
        'https://stackoverflow.com/questions/40411746/scraping-through-every-product-on-retailer-website',
        ]

    def parse(self, response):
        for fw in response.css("user-details"):
            print(fw)
            # yield {
            #     'user-details' : user-info.css("div.user-details a")
            # }

