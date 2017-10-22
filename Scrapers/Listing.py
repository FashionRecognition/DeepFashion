class Listing(object):
    url = None
    url_image = None
    name = None



    def get_json(self):
        return {
            'url': self.url,
            'url_image': self.url_image,
            'url'
            'name': self.name
        }
