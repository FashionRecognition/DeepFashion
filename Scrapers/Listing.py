class Listing(object):
    source = None
    name = None

    def get_json(self):
        return {'url': self.source, 'name': self.name}
