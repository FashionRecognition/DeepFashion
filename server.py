from flask import Flask
from flask import request
from PIL import Image
import numpy as np

app = Flask(__name__)

# To send a POST with image in postman:
# https://stackoverflow.com/questions/39660074/post-image-data-using-postman

# This must be set to the value sent in the post.
with open('api_key.txt', 'r') as keyfile:
    api_key = keyfile.readline().strip()


@app.route('/listings/', methods=['POST'])
def listings():

    # Ensure key is valid
    if request.values['api_key'] != api_key:
        return str({'error': 'invalid api key'})

    img = Image.open(request.files['image'])
    print(np.array(img))

if __name__ == '__main__':
    app.run(port=8888)
