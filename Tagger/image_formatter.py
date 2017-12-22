import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from PIL import Image

# Sobel filtering
from scipy import ndimage as ndi
from skimage import filters
from skimage.color import rgb2gray
from skimage import morphology
from scipy.stats import norm

# Felzenswalb
from skimage.segmentation import felzenszwalb

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.dpi'] = 200

# 1. Spectral decomposition seems to be completely useless. I removed it.
# 2. Sobel watershed provides decent segmentation for obvious cases. It tends to preserve patterning on shirts though
# 3. Felzenswalb tends to break up consistent segments into multiple pieces. Fails dramatically on strong patterns


def sobel_watershed(path):
    image = rgb2gray(imread(path))
    blurred = filters.gaussian(image, sigma=3.0)
    sobel = filters.sobel(blurred)

    # The values for seeds were difficult to estimate, they rely on each picture
    # I tried using a confidence interval for the normal distribution as a heuristic for these bounds
    confidence = .7
    mu = np.mean(image)
    sigma = np.std(image)
    lower, upper = norm.interval(confidence, loc=mu, scale=sigma)

    # Alternatively take top n as seeds. This didn't work too well
    # seeds = 50
    # lower = bn.partition(image.flatten(), seeds)[seeds]
    # upper = -bn.partition(-image.flatten(), seeds)[seeds]

    light_spots = np.array((image > upper).nonzero())
    dark_spots = np.array((image < lower).nonzero())

    bool_mask = np.zeros(image.shape, dtype=np.bool)
    bool_mask[tuple(light_spots)] = True
    bool_mask[tuple(dark_spots)] = True
    seed_mask, num_seeds = ndi.label(bool_mask)
    print(num_seeds)

    ws = morphology.watershed(sobel, seed_mask)
    plt.imshow(ws)

    # plt.imshow(image)
    plt.show()


def felzenszwalb_segment(path):
    # Keep the kernel (sigma) small
    plt.imshow(felzenszwalb(imread(path), scale=4.0, sigma=0.4, min_size=1500))
    plt.show()


# A training image should have standard 224x224 dimensions
def preprocess(im):
    dimensions = min(im.size)

    # Width is smaller dimension
    if im.size[0] == dimensions:
        xoff = 0
        yoff = (im.size[0] - dimensions) / 2
    else:
        xoff = (im.size[1])
        yoff = 0

    # Reduce size
    im = im.crop((xoff, yoff, dimensions, dimensions))
    im = im.resize((200, 200), Image.NEAREST)

    # Normalize the image
    # colors = np.array(im).astype(float)
    # colors -= 128
    # colors /= 256
    # print(colors)
    # colors = (colors - np.mean(colors, axis=2)[..., None]) / np.std(colors, axis=2)[..., None]
    # colors *= 256
    # colors += 128
    # colors = np.clip(colors, 0, 255)

    # Image.fromarray(colors, 'RGB').save(fullname_png)
    return im

# path = './samples/purple.jpg'
# sobel_watershed(path)
# felzenszwalb_segment(path)
