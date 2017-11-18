import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

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

# for root, dirs, filenames in os.walk('./samples/'):
#     for f in filenames:
#         fullpath = root + '\\' + f

path = './samples/purple.jpg'
# sobel_watershed(path)
felzenszwalb_segment(path)
