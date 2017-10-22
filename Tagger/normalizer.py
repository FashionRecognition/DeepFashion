import numpy as np
from PIL import Image
import shutil
import re

import os
root_directory = r'C:\Users\mike_000\Subdir\\'
indir = 'Fashion_Subset'
outdir = 'Fashion_Subset_Processed'

if os.path.exists(root_directory + outdir):
    shutil.rmtree(root_directory + outdir)
if not os.path.exists(root_directory + outdir):
    os.mkdir(root_directory + outdir)

for root, dirs, filenames in os.walk(root_directory + indir):
    for f in filenames:
        fullname = root + '\\' + f
        fullname_png = root.replace(indir, outdir) + '\\' + re.sub('[jpg]{3}$', 'png', f)

        # Change filetype
        im = Image.open(fullname)
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

        if not os.path.exists(root.replace(indir, outdir)):
            os.mkdir(root.replace(indir, outdir))

        im.save(fullname_png)
        # Image.fromarray(colors, 'RGB').save(fullname_png)
