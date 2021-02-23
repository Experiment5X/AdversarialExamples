import sys
import numpy as np
from PIL import Image

if len(sys.argv) < 4:
    print('Invalid params')
    exit(-1)

image1 = np.array(Image.open(sys.argv[1]), dtype=float)
image2 = np.array(Image.open(sys.argv[2]), dtype=float)

print(f'Image 1: {image1.shape}    Image 2: {image2.shape}')

diff_np = image2 - image1
print(f'Max diff: {diff_np.max()}')
diff_scaled_np = (diff_np - diff_np.min()) / (diff_np.max() - diff_np.min()) * 255
image_diff = Image.fromarray(np.uint8(diff_scaled_np))

image_diff.save(sys.argv[3])

print('Wrote diff')
