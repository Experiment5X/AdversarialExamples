import sys
from PIL import Image

from common import predict

if __name__ == '__main__':
    image_path = sys.argv[1]
    image = Image.open(image_path)

    print(predict(image))
