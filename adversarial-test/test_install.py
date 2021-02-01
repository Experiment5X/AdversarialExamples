from PIL import Image
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from labels import label_lookup

mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
model = models.vgg16(pretrained=True).eval()

loader = transforms.Compose([transforms.Scale((224, 224)), transforms.ToTensor()])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def predict(image):
    image_converted = image.convert('RGB')
    image_loaded = loader(image_converted).float()
    image_normalized = normalize(image_loaded)

    outputs = model.forward(image_normalized.unsqueeze(0))
    label_index = int(outputs.argmax().numpy())

    return label_lookup[label_index]


image = Image.open('/Users/adamspindler/Downloads/g.png')
if image is None:
    print('no image')
    exit(-1)

print('prediction: ', predict(image))

