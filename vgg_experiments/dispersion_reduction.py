# Idea from this paper: https://arxiv.org/abs/1911.11616
import sys
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.nn import Identity, Sequential

from common import CrossEntropyLoss, loader, inv_normalize, model, normalize, predict


def reduce_dispersion(image, target_layer_index=14):
    # create the model chopped off at the desired feature map layer
    vgg_model = Sequential(
        *[model.features[i] for i in range(0, target_layer_index + 1)]
    )

    image_tensor = loader(image).float()
    orig_image_tensor = loader(image).float()

    print('Starting iterations...')
    for i in range(0, 25):
        x = image_tensor
        x.requires_grad = True

        x_normalized = normalize(x).unsqueeze(0)
        output = vgg_model.forward(x_normalized)

        adversarial_loss = torch.std(output)
        adversarial_loss.backward()

        image_tensor = x.data - 3 * x.grad.data

        # need to normalize to keep the pixel values between 0 and 1
        # could also clip probably
        image_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min()
        )

        image_class = (
            model.forward(normalize(image_tensor).unsqueeze(0))
            .detach()
            .numpy()
            .argmax()
        )
        print(
            f'[{i}] Dispersion reduction loss: {adversarial_loss:.5f} classification: {image_class}'
        )

    to_image = transforms.ToPILImage()
    adversarial_image = to_image(image_tensor)
    adversarial_image.save('./dispersion_reduced.png')

    return adversarial_image


if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = '/Users/adamspindler/Downloads/g.png'

image = Image.open(image_path)
if image is None:
    print('Could not open image')
    exit(-1)

image_class = predict(image)
print(f'Loaded image, class is "{image_class}"')

reduce_dispersion(image)
