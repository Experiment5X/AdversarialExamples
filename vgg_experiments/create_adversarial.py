import sys
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from labels import label_lookup
from common import CrossEntropyLoss, loader, inv_normalize, model, normalize, predict


def print_stats_summary(name, arr):
    print(f'{name} - mean: {arr.mean()} min: {arr.min()} max: {arr.max()}')


# there is a slight difference between the original image and the image after it has
#  been normalized and de-normalized
def test_image_normalization(image):
    image_np = np.array(image)
    image_tensor = load_image(image)

    to_image = transforms.ToPILImage()
    after_image = to_image(inv_normalize(torch.squeeze(image_tensor)))
    after_image_np = np.array(after_image)

    print_stats_summary('image', image_np)
    print_stats_summary('after_image', after_image_np)
    print_stats_summary('diff', image_np - after_image_np)

    return after_image


def test_image_loader(image):
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image)
    image.save('./garbage.png')

    image_read = Image.open('./garbage.png')
    image_read_tensor = to_tensor(image_read)

    print_stats_summary('image', image_tensor.numpy())
    print_stats_summary('after_image', image_read_tensor.numpy())
    print_stats_summary('diff', image_tensor.numpy() - image_read_tensor.numpy())


def get_adversarial_image(image, target_class=605):
    image_tensor = loader(image).float()
    orig_image_tensor = loader(image).float()

    print('Starting iterations...')
    for i in range(0, 10):
        x = image_tensor
        x.requires_grad = True

        x_normalized = normalize(x).unsqueeze(0)
        output = model.forward(x_normalized)

        y = torch.LongTensor([target_class])

        loss = CrossEntropyLoss(output, y) + torch.norm(x - orig_image_tensor, 2)
        loss.backward()

        # descend the loss function to get close to the target class
        image_tensor = x.data - 0.005 * torch.sign(x.grad.data)

        # need to normalize to keep the pixel values between 0 and 1
        # could also clip probably
        image_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min()
        )

        adversarial_result = int(
            model.forward(normalize(image_tensor).unsqueeze(0)).argmax().numpy()
        )

        print(f'[{i}] - {loss:.5f} - {label_lookup[adversarial_result]}')

        if adversarial_result == target_class:
            print('Stopping early, created adversarial image successfully')
            break

    to_image = transforms.ToPILImage()
    adversarial_image = to_image(image_tensor)
    adversarial_image.save('./adversarial.png')
    print('Wrote adversarial image to ./adversarial.png')

    adversarial_image2 = Image.open('./adversarial.png')
    adversarial_tensor2 = loader(adversarial_image2).float()

    return adversarial_image


if sys.argv[1] is not None:
    image_path = sys.argv[1]
else:
    image_path = '/Users/adamspindler/Downloads/g.png'

image = Image.open(image_path)
if image is None:
    print('Could not open image')
    exit(-1)

image_class = predict(image)
print(f'Loaded image, class is "{image_class}"')

adversarial_image = get_adversarial_image(image)
