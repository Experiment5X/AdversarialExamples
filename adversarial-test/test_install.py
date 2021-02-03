from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

from labels import label_lookup

mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
model = models.vgg16(pretrained=True).eval()
CrossEntropyLoss = torch.nn.CrossEntropyLoss()

loader = transforms.Compose([transforms.Resize(224, 224), transforms.ToTensor()])

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
normalize = transforms.Normalize(mean.tolist(), std.tolist())
denormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


def load_image(image):
    image_converted = image.convert('RGB')
    image_loaded = loader(image_converted).float()
    image_normalized = normalize(image_loaded)

    return image_normalized.unsqueeze(0)


def predict(image):
    image_tensor = load_image(image)
    prediction_tensor = model.forward(image_tensor)
    label_index = int(prediction_tensor.argmax().numpy())

    return label_lookup[label_index]


def normalize_grad(grad):
    grad_np = grad.numpy()
    centered = (grad_np - np.min(grad_np)) / (np.max(grad_np) - np.min(grad_np))
    return centered


def print_stats_summary(name, arr):
    print(f'{name} - mean: {arr.mean()} min: {arr.min()} max: {arr.max()}')


# there is a slight difference between the original image and the image after it has
#  been normalized and de-normalized
def test_image_normalization(image):
    image_np = np.array(image)
    image_tensor = load_image(image)

    to_image = transforms.ToPILImage()
    after_image = to_image(denormalize(torch.squeeze(image_tensor)))
    after_image_np = np.array(after_image)

    print_stats_summary('image', image_np)
    print_stats_summary('after_image', after_image_np)
    print_stats_summary('diff', image_np - after_image_np)

    return after_image


def get_adversarial_image(image):
    image_tensor = load_image(image)

    for i in range(0, 10):
        x = Variable(image_tensor, requires_grad=True)
        output = model.forward(x)

        y = Variable(torch.LongTensor(np.array([919])), requires_grad=False,)

        loss = CrossEntropyLoss(output, y)
        loss.backward()

        # ascend the loss function to get as far away from classifying this image
        # as a street sign as possible
        image_tensor = x.data + 0.05 * normalize_grad(x.grad.data)

    normalized_grad = normalize_grad(x.grad.data)
    print(
        'grad mean: ',
        np.mean(normalized_grad),
        np.min(normalized_grad),
        np.max(normalized_grad),
    )

    adversarial_result = int(model.forward(image_tensor).argmax().numpy())
    print('adversarial label: ', label_lookup[adversarial_result])

    to_image = transforms.ToPILImage()
    adversarial_image = to_image(inv_normalize(torch.squeeze(image_tensor)))

    return adversarial_image


image = Image.open('/Users/adamspindler/Downloads/stop.png')
if image is None:
    print('no image')
    exit(-1)

print('prediction: ', predict(image))

adversarial_image = get_adversarial_image(image)
adversarial_image.save('./adversarial.png')
