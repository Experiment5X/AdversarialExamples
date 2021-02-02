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
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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


def get_adversarial_image(image):
    image_tensor = load_image(image)
    x = Variable(image_tensor, requires_grad=True)
    output = model.forward(x)

    target_y = torch.tensor([479])

    loss = CrossEntropyLoss(output, target_y)
    loss.backward()

    adversarial_tensor = x.data + 0.2 * torch.sign(x.grad.data)
    print('grad mean: ', x.grad.data.numpy().mean())

    adversarial_result = int(model.forward(adversarial_tensor).argmax().numpy())
    print('adversarial label: ', label_lookup[adversarial_result])

    to_image = transforms.ToPILImage()
    adversarial_image = to_image(inv_normalize(torch.squeeze(adversarial_tensor)))

    return adversarial_image


image = Image.open('/Users/adamspindler/Downloads/stop.png')
if image is None:
    print('no image')
    exit(-1)

print('prediction: ', predict(image))

adversarial_image = get_adversarial_image(image)
adversarial_image.save('./adversarial.png')
