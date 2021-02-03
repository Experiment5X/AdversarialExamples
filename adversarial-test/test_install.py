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

loader = transforms.Compose([transforms.ToTensor()])

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
normalize = transforms.Normalize(mean.tolist(), std.tolist())
denormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


def load_image(image):
    image_tensor = loader(image).float()
    image_normalized = normalize(image_tensor)

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


def test_image_loader(image):
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image)
    image.save('./garbage.png')

    image_read = Image.open('./garbage.png')
    image_read_tensor = to_tensor(image_read)

    print_stats_summary('image', image_tensor.numpy())
    print_stats_summary('after_image', image_read_tensor.numpy())
    print_stats_summary('diff', image_tensor.numpy() - image_read_tensor.numpy())


def get_adversarial_image(image):
    image_tensor = loader(image).float()
    orig_image_tensor = loader(image).float()

    for i in range(0, 10):
        x = Variable(image_tensor, requires_grad=True)
        x_normalized = normalize(x).unsqueeze(0)
        output = model.forward(x_normalized)

        y = Variable(torch.LongTensor(np.array([605])), requires_grad=False,)

        loss = CrossEntropyLoss(output, y)
        loss.backward()

        # ascend the loss function to get as far away from classifying this image
        # as a street sign as possible
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

    to_image = transforms.ToPILImage()
    adversarial_image = to_image(image_tensor)
    adversarial_image.save('./adversarial.png')

    adversarial_image2 = Image.open('./adversarial.png')
    adversarial_tensor2 = loader(adversarial_image2).float()

    # something still might be messed up with these, diff should be zero i think still
    # print_stats_summary('image_tensor', image_tensor.numpy())
    # print_stats_summary('adversarial_tensor', adversarial_tensor2.numpy())
    # print_stats_summary('diff', (image_tensor - adversarial_tensor2).numpy())

    return adversarial_image


image = Image.open('/Users/adamspindler/Downloads/stop.png')
# image = Image.open('./adversarial.png')
if image is None:
    print('no image')
    exit(-1)

print('prediction: ', predict(image))

adversarial_image = get_adversarial_image(image)
