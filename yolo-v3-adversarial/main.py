import cv2
import torch
from pytorch_yolov3.models import Darknet
from pytorch_yolov3.utils.datasets import ImageFolder
from pytorch_yolov3.utils.transforms import DEFAULT_TRANSFORMS, Resize
from pytorch_yolov3.utils.utils import load_classes, non_max_suppression

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

"""
output:
[1, 10647, 85]

10647 possible bounding boxes with labels
    0: bbox center x
    1: bbox center y
    2: bbox width
    3: bbox height
    4: bbox existence confidence
    5-85: classification scores
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up model
model = Darknet('pytorch_yolov3/config/yolov3.cfg', img_size=416).to(device)
model.load_darknet_weights('pytorch_yolov3/weights/yolov3.weights')
model.train()

dataloader = DataLoader(
    ImageFolder(
        'pytorch_yolov3/data/test',
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(416)]),
    ),
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

classes = load_classes(
    'pytorch_yolov3/data/coco.names'
)  # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    original_image_tensor = input_imgs.type(Tensor)
    image_tensor = input_imgs.type(Tensor)
    for i in range(0, 1):
        # Configure input
        x = image_tensor
        x.requires_grad = True

        # Get detections
        detections = model.forward(x)

        bbox_confidence_mask = torch.zeros(detections.shape)
        bbox_confidence_mask[:, :, 4] = 1
        bbox_confidences = detections * bbox_confidence_mask

        adversarial_loss = torch.sum(bbox_confidences)
        adversarial_loss.backward()

        print(f'adversarial loss = {adversarial_loss:.5f}')

        image_tensor = image_tensor.data - 0.005 * torch.sign(x.grad.data)

        # need to normalize to keep the pixel values between 0 and 1
        # could also clip probably
        image_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min()
        )

        detections = non_max_suppression(detections, 0.8, 0.4)

        print('Detected: ')
        if detections[0] is not None:
            for detection in detections[0]:
                print(classes[int(detection[6])])
            print()
        else:
            print('Nothing!')
            break

to_pil_image = transforms.ToPILImage()
to_pil_image(image_tensor[0, ...]).save('./adversarial.png')
to_pil_image(original_image_tensor[0, ...]).save('./original.png')

print('Wrote adversarial image')
