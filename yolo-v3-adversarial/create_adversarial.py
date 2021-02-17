import cv2
import sys
import torch
import pickle
import numpy as np
from PIL import Image
from setup_model import setup_model
from pytorch_yolov3.models import Darknet
from pytorch_yolov3.utils.datasets import ImageFile
from pytorch_yolov3.utils.transforms import DEFAULT_TRANSFORMS, Resize
from pytorch_yolov3.utils.utils import load_classes, non_max_suppression

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from predict import predict_image_tensor

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

class_to_hide = 12

(model, dataloader, classes, Tensor) = setup_model()

for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    original_image_tensor = input_imgs.type(Tensor)
    image_tensor = input_imgs.type(Tensor)
    for i in range(0, 10):
        # Configure input
        x = image_tensor
        x.requires_grad = True

        # Get detections
        detections = model.forward(x)

        # Get the confidences in the bboxes existing
        bbox_confidence_mask = torch.zeros(detections.shape)
        bbox_confidence_mask[:, :, 4] = 1
        bbox_confidences = detections * bbox_confidence_mask

        # Get the confidences in stop signs existing
        stop_sign_confidence_mask = torch.zeros(detections.shape)
        stop_sign_confidence_mask[:, :, 4 + class_to_hide] = 1
        stop_sign_confidences = detections * stop_sign_confidence_mask

        # loss function wants all the bbox confidences to be 0
        adversarial_loss = (
            torch.sum(bbox_confidences)
            + torch.sum(stop_sign_confidences)
            + torch.norm(image_tensor - original_image_tensor, 2)
        )
        adversarial_loss.backward()

        # interpret the output matrix into the names of the objects detected
        detections = non_max_suppression(detections, 0.8, 0.4)
        if detections[0] is not None:
            detected_objects = [
                f'{classes[int(detection[6])]} {detection[5]:.5f}'
                for detection in detections[0]
            ]
        else:
            detected_objects = ['Nothing!']

        detected_objects_str = ', '.join(detected_objects)
        print(
            f'[{i}] Adversarial loss = {adversarial_loss:.5f} - detected objects: {detected_objects_str}'
        )

        # update the image in the adversarial direction with random regularization
        image_tensor = (
            image_tensor.data
            - 0.007 * torch.sign(x.grad.data)
            + 0.003 * torch.rand(x.grad.shape)
        )

        # need to normalize to keep the pixel values between 0 and 1
        image_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min()
        )


to_pil_image = transforms.ToPILImage()
pil_image = to_pil_image(image_tensor[0, ...])
pil_image.save('./adversarial.png')

boxes = np.zeros((1, 5))
img = np.array(Image.open('./adversarial.png').convert('RGB'), dtype=np.uint8)

transforms = transforms.Compose([DEFAULT_TRANSFORMS, Resize(416)])
image_tensor_reloaded, _ = transforms((img, boxes))

print('Wrote adversarial image')

print('Re-running detection on PNG image...')
predict_image_tensor(image_tensor_reloaded.unsqueeze(0))
