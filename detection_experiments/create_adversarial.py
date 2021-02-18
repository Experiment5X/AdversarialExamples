import os
import cv2
import sys
import torch
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from setup_yolo_model import setup_model
from pytorch_yolov3.models import Darknet
from pytorch_yolov3.utils.datasets import ImageFile
from pytorch_yolov3.utils.transforms import DEFAULT_TRANSFORMS, Resize
from pytorch_yolov3.utils.utils import load_classes, non_max_suppression

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from predict_yolo import predict_image_tensor

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

to_pil_image = transforms.ToPILImage()
transforms = transforms.Compose([DEFAULT_TRANSFORMS, Resize(416)])


def is_image_file(file_name):
    f = file_name.lower()
    return f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')


class_to_hide = 12

if len(sys.argv) < 2:
    print('Must specify an image file or directory full of image files')
    exit(-1)

specified_path = sys.argv[1]
output_directory = sys.argv[2] if len(sys.argv) > 2 else './'

if os.path.isfile(specified_path):
    file_paths = [specified_path]
else:
    file_paths = [
        os.path.join(specified_path, f)
        for f in os.listdir(specified_path)
        if os.path.isfile(os.path.join(specified_path, f)) and is_image_file(f)
    ]

    print(f'Found {len(file_paths)} files to make adversarial')

for file_path in file_paths:
    (model, dataloader, classes, Tensor) = setup_model(file_path)

    print(f'Working on {file_path}')
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        original_image_tensor = input_imgs.type(Tensor)
        image_tensor = input_imgs.type(Tensor)

        print(image_tensor)
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
                + 0.0005 * torch.rand(x.grad.shape)
            )

            # need to normalize to keep the pixel values between 0 and 1
            image_tensor = (image_tensor - image_tensor.min()) / (
                image_tensor.max() - image_tensor.min()
            )

    original_file_name = os.path.basename(file_path)
    base_file_name = Path(original_file_name).stem
    adversarial_file_name = os.path.join(
        output_directory, f'adversarial_{base_file_name}.png'
    )

    pil_image = to_pil_image(image_tensor[0, ...])
    pil_image.save(adversarial_file_name)

    print(f'Wrote adversarial image to: {adversarial_file_name}')
    print('Re-running detection on PNG image...')

    boxes = np.zeros((1, 5))
    img = np.array(Image.open(adversarial_file_name).convert('RGB'), dtype=np.uint8)

    image_tensor_reloaded, _ = transforms((img, boxes))
    predict_image_tensor(image_tensor_reloaded.unsqueeze(0))
