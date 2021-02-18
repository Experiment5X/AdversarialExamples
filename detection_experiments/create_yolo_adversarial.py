import os
import cv2
import sys
import torch
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from setup_yolo_model import setup_model, get_adversarial_loss, get_prediction_names
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
to_tensor = transforms.ToTensor()


def is_image_file(file_name):
    f = file_name.lower()
    return f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')


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
    (model, classes) = setup_model()

    print(f'Working on {file_path}')

    image = Image.open(file_path).convert('RGB').resize((416, 416))
    image_tensor = to_tensor(image).unsqueeze(0)
    original_image_tensor = to_tensor(image).unsqueeze(0)

    for i in range(0, 15):
        # Configure input
        x = image_tensor
        x.requires_grad = True

        # Get detections
        detections = model.forward(x)

        # loss function wants all the bbox confidences to be 0
        raw_adversarial_loss = get_adversarial_loss(detections)
        adversarial_loss = raw_adversarial_loss + torch.norm(
            image_tensor - original_image_tensor, 2
        )
        adversarial_loss.backward()

        # interpret the output matrix into the names of the objects detected
        detected_objects_str = get_prediction_names(detections, classes)

        print(
            f'[{i}] Adversarial loss = {adversarial_loss:.5f} - detected objects: {detected_objects_str}'
        )

        # update the image in the adversarial direction with random regularization
        image_tensor = (
            image_tensor.data
            - 0.005 * torch.sign(x.grad.data)
            + 0.0003 * torch.rand(x.grad.shape)
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

    image = Image.open(adversarial_file_name).convert('RGB')
    image_tensor_reloaded = to_tensor(image).unsqueeze(0)

    predict_image_tensor(image_tensor_reloaded)
