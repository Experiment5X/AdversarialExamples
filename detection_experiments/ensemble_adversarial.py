import sys
import torch
import math
import random
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from faster_rcnn import process_prediction
from setup_yolo_model import setup_model, get_adversarial_loss, get_prediction_names
from predict_rcnn import predict_image_tensor as predict_rcnn
from predict_yolo import predict_image_tensor as predict_yolo

image_path = sys.argv[1]
to_tensor = ToTensor()
to_pil_image = ToPILImage()


def gaussian_blur(image):
    # Set these to whatever you want for your gaussian filter
    kernel_size = 3
    sigma = 0.25
    channels = 3

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.0
    variance = sigma ** 2.0

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
    )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        groups=channels,
        bias=False,
        padding=1,
    )

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter(image)


def create_adversarial(image_path):
    image = Image.open(image_path).convert('RGB').resize((416, 416))
    image_tensor = to_tensor(image).unsqueeze(0)
    orginal_image_tensor = to_tensor(image).unsqueeze(0)

    rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    rcnn_model.eval()

    yolo_model, yolo_classes = setup_model()

    for iteration in range(0, 30):
        image_tensor.requires_grad = True

        rccn_detections = rcnn_model.forward(image_tensor)
        all_confidence_scores_rcnn = process_prediction(rccn_detections)
        rcnn_loss = torch.norm(all_confidence_scores_rcnn, 2)

        yolo_detections = yolo_model.forward(image_tensor)
        yolo_loss = get_adversarial_loss(yolo_detections) / 200

        yolo_detection_names = get_prediction_names(yolo_detections, yolo_classes)
        print(f'YOLOv3 Detections: {yolo_detection_names}')

        image_diff_regularization = (
            torch.norm(orginal_image_tensor - image_tensor, 2) / 25
        )
        adversarial_loss = rcnn_loss + yolo_loss + image_diff_regularization
        adversarial_loss.backward()

        image_tensor = image_tensor.data - 0.01 * torch.sign(image_tensor.grad.data)
        image_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min()
        )

        if iteration % 3 == 0 and iteration != 0 and iteration < 29:
            image_tensor = gaussian_blur(image_tensor)
            print('\tAdding gassian blur')

        print(
            f'[{iteration}] Adversarial loss total: {adversarial_loss:.5f}, RCNN Loss: {rcnn_loss:.5f}, YOLO Loss: {yolo_loss:.5f}, Image Diff: {image_diff_regularization:.5f}'
        )
        print()

    return image_tensor


adversarial_image_tensor = create_adversarial(image_path)
adversarial_image = to_pil_image(adversarial_image_tensor.squeeze(0))

image_id = random.randint(0, 1000)
adversarial_image_path = f'./ensemble_adversarial-{image_id}.png'
adversarial_image.save(adversarial_image_path)

reloaded_image = Image.open(adversarial_image_path)
reloaded_image_tensor = to_tensor(reloaded_image).unsqueeze(0)

print('Final adversarial image predictions...')
predict_rcnn(reloaded_image_tensor)
predict_yolo(reloaded_image_tensor)
