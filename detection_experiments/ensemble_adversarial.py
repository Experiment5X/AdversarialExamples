import os
import sys
import torch
import math
import random
import numpy as np
from PIL import Image
from pathlib import Path
from SSD.ssd.config import cfg
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms.functional import affine
from faster_rcnn import process_prediction, setup_faster_rcnn_model
from setup_yolo_model import setup_model, get_adversarial_loss, get_prediction_names
from predict_rcnn import predict_image_tensor as predict_rcnn
from predict_yolo import predict_image_tensor as predict_yolo
from predict_ssd import (
    setup_ssd_model,
    process_ssd_predictions,
    get_ssd_adversarial_loss,
)

image_path = sys.argv[1]
to_tensor = ToTensor()
to_pil_image = ToPILImage()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_blur(image, device):
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
    ).to(device)
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


def create_adversarial(image_path, shifts=[(0, 0), (2, 2), (4, 0), (0, 4), (-4, -4), (-8, 0), (0, -8), (0, 8), (0, 8), (8, 8), (-8, -8)]):
    image = Image.open(image_path).convert('RGB').resize((416, 416))
    image_tensor = to_tensor(image).unsqueeze(0).to(device)
    original_image_tensor = to_tensor(image).unsqueeze(0).to(device)

    yolo_model, yolo_classes = setup_model()
    yolo_model.to(device)
    rcnn_model = setup_faster_rcnn_model()
    rcnn_model.to(device)
    ssd_model = setup_ssd_model()
    ssd_model.to(device)

    def ensemble_iteration(current_image_tensor, shift):
        current_image_tensor = torch.clone(current_image_tensor).to(device)
        current_image_tensor = affine(current_image_tensor, 0, shift, 1, [0])
        current_image_tensor.requires_grad = True

        original_image_tensor_shifted = affine(original_image_tensor, 0, shift, 1, [0])

        rccn_detections = rcnn_model.forward(current_image_tensor)
        rcnn_loss = process_prediction(rccn_detections, True)

        yolo_detections = yolo_model.forward(current_image_tensor)
        yolo_loss = get_adversarial_loss(yolo_detections) / 200

        image_mean = torch.Tensor(cfg.INPUT.PIXEL_MEAN).unsqueeze(0).to(device)
        conv_size = conv_sizes[iteration % len(conv_sizes)]

        ssd_image = torch.zeros((1, 3, 512, 512)).to(device)
        ssd_image[
            :, :, : current_image_tensor.shape[-2], : current_image_tensor.shape[-1]
        ] = (current_image_tensor[:] * 255 - image_mean[:, :, None, None])

        ssd_predictions = ssd_model(ssd_image)
        process_ssd_predictions(
            ssd_predictions[1][0]['boxes'],
            ssd_predictions[1][0]['labels'],
            ssd_predictions[1][0]['scores'],
        )
        ssd_loss = get_ssd_adversarial_loss(ssd_predictions)

        yolo_detection_names = get_prediction_names(yolo_detections, yolo_classes)
        print(f'YOLOv3 Detections: {yolo_detection_names}')

        image_diff_regularization = (
            torch.norm(original_image_tensor_shifted - current_image_tensor, 2) / 25
        )
        adversarial_loss = rcnn_loss + yolo_loss + ssd_loss + image_diff_regularization
        adversarial_loss.backward()

        print(
            f'[{iteration}] Adversarial loss total: {adversarial_loss:.5f}, RCNN Loss: {rcnn_loss:.5f}, YOLO Loss: {yolo_loss:.5f}, SSD Loss: {ssd_loss:.5f}, Image Diff: {image_diff_regularization:.5f}'
        )

        return current_image_tensor.grad.data

    conv_sizes = [3, 5, 7]
    for iteration in range(0, 25):
        total_gradient = torch.zeros(image_tensor.shape).to(device)
        for shift in shifts:
            print(f'[{iteration}] Using shift {shift}')
            image_tensor_gradient = ensemble_iteration(image_tensor, shift)
            image_tensor_gradient_shifted = affine(
                image_tensor_gradient, 0, (-shift[0], -shift[1]), 1, [0]
            )
            total_gradient += image_tensor_gradient_shifted

        total_gradient /= len(shifts)

        image_tensor = image_tensor.data - 0.01 * torch.sign(total_gradient)
        image_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min()
        )

        if iteration % 10 == 1 and iteration != 0 and iteration < 48:
            image_tensor = gaussian_blur(image_tensor, device)
            print('\tAdding gassian blur')

        if iteration % 10 == 9 and iteration < 47:
            image_tensor = image_tensor + torch.rand(image_tensor.shape).to(device) / 75
            print('\tAdding random noise')

        print()

    return image_tensor


original_file_name = os.path.basename(image_path)
base_file_name = Path(original_file_name).stem
adversarial_image_path = f'./ensemble_adversarial_{base_file_name}.png'

adversarial_image_tensor = create_adversarial(image_path)
adversarial_image = to_pil_image(adversarial_image_tensor.squeeze(0))
adversarial_image.save(adversarial_image_path)

reloaded_image = Image.open(adversarial_image_path)
reloaded_image_tensor = to_tensor(reloaded_image).unsqueeze(0)

print('Final adversarial image predictions...')
predict_rcnn(reloaded_image_tensor)

reloaded_image_tensor = reloaded_image_tensor.to(device)
predict_yolo(reloaded_image_tensor)
