import sys
import torch
import math
import random
import numpy as np
from PIL import Image
from SSD.ssd.config import cfg
from torchvision.transforms import ToTensor, ToPILImage
from faster_rcnn import process_prediction, setup_faster_rcnn_model
from setup_yolo_model import setup_model, get_adversarial_loss, get_prediction_names
from predict_rcnn import predict_image_tensor as predict_rcnn
from predict_yolo import predict_image_tensor as predict_yolo
from predict_ssd import (
    setup_ssd_model,
    process_ssd_predictions,
    get_ssd_adversarial_loss,
)

image_path = (
    '/Users/adamspindler/Developer/MS-Project/test_images/stop2.png'  # sys.argv[1]
)
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

    yolo_model, yolo_classes = setup_model()
    rcnn_model = setup_faster_rcnn_model()
    ssd_model = setup_ssd_model()

    conv_sizes = [3, 5, 7]
    for iteration in range(0, 10):
        image_tensor.requires_grad = True

        rccn_detections = rcnn_model.forward(image_tensor)
        rcnn_loss = process_prediction(rccn_detections, True)

        yolo_detections = yolo_model.forward(image_tensor)
        yolo_loss = get_adversarial_loss(yolo_detections) / 200

        conv_size = conv_sizes[iteration % len(conv_sizes)]
        ssd_sized_image = (
            torch.nn.FractionalMaxPool2d(conv_size, (300, 300))(image_tensor) * 255
        )
        image_mean = torch.Tensor(cfg.INPUT.PIXEL_MEAN).unsqueeze(0)
        ssd_image = ssd_sized_image - image_mean[:, :, None, None]

        ssd_predictions = ssd_model(ssd_sized_image)
        process_ssd_predictions(
            ssd_predictions[1][0]['boxes'],
            ssd_predictions[1][0]['labels'],
            ssd_predictions[1][0]['scores'],
        )
        ssd_loss = get_ssd_adversarial_loss(ssd_predictions)

        yolo_detection_names = get_prediction_names(yolo_detections, yolo_classes)
        print(f'YOLOv3 Detections: {yolo_detection_names}')

        image_diff_regularization = (
            torch.norm(orginal_image_tensor - image_tensor, 2) / 25
        )
        adversarial_loss = rcnn_loss + yolo_loss + ssd_loss + image_diff_regularization
        adversarial_loss.backward()

        image_tensor = image_tensor.data - 0.01 * torch.sign(image_tensor.grad.data)
        image_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min()
        )

        if iteration % 3 == 0 and iteration != 0 and iteration < 29:
            image_tensor = gaussian_blur(image_tensor)
            print('\tAdding gassian blur')

        if iteration % 8 == 0 and iteration < 45:
            image_tensor = image_tensor + torch.rand(image_tensor.shape) / 75
            print('\tAdding random noise')

        print(
            f'[{iteration}] Adversarial loss total: {adversarial_loss:.5f}, RCNN Loss: {rcnn_loss:.5f}, YOLO Loss: {yolo_loss:.5f}, SSD Loss: {ssd_loss:.5f}, Image Diff: {image_diff_regularization:.5f}'
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
