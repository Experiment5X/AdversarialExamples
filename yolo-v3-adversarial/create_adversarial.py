import cv2
import sys
import torch
from setup_model import setup_model
from pytorch_yolov3.models import Darknet
from pytorch_yolov3.utils.datasets import ImageFile
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

(model, dataloader, classes, Tensor) = setup_model(False)

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

        # loss function wants all the bbox confidences to be 0
        adversarial_loss = torch.sum(bbox_confidences)
        adversarial_loss.backward()

        # interpret the output matrix into the names of the objects detected
        detections = non_max_suppression(detections, 0.8, 0.4)
        if detections[0] is not None:
            detected_objects = [
                classes[int(detection[6])] for detection in detections[0]
            ]
        else:
            detected_objects = ['Nothing!']

        detected_objects_str = ', '.join(detected_objects)
        print(
            f'Adversarial loss = {adversarial_loss:.5f} - detected objects: {detected_objects_str}'
        )

        # stop early if no objects detected any more
        if detected_objects[0] == 'Nothing!':
            print('Successfully created adversarial image, stopping early')
            break

        # update the image in the adversarial direction
        image_tensor = image_tensor.data - 0.005 * torch.sign(x.grad.data)

        # need to normalize to keep the pixel values between 0 and 1
        # could also clip probably
        image_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min()
        )


to_pil_image = transforms.ToPILImage()
to_pil_image(image_tensor[0, ...]).save('./adversarial.png')

print('Wrote adversarial image')
