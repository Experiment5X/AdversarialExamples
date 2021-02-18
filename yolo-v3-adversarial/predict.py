import sys
import torch
from pytorch_yolov3.models import Darknet
from pytorch_yolov3.utils.datasets import ImageFile
from pytorch_yolov3.utils.transforms import DEFAULT_TRANSFORMS, Resize
from pytorch_yolov3.utils.utils import load_classes, non_max_suppression
from setup_model import setup_model

import torchvision.transforms as transforms
from torch.utils.data import DataLoader


(model, dataloader, classes, Tensor) = setup_model(sys.argv[1])


def predict_image_tensor(image_tensor):
    # Get detections
    detections = model.forward(image_tensor)
    detections = non_max_suppression(detections, 0.8, 0.4)

    print('Detected: ')
    if detections[0] is not None:
        for detection in detections[0]:
            bbox_confidence_score = detection[4]
            class_confidence_score = detection[5]
            detected_class = classes[int(detection[6])]
            print(
                f'{detected_class} {class_confidence_score:.5f} - Bbox score: {bbox_confidence_score:.5f}'
            )
        print()
    else:
        print('Nothing!')


if __name__ == '__main__':
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        image_tensor = input_imgs.type(Tensor)

        predict_image_tensor(image_tensor)
