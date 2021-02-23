import sys
import torch
from PIL import Image
from pytorch_yolov3.models import Darknet
from pytorch_yolov3.utils.datasets import ImageFile
from pytorch_yolov3.utils.transforms import DEFAULT_TRANSFORMS, Resize
from pytorch_yolov3.utils.utils import load_classes, non_max_suppression
from setup_yolo_model import setup_model

import torchvision.transforms as transforms
from torch.utils.data import DataLoader


to_tensor = transforms.ToTensor()

image_path = sys.argv[1]
(model, classes) = setup_model()


def predict_image_tensor(image_tensor):
    # Get detections
    detections = model.forward(image_tensor)
    detections = non_max_suppression(detections, 0.8, 0.4)

    print('YOLOv3 Detected: ')
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


def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = to_tensor(image).unsqueeze(0)

    predict_image_tensor(image_tensor)


if __name__ == '__main__':
    predict(image_path)
