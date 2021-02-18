import torch
from pytorch_yolov3.models import Darknet
from pytorch_yolov3.utils.datasets import ImageFile
from pytorch_yolov3.utils.transforms import DEFAULT_TRANSFORMS, Resize
from pytorch_yolov3.utils.utils import load_classes, non_max_suppression

import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_adversarial_loss(detections, class_to_hide=12):
    # Get the confidences in the bboxes existing
    bbox_confidence_mask = torch.zeros(detections.shape)
    bbox_confidence_mask[:, :, 4] = 1
    bbox_confidences = detections * bbox_confidence_mask

    # Get the confidences in stop signs existing
    stop_sign_confidence_mask = torch.zeros(detections.shape)
    stop_sign_confidence_mask[:, :, 4 + class_to_hide] = 1
    stop_sign_confidences = detections * stop_sign_confidence_mask

    return torch.sum(bbox_confidences) + torch.sum(stop_sign_confidences)


def get_prediction_names(detections, classes):
    detections = non_max_suppression(detections, 0.8, 0.4)
    if detections[0] is not None:
        detected_objects = [
            f'{classes[int(detection[6])]} {detection[5]:.5f}'
            for detection in detections[0]
        ]
    else:
        detected_objects = ['Nothing!']

    detected_objects_str = ', '.join(detected_objects)
    return detected_objects_str


def setup_model(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet('pytorch_yolov3/config/yolov3.cfg', img_size=416).to(device)
    model.load_darknet_weights('pytorch_yolov3/weights/yolov3.weights')

    model.eval()

    dataloader = DataLoader(
        ImageFile(
            image_path, transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(416)]),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    classes = load_classes(
        'pytorch_yolov3/data/coco.names'
    )  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    return (model, dataloader, classes, Tensor)

