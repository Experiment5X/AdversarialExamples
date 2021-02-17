import sys
import torch
from pytorch_yolov3.models import Darknet
from pytorch_yolov3.utils.datasets import ImageFile
from pytorch_yolov3.utils.transforms import DEFAULT_TRANSFORMS, Resize
from pytorch_yolov3.utils.utils import load_classes

import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def setup_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet('pytorch_yolov3/config/yolov3.cfg', img_size=416).to(device)
    model.load_darknet_weights('pytorch_yolov3/weights/yolov3.weights')

    model.eval()

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'pytorch_yolov3/data/test/field.jpg'

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

