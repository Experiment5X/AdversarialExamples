import sys
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from faster_rcnn import process_prediction

image_path = sys.argv[1]
to_tensor = ToTensor()
to_pil_image = ToPILImage()


def predict_image_tensor(image_tensor):
    model = fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    model.eval()

    prediction_infos = model.forward(image_tensor)
    all_confidence_scores = process_prediction(prediction_infos)


def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = to_tensor(image).unsqueeze(0)

    predict_image_tensor(image_tensor)


if __name__ == '__main__':
    predict(image_path)
