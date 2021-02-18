import sys
import torch
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


def create_adversarial(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = to_tensor(image).unsqueeze(0)
    orginal_image_tensor = to_tensor(image).unsqueeze(0)

    rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    rcnn_model.eval()

    yolo_model, yolo_classes = setup_model(image_path)

    for iteration in range(0, 25):
        image_tensor.requires_grad = True

        rccn_detections = rcnn_model.forward(image_tensor)
        all_confidence_scores_rcnn = process_prediction(rccn_detections)

        yolo_detections = yolo_model.forward(image_tensor)
        yolo_adversarial_loss = get_adversarial_loss(yolo_detections)

        yolo_detection_names = get_prediction_names(yolo_detections, yolo_classes)
        print(f'YOLOv3 Detections: {yolo_detection_names}')

        adversarial_loss = (
            10 * torch.norm(all_confidence_scores_rcnn, 2)
            + 0.5 * yolo_adversarial_loss
            + 0.1 * torch.norm(orginal_image_tensor - image_tensor, 2)
        )
        adversarial_loss.backward()

        image_tensor = image_tensor.data - 0.01 * torch.sign(image_tensor.grad.data)
        image_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min()
        )

        print(f'[{iteration}] Adversarial loss: {adversarial_loss:.5f}')
        print()

    return image_tensor


adversarial_image_tensor = create_adversarial(image_path)
adversarial_image = to_pil_image(adversarial_image_tensor.squeeze(0))

adversarial_image.save('./ensemble_adversarial.png')

print('Final adversarial image predictions...')
predict_rcnn(adversarial_image_tensor)
predict_yolo(adversarial_image_tensor)
