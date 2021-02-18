import sys
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from faster_rcnn import process_prediction

image_path = sys.argv[1]
to_tensor = ToTensor()
to_pil_image = ToPILImage()


def create_adversarial(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = to_tensor(image).unsqueeze(0)

    model = fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    model.eval()

    for iteration in range(0, 5):
        image_tensor.requires_grad = True

        prediction_infos = model.forward(image_tensor)
        all_confidence_scores = process_prediction(prediction_infos)

        adversarial_loss = torch.norm(all_confidence_scores, 2)
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
