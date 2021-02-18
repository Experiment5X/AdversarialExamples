import sys
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from coco_names import COCO_INSTANCE_CATEGORY_NAMES


image_path = (
    '/Users/adamspindler/Developer/MS-Project/test_images/stop.png'  # sys.argv[1]
)
to_tensor = ToTensor()
to_pil_image = ToPILImage()


def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = to_tensor(image).unsqueeze(0)

    model = fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    model.eval()

    for iteration in range(0, 5):
        image_tensor.requires_grad = True

        prediction_infos = model.forward(image_tensor)

        confidence_score_size = len(
            max(prediction_infos, key=lambda info: len(info['scores']))['scores']
        )
        all_confidence_scores = torch.zeros(
            (len(prediction_infos), confidence_score_size)
        )
        print('Detections...')
        for image_index, info in enumerate(prediction_infos):
            for prediction_class, confidence_score in zip(
                info['labels'], info['scores']
            ):
                class_name = COCO_INSTANCE_CATEGORY_NAMES[prediction_class]
                print(f'{class_name} - {confidence_score:.5f}')

            all_confidence_scores[image_index, :] = info['scores']

        adversarial_loss = torch.norm(all_confidence_scores, 2)
        adversarial_loss.backward()

        image_tensor = image_tensor.data - 0.01 * torch.sign(image_tensor.grad.data)
        image_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min()
        )

        print(f'[{iteration}] Adversarial loss: {adversarial_loss:.5f}')
        print()

    return image_tensor


adversarial_image_tensor = predict(image_path)
adversarial_image = to_pil_image(adversarial_image_tensor.squeeze(0))

adversarial_image.save('./ensemble_adversarial.png')
