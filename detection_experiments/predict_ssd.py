import sys
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from SSD.ssd.config import cfg
from SSD.ssd.data.datasets import COCODataset
from SSD.ssd.modeling.detector import build_detection_model
from SSD.ssd.data.transforms import build_transforms
from SSD.ssd.utils.checkpoint import CheckPointer


score_threshold = 0.5


def get_ssd_adversarial_loss(model_output, classes_to_hide=[1, 12]):
    (class_predictions, _), predictions_all = model_output
    predictions = predictions_all[0]

    bbox_loss = torch.norm(predictions['scores'], 2)
    class_loss = torch.sum(class_predictions[0][:, classes_to_hide])

    return (bbox_loss + class_loss) / 50


def setup_ssd_model():
    cfg.merge_from_file('./SSD/configs/vgg_ssd512_coco_trainval35k.yaml')
    cfg.freeze()

    model = build_detection_model(cfg)
    model.eval()

    ckpt = 'https://github.com/lufficc/SSD/releases/download/1.2/vgg_ssd512_coco_trainval35k.pth'
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()

    return model


def process_ssd_predictions(boxes, labels, scores):
    labels_np = labels.detach().numpy()
    scores_np = scores.detach().numpy()

    print('SSD Predictions: ')
    for prediction_index in range(0, labels_np.shape[0]):
        confidence_score = scores_np[prediction_index]

        if confidence_score >= score_threshold:
            class_index = int(labels_np[prediction_index])
            class_name = COCODataset.class_names[class_index]

            print(f'\t{class_name} {confidence_score}')

    if labels is None:
        print('Nothing!')


if __name__ == '__main__':
    model = setup_ssd_model()

    # images need to be ([1, 3, 300, 300])
    ssd_transforms = build_transforms(cfg, False)

    image_path = sys.argv[1]

    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    use_orig_image_size = True
    if use_orig_image_size:
        image_np = image_np.swapaxes(0, 2)
        image_tensor = torch.zeros((1, 3, 512, 512))
        image_tensor[0, :, : image_np.shape[-2], : image_np.shape[-1]] = torch.Tensor(
            image_np
        )
    else:
        image_tensor = ssd_transforms(image_np)[0].unsqueeze(0)

    (class_scores_all, _), predictions_all = model(image_tensor)
    predictions = predictions_all[0]
    boxes, labels, scores = (
        predictions['boxes'],
        predictions['labels'],
        predictions['scores'],
    )

    class_scores = class_scores_all[0]

    process_ssd_predictions(boxes, labels, scores)
