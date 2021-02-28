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


if __name__ == '__main__':
    cfg.merge_from_file('./SSD/configs/vgg_ssd300_coco_trainval35k.yaml')
    cfg.freeze()

    model = build_detection_model(cfg)
    model.eval()

    ckpt = 'https://github.com/lufficc/SSD/releases/download/1.2/vgg_ssd300_coco_trainval35k.pth'
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()

    # images need to be ([1, 3, 300, 300])
    ssd_transforms = build_transforms(cfg, False)

    image = Image.open(sys.argv[1]).convert('RGB')
    image_np = np.array(image)
    image_tensor = ssd_transforms(image_np)[0].unsqueeze(0)

    predictions = model(image_tensor)[0]
    boxes, labels, scores = (
        predictions['boxes'],
        predictions['labels'],
        predictions['scores'],
    )

    labels_np = labels.detach().numpy()
    scores_np = scores.detach().numpy()

    print('SSD Predictions: ')
    for prediction_index in range(0, labels_np.shape[0]):
        confidence_score = scores_np[prediction_index]

        if confidence_score >= score_threshold:
            class_index = int(labels_np[prediction_index])
            class_name = COCODataset.class_names[class_index]

            print(f'{class_name} {confidence_score}')

    if labels is None:
        print('Nothing!')

