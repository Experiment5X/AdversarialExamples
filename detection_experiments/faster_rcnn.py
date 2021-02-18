import torch
from coco_names import COCO_INSTANCE_CATEGORY_NAMES


def process_prediction(prediction_infos):
    confidence_score_size = len(
        max(prediction_infos, key=lambda info: len(info['scores']))['scores']
    )
    all_confidence_scores = torch.zeros((len(prediction_infos), confidence_score_size))
    print('Detections...')
    for image_index, info in enumerate(prediction_infos):
        for prediction_class, confidence_score in zip(info['labels'], info['scores']):
            class_name = COCO_INSTANCE_CATEGORY_NAMES[prediction_class]
            print(f'{class_name} - {confidence_score:.5f}')

        all_confidence_scores[image_index, :] = info['scores']

        if len(info['labels']) == 0:
            print('No detections!')

    return all_confidence_scores
