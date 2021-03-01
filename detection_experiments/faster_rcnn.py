import torch
from coco_names import COCO_INSTANCE_CATEGORY_NAMES
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def process_prediction(prediction_infos, monkey_patched=False):
    confidence_score_size = len(
        max(prediction_infos, key=lambda info: len(info['scores']))['scores']
    )
    all_confidence_scores = torch.zeros((len(prediction_infos), confidence_score_size))
    print('Faster-RCNN Detections...')
    for image_index, info in enumerate(prediction_infos):
        if monkey_patched:
            class_logits, labels = info['labels']
        else:
            labels = info['labels']
        for prediction_class, confidence_score in zip(labels, info['scores']):
            class_name = COCO_INSTANCE_CATEGORY_NAMES[prediction_class]
            if confidence_score > 0.5:
                print(f'\t{class_name} - {confidence_score:.5f}')

        all_confidence_scores[image_index, :] = info['scores']

        if len(info['labels']) == 0:
            print('No detections!')

    return all_confidence_scores


def create_new_postprocess_detections(orig_post_process):
    def postprocess_detections(
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):

        boxes, scores, labels = orig_post_process(
            class_logits, box_regression, proposals, image_shapes
        )
        labels = [(class_logits, label) for label in labels]

        return boxes, scores, labels

    return postprocess_detections


def setup_faster_rcnn_model():
    rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    rcnn_model.eval()
    rcnn_model.transform.postprocess = lambda a, b, c: a

    orig_post_process = rcnn_model.roi_heads.postprocess_detections
    rcnn_model.roi_heads.postprocess_detections = create_new_postprocess_detections(
        orig_post_process
    )

    return rcnn_model
