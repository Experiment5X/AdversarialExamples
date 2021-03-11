import os
import sys
import boto3


def predict_image(image_path):
    with open(image_path, 'rb') as image_file:
        image = {'Bytes': image_file.read()}

    rekognition_client = boto3.client('rekognition')
    response = rekognition_client.detect_labels(Image=image, MaxLabels=10)

    return response


def print_response(response):
    for label in response['Labels']:
        name = label['Name']
        confidence = label['Confidence']
        instances = ['%.4f' % instance['Confidence'] for instance in label['Instances']]
        parents = ', '.join([parent['Name'] for parent in label['Parents']])

        print(f'[{confidence:.5f}] {name}: {parents}')
        for instance in instances:
            print(f'\t- {instance}')


def images_in_directory(image_directory):
    for image_name in os.listdir(image_directory):
        if image_name.endswith('.png') or image_name.endswith('.jpeg') or image_name.endswith('.jpg'):
            image_path = os.path.join(image_directory, image_name)
            yield image_path


def predict_aggregate(image_directory):
    confidence_scores = {}
    bboxes = {}

    image_paths = list(images_in_directory(image_directory))[:20]

    for image_path in image_paths:
        prediction_response = predict_image(image_path)
        print(f'Raw labels for {image_path}: ', prediction_response['Labels'])

        for label in prediction_response['Labels']:
            label_class = label['Name']

            if label_class not in confidence_scores:
                confidence_scores[label_class] = []
            
            confidence_scores[label_class].append(label['Confidence'])

            if len(label['Instances']) > 0:
                if label_class not in bboxes:
                    bboxes[label_class] = []
                
                bboxes[label_class].extend(label['Instances'])
    
    print('Raw aggregate dictionaries: ')
    print('Confidence scores: ', confidence_scores)
    print('Bboxes: ', bboxes)
    print()

    image_count = len(image_paths)
    print(f'Predicted {image_count} images')

    for class_name in confidence_scores:
        class_count = len(confidence_scores[class_name])
        average_confidence_score = sum(confidence_scores[class_name]) / class_count
        bbox_count = 0 if class_name not in bboxes else len(bboxes[class_name])

        print(f'{class_name} - Found in {class_count} images, Average confidence score: {average_confidence_score:.5f}, Bbox count: {bbox_count}')


if len(sys.argv) < 2:
    print('Usage: python predict_aws_rekognition.py IMAGE_PATH')

if os.path.isfile(sys.argv[1]):
    image_prediction = predict_image(sys.argv[1])
    print(image_prediction)
elif os.path.isdir(sys.argv[1]):
    predict_aggregate(sys.argv[1])
else:
    print('Invalid input path')