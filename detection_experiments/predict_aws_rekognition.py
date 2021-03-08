import sys
import boto3


if len(sys.argv) < 2:
    print('Usage: python predict_aws_rekognition.py IMAGE_PATH')


image_path = sys.argv[1]
with open(image_path, 'rb') as image_file:
    image = {'Bytes': image_file.read()}

rekognition_client = boto3.client('rekognition')
response = rekognition_client.detect_labels(Image=image, MaxLabels=10)

for label in response['Labels']:
    name = label['Name']
    confidence = label['Confidence']
    instances = ['%.4f' % instance['Confidence'] for instance in label['Instances']]
    parents = ', '.join([parent['Name'] for parent in label['Parents']])

    print(f'[{confidence:.5f}] {name}: {parents}')
    for instance in instances:
        print(f'\t- {instance}')

