# Adversarial Examples for AWS Rekognition
Getting started with creating adversarial images to trick AWS Rekognition!

So far I can create basic adversarial images to trick VGG-16.
![AdversarialVGGDemo](/demos/vgg_adversarial.gif)


I can also create basic adversarial images to trick an object detector, YOLOv3:
![AdversarialYoloDemo](/demos/yolo_adversarial.gif)

Here I ran my YOLOv3 adversarial attack on several different stop sign images, where the stop signs are at various distances from the camera. My attack is able to work for all of the images. These adversarial examples are very brittle though, if the images are converted to a lossy format like JPEG then YOLOv3 can see them again.
![AdversarialYoloDemo](/demos/stop-sign-adversarial-demo.gif)
