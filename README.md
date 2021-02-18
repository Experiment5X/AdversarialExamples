# Adversarial Examples for AWS Rekognition
Getting started with creating adversarial images to trick AWS Rekognition!

## Basic VGG Adversarial Attack
So far I can create basic adversarial images to trick VGG-16.
![AdversarialVGGDemo](/demos/vgg_adversarial.gif)

## YOLOv3 Basic Adversarial Attacks
I can also create basic adversarial images to trick an object detector, YOLOv3:
![AdversarialYoloDemo](/demos/yolo_adversarial.gif)

Here I ran my YOLOv3 adversarial attack on several different stop sign images, where the stop signs are at various distances from the camera. My attack is able to work for all of the images. These adversarial examples are very brittle though, if the images are converted to a lossy format like JPEG then YOLOv3 can see them again.
![AdversarialStopSignYoloDemo](/demos/stop-sign-adversarial-demo.gif)

## Dispersion Reduction Attack on YOLOv3
I implemented dispersion reduction which is a method of creating adversarial images where you try to attack basic image features that would be common across several models. This is done by reducing the standard deviation of a feature map, with the hope that there will be less useful information in the image when the feature map activation variance is reduced. For my experiment I used the 19th layer in the VGG model which is a convolutional layer with 512 filters that are 3x3 in size.

As you can see in this GIF, the attack only seems to work for objects that are small or not very pronounced in the image. For the image of the skiers it was able to hide the skis and the backpack, but not the person. For the stop sign image it wasn't able to hide the stop sign at all, probably because stop signs are designed to be very pronounced. The font of the stop sign is common and there is a lot of contrast between the white letters and the red stop sign. This is exciting though because this is the first time seeing a black box attack work. The adversarial image was created using VGG-16 and it was able to fool YOLOv3.

Dispersion reduction may work well when used in combination with an ensemble of object detectors. If I create an adversarial example that can fool YOLOv3 and FasterRCNN while also having a low feature map dispersion in VGG that may have a good chance at fooling AWS.

Dispersion reduction paper: https://arxiv.org/abs/1911.11616
![DispersionReductionDemo](/demos/disperson_reduction.gif)

## Ensemble Adversarial
I created a script that produces adversarial images for YOLOv3 and Faster-RCNN simultaneously. The resulting adversarial images are able to fool both YOLOv3 and Faster-RCNN as shown in the GIF below. However, these images are significantly distorted. The attacks on a single model require much less distortion than is required to trick multiple models. More work will need to be done to try and reduce the amount of changes needed for an image to become adversarial. The adversarial images shown in this GIF are in this repo at `/test_images/ensemble_adversarials/`.

![EnsembleAdversarialDemo](/demos/ensemble-adversarial1.gif)