# Adversarial Attack on Face 


## Face Recognition Models

Please put the downloaded weights in a local directory called "weights" under each model directory (or change location in the [config](https://github.com/AlonZolfi/AdversarialMask/blob/master/patch/config.py) file).
### ArcFace and CosFace

Code is taken from [here](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch).
Download weights from [here](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d).

### MagFace

Code is taken from [here](https://github.com/IrvingMeng/MagFace).
Download weights from [here](https://drive.google.com/file/d/1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H/view).

## Landmark Detection Models

### MobileFaceNet

Code is taken from [here](https://github.com/cunjian/pytorch_face_landmark).
Download weights from [here](https://drive.google.com/file/d/1T8J73UTcB25BEJ_ObAJczCkyGKW5VaeY/view?usp=sharing).
(Weights file is already included in this repository under [landmark_detection/pytorch_face_landmark/weights](https://github.com/AlonZolfi/AdversarialMask/tree/master/landmark_detection/pytorch_face_landmark/weights)).

### Face Alignment

Code is taken from [here](https://github.com/1adrianb/face-alignment).
Weights are downloaded automatically on the first run.

Note: this model is more accurate, however, it is a lot larger than MobileFaceNet and requires a large memory GPU to be able to backpropagate when training the adversarial mask.



## GAN Inversion
With stylegan2-encoder-pytorch:
We have a good performance, but cannot manipulation

With idinvert_pytorch:
We have a bad performance, but can manipulation (with correct boundary)


## DiffPure
purify adv image

## DiffusioinCLIP
use it to conduct adv attack with diffusion model

##ddim
train our diffusion model on NIO dataset, which can bu used by DiffPure and DiffusionCLIP



## Datasets

### NIO

## Installation

Install the required packages in [req.txt](https://github.com/AlonZolfi/AdversarialMask/tree/master/req.txt).

## Usage

### Configuration

Configurations can be changed in the [config](https://github.com/AlonZolfi/AdversarialMask/blob/master/patch/config.py) file.

### Train

Run the [patch/train.py](https://github.com/AlonZolfi/AdversarialMask/blob/master/patch/train.py) file.

### Test

Run the [patch/test.py](https://github.com/AlonZolfi/AdversarialMask/blob/master/patch/test.py) file. Specify the location of the adversarial mask image in main function.


