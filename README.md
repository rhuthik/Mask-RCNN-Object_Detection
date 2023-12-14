# Mask-RCNN-Object_Detection

## Overview
This project implements the [Mask R-CNN ](https://arxiv.org/pdf/1703.06870.pdf)(He et al., ICCV 2017) algorithm using OpenCV in Python. Mask R-CNN is an extension of Faster R-CNN that adds a branch for predicting segmentation masks on each Region of Interest (RoI), parallel to the existing branch for classification and bounding box regression. The implementation runs efficiently on GPUs and can also be executed on CPUs for testing purposes.

Our implementation uses the InceptionV2 backbone for a balance between speed and accuracy, but it can be adapted to use more powerful backbones like ResNeXt-101 for improved performance.

## Installation
### Step 1: Download the Model
Download and extract the pre-trained Mask R-CNN InceptionV2 model:
```
wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar zxvf mask_rcnn_inception_v2_coco_2018_01_28.tar.gz
```
### Step 2: Initialize Parameters
Set the confidence and mask thresholds in the script. The default values are:
0.5  # Confidence threshold
0.3  # Mask threshold
### Step 3: Load the Model and Classes
The script reads class names from mscoco_labels.names and color mappings from colors.txt. The model is loaded with the frozen_inference_graph.pb and mask_rcnn_inception_v2_coco_2018_01_28.pbtxt files.

### Step 4: Process the Input
The script accepts both image and video inputs. For video inputs, including webcam streams, it processes each frame and saves the output with bounding boxes and segmentation masks.

## Usage
To run the Mask R-CNN on an image:
```
python3 mask_rcnn.py --image=cars.jpg
```
To run the Mask R-CNN on a video:
```
python3 mask_rcnn.py --video=cars.mp4
```
By default, if no argument is provided, the script starts the webcam:
```
python3 mask_rcnn.py
```
## Results
### Image Output
![image output](/main/cars_mask_rcnn_out_py.jpg)

### Video Output
Link to video result: Mask R-CNN Result on Video
