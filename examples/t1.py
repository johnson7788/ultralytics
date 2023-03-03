#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/3/2 14:16
# @File  : t1.py
# @Author: 
# @Desc  :

from ultralytics import YOLO

# Load a model
model = YOLO("../yolov8n.pt")  # load a pretrained model (recommended for training)
# success = model.export(format="onnx")  # export the model to ONNX format
# Use the model
results = model("bus.jpg")  # predict on an image
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmenation masks outputs
    probs = result.probs  # Class probabilities for classification outputs
    names = result.names  # Class probabilities for classification outputs
    print(boxes, masks, probs)
    boxes_cordinates = boxes.boxes
    boxes_cls = boxes.cls
    boxes_confidence = boxes.conf
    print(boxes_cls)
