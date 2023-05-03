#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/4/28 09:37
# @File  : t2.py
# @Author: 
# @Desc  :
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import ultralytics
from ultralytics.yolo.data import utils

model = YOLO('yolov8n.pt')

yaml_path = str(Path(ultralytics.__file__).parent / 'datasets/coco128.yaml')
class_names = utils.yaml_load(yaml_path)['names']

def print_detection(boxes, class_names, min_score=0.2):
    for *box, conf, cls in reversed(boxes):
        if conf>min_score:
            label = f'{class_names[int(cls)]} {conf:.2f}'
            print(label)

img_path = "/Users/admin/Downloads/v0200f320000c0rjic78pv7mjonbo7jg_0.jpg"
img = Image.open(img_path).convert('RGB')
result = model.predict(source=img)
print(f"结果是:")
print(result[0].boxes.boxes)
print_detection(result[0].boxes.boxes, class_names, min_score=0.2)

