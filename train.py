#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/3/3 13:17
# @File  : train.py.py
# @Author: 
# @Desc  :


from ultralytics import YOLO
model = YOLO(model="yolov8n.pt") # pass any model type
# model.train(data="all.yaml", epochs=1)
results = model.predict(source="folder", show=True)
print(results)