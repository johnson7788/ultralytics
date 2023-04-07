#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/2/20 2:20 下午
# @File  : yolo_api.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 推理和训练的api
######################################################
# 包括训练接口api和预测接口api
# /api/train
# /api/predict
######################################################

import logging
import re
import sys
import json
import os
import hashlib
import time
from pathlib import Path
import requests
import torch
from numpy import random
import argparse
from PIL import Image
import base64
from ultralytics import YOLO
from flask import Flask, request, jsonify, abort

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger("Main")

app = Flask(__name__)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class YOLOModel(object):
    def __init__(self, verbose=0, cpu=False):
        self.verbose = verbose
        self.label_list = ['table','figure','equation','algorithm','title','paragraph','other','pgequation']
        self.label_list_cn = ['表格','图像','公式','算法','标题','段落','其他','段落公式']
        self.label_en2zh = dict(zip(self.label_list, self.label_list_cn))
        #给每个类别的候选框设置一个颜色
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.label_list]
        self.num_labels = len(self.label_list)
        # 判断使用的设备
        if not cpu and torch.cuda.is_available():
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
            logger.info(f"使用GPU进行预测")
        else:
            self.device = torch.device("cpu")
            self.n_gpu = 0
            logger.info(f"使用CPU进行预测")
        # 预测的batch_size大小
        self.train_batch_size = 8
        # 预测的batch_size大小
        self.predict_batch_size = 16
        #模型的名称或路径
        self.weights = 'runs/train/all/weights/best.pt'      # 'yolov5s.pt'
        # self.weights = 'runs/train/exp2/weights/last.pt'      # 'yolov5s.pt'
        self.upload_dir = "runs/api/images"
        self.img_size = 1024   #像素864, 1024
        self.conf_thres = 0.5  #置信度, 大于这个置信度的才类别才取出来
        self.iou_thres = 0.5  #IOU的NMS阈值
        self.view_img = False   #是否显示图片的结果
        self.save_img = True    #保存图片预测结果
        self.save_conf = False  #同时保存置信度到save_txt文本中
        self.classes = None  # 0, 1, 2 ，只过滤出我们希望的类别, None表示保留所有类别
        self.agnostic_nms = False #使用nms算法
        self.project = 'runs/api' #项目保存的路径
        self.image_dir = os.path.join(self.project, 'images')   #保存从网络下载的图片
        self.predict_dir = os.path.join(self.project, 'predict')   #保存预测结果
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        if not os.path.exists(self.predict_dir):
            os.makedirs(self.predict_dir)
        mymodel = YOLO(model=self.weights)
        mymodel.to(self.device)
        self.predict_model = mymodel

    def download(self, image_url, image_path_name, force_download=False):
        """
        根据提供的图片的image_url，下载图片保存到image_path_name
        :param: force_download: 是否强制下载
        :return_exists: 是否图片已经存在，如果已经存在，那么返回2个True
        """
        # 如果存在图片，并且不强制下载，那么直接返回
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.132 Safari/537.36"
        }
        if os.path.exists(image_path_name) and not force_download:
            print(f"{image_path_name}图片已存在，不需要下载")
            return True
        try:
            response = requests.get(image_url, stream=True, headers=headers)
            if response.status_code != 200:
                return False
            with open(image_path_name, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
            return True
        except Exception as e:
            print(f"{image_url}下载失败")
            print(f"{e}")
            return False

    def cal_md5(self,content):
        """
        计算content字符串的md5
        :param content:
        :return:
        """
        # 使用encode
        result = hashlib.md5(content.encode())
        # 打印hash
        md5 = result.hexdigest()
        return md5
    def predict(self, data):
        """
        返回的bboxes是实际的坐标，x1，y1，x2，y2，是左上角和右下角的坐标
        :param data: 图片数据的列表 [image1, image2]
        :return: [[images, bboxes, confidences, labels],[images, bboxes,confidences, labels],...] bboxes是所有的bboxes, confidence是置信度， labels是所有的bboxes对应的label，
        """
        if isinstance(data, str):
            data = [data]
        #下载数据集, images保存图片的本地的路径
        images = []
        #设置数据集
        image_array = []
        for image in data:
            images.append(image)
            im1 = Image.open(image)
            image_array.append(im1)
        if self.device == "cuda":
            device = "0"
        else:
            device = "cpu"
        predict_results = self.predict_model(source=image_array, save=self.save_img, device=device)  # save plotted images
        #计算耗时
        start = time.time()
        # path是图片的路径，img是图片的改变size后的numpy格式[channel, new_height,new_witdh], im0s是原始的图片,[height, width,channel], eg: (2200, 1700, 3), vid_cap 是None如果是图片，只对视频有作用
        results = []
        for image_path, result in zip(images, predict_results):
            boxes = result.boxes  # Boxes object for bbox outputs
            masks = result.masks  # Masks object for segmenation masks outputs
            probs = result.probs  # Class probabilities for classification outputs
            boxes_cordinates = boxes.xyxy
            boxes_cordinates = boxes_cordinates.cpu().numpy()
            boxes_cordinates_list = boxes_cordinates.tolist()
            boxes_confidence = boxes.conf
            boxes_confidence = boxes_confidence.cpu().numpy()
            boxes_confidence_list = boxes_confidence.tolist()
            boxes_cls = boxes.cls
            boxes_cls = boxes_cls.cpu().numpy()
            boxes_cls_list = boxes_cls.tolist()
            label_dict = result.names
            image_shape = result.orig_shape
            labels = []
            for cls in boxes_cls_list:
                label_en_name = label_dict[cls]
                label = self.label_en2zh[label_en_name]
                labels.append(label)
            # 图片的名称，bboex，置信度，标签，都加到结果, 原始图像的尺寸
            one_res = [image_path, boxes_cordinates_list, boxes_confidence_list, labels, image_shape]
            results.append(one_res)
        print(f'Done. ({time.time() - start:.3f}s)')
        return results
    def predict_image(self, image_path):
        """
        返回的bboxes是实际的坐标，x1，y1，x2，y2，是左上角和右下角的坐标
        :param data: 图片数据的列表 [image1, image2]
        :return: [[images, bboxes, confidences, labels],[images, bboxes,confidences, labels],...] bboxes是所有的bboxes, confidence是置信度， labels是所有的bboxes对应的label，
        """
        #检查图片确实存在
        if isinstance(image_path, str):
            logging.info(f"预测本地的一个单张的图片")
            image_pathes = [image_path]
        else:
            logging.info(f"预测本地的图片列表")
            image_pathes = image_path
        for img_path in image_pathes:
            if not os.path.exists(img_path):
                logging.error(f"{img_path}图片不存在")
                return {"code": 500, "msg": f"图片不存在: {img_path}", "data": []}
        prediction = self.predict(image_pathes)
        if isinstance(image_path, str):
            prediction = prediction[0]
        results = {"code": 200, "msg": "success", "data": prediction}
        return results
    def predict_url(self, url):
        """
        :param data: 一个图片的url，下载后预测
        :return: [[images, bboxes, confidences, labels],[images, bboxes,confidences, labels],...] bboxes是所有的bboxes, confidence是置信度， labels是所有的bboxes对应的label，
        """
        #检查图片确实存在
        logging.info(f"预测本地的一个的图片的url地址")
        if isinstance(url, str):
            urls = [url]
        else:
            urls = url
        image_pathes = []
        for ul in urls:
            image_suffix = ul.split('.')[-1]
            md5 = self.cal_md5(ul)
            new_image_name = md5 + '.' + image_suffix
            image_path = os.path.join(self.upload_dir, new_image_name)
            download_reuslt = self.download(image_url=ul, image_path_name=image_path)
            if not download_reuslt:
                results = {"code": 400, "msg": f"图片下载失败，请检查图片地址是否正确: {ul}", "data": []}
                return results
            image_pathes.append(image_path)
        prediction = self.predict(image_pathes)
        if isinstance(url, str):
            prediction = prediction[0]
        results = {"code": 200, "msg": "success", "data": prediction}
        return results
    def predict_raw_image(self, raw_file, image_name=None):
        """
        :param raw_file: 图片的base64 str 格式
        :return:
        """
        #检查图片确实存在
        logging.info(f"预测本地的一个base64 str的图片")
        raw_str = raw_file.encode('utf-8')
        raw_img = base64.b64decode(raw_str)
        if image_name:
            img_name = os.path.join(self.upload_dir, image_name)
        else:
            img_name = os.path.join(self.upload_dir, 'tmp.jpg')
        with open(img_name, 'wb') as f:
            f.write(raw_img)
        prediction = self.predict(img_name)
        results = {"code": 200, "msg": "success", "data": prediction[0]}
        return results
    def predict_directory(self,directory_path):
        """
        预测的是一个目的目录下的所有图片结果
        """
        logging.info(f"预测一个本地的目录，目录包含多张图片")
        if os.path.exists(directory_path):
            data = []
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if file.endswith(".jpg") or file.endswith(".png"):
                        image_path = os.path.join(root, file)
                        data.append(image_path)
            prediction = self.predict(data)
            results = {"code": 200, "msg": "success", "data": prediction}
        else:
            results = {"code": 400, "msg": "给定的目录不存在", "data": ""}
        return results
    def do_train(self, data):
        """
        训练模型, 数据集分成2部分，训练集和验证集, 默认比例9:1
        :param data: 输入的数据，注意如果做truncated，那么输入的数据为 []
        :return:
        """
        logger.info(f"训练完成")
        return "Done"

@app.route("/api/predict", methods=['POST'])
def predict():
    """
    接收POST请求，获取data参数,  bbox左上角的x1，y1, 右下角的x2,y2
    Args:
        test_data: 需要预测的数据，是一个图片的url列表, [images1, images2]
    Returns: 返回格式是[[images, bboxes, confidences, labels,image_size],[images, bboxes,confidences, labels, image_size],...]
    results = {list: 4} [['/Users/admin/git/yolov5/runs/api/images/Reference-less_Measure_of_Faithfulness_for_Grammatical_Er1804.038240001-2.jpg', [[832.0, 160.0, 1495.0, 610.0], [849.0, 1918.0, 1467.0, 2016.0], [204.0, 0.0, 798.0, 142.0]], [0.9033942818641663, 0.8640206456184387, 0.2842876613140106], ['figure', 'equation', 'figure']], ['/Users/admin/git/yolov5/runs/api/images/A_Comprehensive_Survey_of_Grammar_Error_Correction0001-21.jpg', [[864.0, 132.0, 1608.0, 459.0], [865.0, 1862.0, 1602.0, 1944.0], [863.0, 1655.0, 1579.0, 1753.0], [115.0, 244.0, 841.0, 327.0], [116.0, 398.0, 837.0, 486.0], [124.0, 130.0, 847.0, 235.0], [119.0, 1524.0, 830.0, 1616.0], [161.0, 244.0, 799.0, 447.0]], [0.9183754920959473, 0.8920623660087585, 0.8884797692298889, 0.8873556852340698, 0.8276346325874329, 0.5401338934898376, 0.33260053396224976, 0.2832690477371216], ['table', 'equation', 'equation', 'equation', 'equation', 'equation', 'equation', 'equation']], ['/Users/admin/git/yolov5/runs/api/images/2007.158710001-09.jpg', [], ...
     0 = {list: 4} ['/Users/admin/git/yolov5/runs/api/images/Reference-less_Measure_of_Faithfulness_for_Grammatical_Er1804.038240001-2.jpg', [[832.0, 160.0, 1495.0, 610.0], [849.0, 1918.0, 1467.0, 2016.0], [204.0, 0.0, 798.0, 142.0]], [0.9033942818641663, 0.8640206456184387, 0.2842876613140106], ['figure', 'equation', 'figure']]
      0 = {str} '/Users/admin/git/yolov5/runs/api/images/Reference-less_Measure_of_Faithfulness_for_Grammatical_Er1804.038240001-2.jpg'
      1 = {list: 3} [[832.0, 160.0, 1495.0, 610.0], [849.0, 1918.0, 1467.0, 2016.0], [204.0, 0.0, 798.0, 142.0]]
      2 = {list: 3} [0.9033942818641663, 0.8640206456184387, 0.2842876613140106]
      3 = {list: 3} ['figure', 'equation', 'figure']
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    logger.info(f"收到的数据是:{test_data}")
    file_format = jsonres.get('format', "image")
    logger.info(f"参数是:{file_format}")
    if file_format == "image":
        results = model.predict_image(image_path=test_data)
    elif file_format == "directory":
        results = model.predict_directory(directory_path=test_data)
    elif file_format == "base64":
        image_name = jsonres.get('name')
        results = model.predict_raw_image(raw_file=test_data, image_name=image_name)
    elif file_format == "url":
        results = model.predict_url(url=test_data)
    logging.info(f"预测的结果是:{results}")
    return jsonify(results)

@app.route("/api/train", methods=['POST'])
def train():
    """
    接收data参数，
    Args:
        data: 训练的数据，是一个图片列表, [images1, images2,...]
    Returns:
    """
    jsonres = request.get_json()
    data = jsonres.get('data', None)
    logger.info(f"收到的数据是:{data}, 进行训练")
    results = model.do_train(data)
    return jsonify(results)

def parse_args():
    """
    返回arg变量和help
    :return:
    """
    parser = argparse.ArgumentParser(description="YOLO 推理",formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-cpu', "--cpu", action='store_true', help="是否只使用cpu推理")
    return parser.parse_args(), parser.print_help


if __name__ == "__main__":
    arg, helpmsg = parse_args()
    model = YOLOModel(cpu=arg.cpu)
    app.run(host='0.0.0.0', port=5008, debug=False, threaded=True)
