---
layout: post
title:  "FaceRecognition RKNN"
date:   2025-09-29 08:44:43 +0900
categories: OrangePi
comments: true
tags: orangepi tips
---

## [FaceRecognitionRKNN](https://github.com/darkice9x/FaceRecongnitionRKNN)
Deploy face recognition(retinaface + mobilefacenet) to RK3588S, optimized for rknpu.

## 1. Model 변환
준비사항

-PC

    *Python 3.10
    *rknn-toolkit2 : rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
### 1.1 MobileFaceNet
  ~~~bash
  cd rk3588_mobilefacenet
  python mobilefacenetConvert.py
  ~~~
facenet_mxnet_caffe만 가능
### 1.2 RetinaFace
  ~~~bash
  cd convert
  python convert.py
  ~~~

## 2. 사용법
준비사항

-BOARD
  ~~~bash
  conda create -n pyside6rknn python=3.12 -y
  conda activate pyside6rknn
  pip install pyside6==6.9.0
  conda install -c conda-forge libstdcxx-ng --update-deps -y
  pip install onvif-client
  pip install faiss-cpu
  pip install rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
  ~~~
-Python 
  ~~~python
  import cv2
  from embeddings import Embeddings
  from rknnlite.api import RKNNLite
  
  embeddings = Embeddings('./RetinaFace_mobile320_i8_v2.3.2.rknn', './rk3588_mobilefacenet/mobilefacenet_v2.3.2.rknn')
  img1 = cv2.imread(image1)
  get_face1 = embeddings.get_embeddings(img1)
  feature1 = get_face1[0]['embedding']
  img2 = cv2.imread(image2)
  get_face2 = embeddings.get_embeddings(img2)
  feature2 = get_face2[0]['embedding']
  match, cosine_similarity = embeddings.compare_face(feature1, feature2)
  print( f'name: {personname}, match: {match}, face_distances : {cosine_similarity}'  )
  ~~~
  match가 True이고  cosine_similarity가 가장 큰값이 가장 근접한 인식이다. 

-faiss Python 
  ~~~python
  import cv2
  from face_faiss_db import FaceDB
  from embeddings import Embeddings
  from rknnlite.api import RKNNLite
  
  embeddings = Embeddings('./RetinaFace_mobile320_i8_v2.3.2.rknn', './rk3588_mobilefacenet/mobilefacenet_v2.3.2.rknn')
  db = FaceDB(dim=128, db_path="./")
  img1 = cv2.imread(image1)
  get_face1 = embeddings.get_embeddings(img1)
  feature1 = get_face1[0]['embedding']
  result = db.search(feature1)
    if result :
        print(f'Face recognized: {result[0]["name"]}')
    else:
        print('Face not recognized')
        db.add("personname", feature1)
  ~~~
