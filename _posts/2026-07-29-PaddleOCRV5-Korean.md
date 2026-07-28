---
layout: post
title:  "PaddleOCR V5 Korean"
date:   2026-07-29 08:40:25 +0900
categories: OrangePi
comments: true
tags: orangepi tips
---

## [PaddleOCRV5 Korean](https://github.com/darkice9x/PaddleOCRV5_Korean)
Deploy PaddleOCR V5 Korean to RK3588, optimized for rknpu.

## 1. Model 변환
준비사항

-PC

    *Python 3.11
    *rknn-toolkit2 : rknn_toolkit2-2.3.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    pip install rknn_toolkit2-2.3.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    pip install setuptools==69.5.1
    pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    pip install paddleocr
    paddlex --install paddle2onnx
### 1.1 PaddleOCR V5 Detection
  ~~~bash
  cd Models
  paddleocr text_detection -i ./hangul.png \
  --model_name PP-OCRv5_mobile_det \
  --engine onnxruntime
  cp ~/.paddlex/official_models/PP-OCRv5_mobile_det_onnx/inference.onnx ./PP-OCRv5_mobile_det.onnx
  python convert_det.py  ./PP-OCRv5_mobile_det.onnx  rk3588 fp PP-OCRv5_mobile_det.rknn
  ~~~

### 1.2 PaddleOCR V5 Recognition
  ~~~bash
  cd Models
  paddleocr text_recognition -i ./hangul.png \
  --model_name korean_PP-OCRv5_mobile_rec \
  --engine onnxruntime
  cp ~/.paddlex/official_models/korean_PP-OCRv5_mobile_rec_onnx/inference.onnx ./korean_PP-OCRv5_mobile_rec.onnx
  cp ~/.paddlex/official_models/korean_PP-OCRv5_mobile_rec_onnx/inference.yml ./korean_dic.yml
  python convert_rec.py  ./korean_PP-OCRv5_mobile_rec.onnx  rk3588 fp korean_PP-OCRv5_mobile_rec.rknn
  ~~~

## 2. 사용예
### 1. 입력 이미지
<img src="https://github.com/darkice9x/PaddleOCRV5_Korean/blob/main/hangul.png" width="50%">

### 2. 처리 결과 이미지
<img src="https://github.com/darkice9x/PaddleOCRV5_Korean/blob/main/output/hangul_det_rec_rknn.png" width="50%">