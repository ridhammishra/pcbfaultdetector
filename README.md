# PCB Fault Detector

**AI-Based PCB Defect Detection Using YOLO26n, NCNN and Raspberry Pi 4**

## Introduction

Printed Circuit Boards (PCBs) are essential components of almost every electronic device. During manufacturing, defects such as broken connections, missing holes, unwanted copper formations and short circuits can affect the reliability of electronic products.

Traditional manual inspection can be time-consuming and inconsistent, especially when production volumes are high. Small defects may go unnoticed, resulting in defective products, material wastage and increased manufacturing costs.

Our project aims to address this problem by developing an AI-based PCB inspection system that automatically detects and localizes manufacturing defects. We use computer vision and edge computing to perform inspection directly on Raspberry Pi 4, making the system less dependent on expensive computing infrastructure or cloud services.

## Hardware and System Setup

**Raspberry Pi 4**

We use Raspberry Pi 4 as the main computing unit. It is powered by the Broadcom BCM2711 processor, which uses a quad-core 64-bit ARM Cortex-A72 architecture.

Since Raspberry Pi 4 uses an ARM-based processor, it provides a compact and relatively low-power platform for deploying AI applications. It also supports camera connectivity and Python-based computer vision libraries.

**Raspberry Pi Camera Module**

The camera captures PCB images continuously using the Picamera2 library. These frames are passed to the detection pipeline for processing and defect identification.

## Object Detection Using YOLO26n

We use YOLO26n from Ultralytics for detecting PCB defects.

YOLO (You Only Look Once) is an object detection approach that identifies objects and their locations within an image. Unlike image classification, it can detect multiple defects in a single PCB image and mark each one using a bounding box.

At a high level, the detection architecture consists of three components:

- **Backbone:** Extracts important visual features such as edges, textures and patterns.
- **Neck:** Combines features at different resolutions to help detect defects of different sizes.
- **Detection Head:** Predicts defect categories, bounding-box coordinates and confidence scores.

We selected YOLO26n to explore lightweight object detection on resource-constrained hardware.

## Dataset and Training

We use the PCB Defect Detection Ultra dataset from Roboflow Universe.

Dataset: https://universe.roboflow.com/cnn-pcb-defect-detection/pcb-defect-detection-ultra

The model is trained to detect six types of PCB defects:

1. Missing Hole
2. MouseBite
3. Open Circuit
4. Short Circuit
5. Spur
6. Spurious Copper

Training is performed using Python and the Ultralytics framework. We initialize YOLO26n using its model configuration and train it on the annotated PCB dataset.

Our training configuration includes:

| Parameter | Value |
|---|---|
| Model | YOLO26n |
| Epochs | 100 |
| Image Size | 640 × 640 |
| Batch Size | 4 |
| Device | CPU |
| Pretrained Weights | Disabled |

The trained model is exported into NCNN format for deployment on Raspberry Pi 4.

## Edge Deployment Using NCNN

Running a deep learning model on resource-constrained hardware requires an efficient inference framework. We use NCNN, a neural network inference framework designed for mobile and edge devices.

The trained YOLO26n model is exported into two files:

- `model.ncnn.param` contains the model architecture and layer configuration.
- `model.ncnn.bin` contains the trained model weights.

NCNN loads these files and performs inference directly on Raspberry Pi 4 using its ARM-based CPU. Our current implementation uses four CPU threads with Vulkan acceleration disabled.

## Python Inference Pipeline

The live detection implementation is available in `liveinference.py`.

The complete process involves the following steps:

1. **Image Capture:** Picamera2 captures PCB images through the Raspberry Pi Camera.
2. **Preprocessing:** OpenCV performs center cropping and resizing. The image is converted into NCNN format and normalized.
3. **Inference:** The preprocessed image is passed to the trained YOLO26n model through NCNN.
4. **Postprocessing:** Predictions are filtered using a confidence threshold, followed by Non-Maximum Suppression (NMS) to remove overlapping detections.
5. **Visualization:** OpenCV draws bounding boxes and displays defect labels, confidence scores, FPS and inference time.

**Training Pipeline**

```text
PCB Dataset
     |
     v
Data Preparation
     |
     v
YOLO26n Training
     |
     v
Model Validation
     |
     v
Trained Weights
     |
     v
NCNN Export
```

**Live Inference Pipeline**

```text
Raspberry Pi Camera
     |
     v
Image Capture
     |
     v
Preprocessing
     |
     v
NCNN Inference
     |
     v
Defect Detection
     |
     v
Confidence Filtering and NMS
     |
     v
Bounding Box Visualization
```

## Performance and Limitations

The current system performs inference locally on Raspberry Pi 4. However, the FPS is relatively low due to the limited computational resources of the ARM-based CPU and the absence of dedicated AI acceleration hardware.

Improving inference speed through model optimization, quantization and compatible hardware acceleration is an important area for future development.

## Technologies Used

Python, YOLO26n, Ultralytics, PyTorch, NCNN, OpenCV, NumPy, Picamera2, Raspberry Pi 4 and Roboflow.

## References

- Dataset: https://universe.roboflow.com/cnn-pcb-defect-detection/pcb-defect-detection-ultra
- YOLO Research Paper: https://arxiv.org/abs/1506.02640
- Ultralytics: https://docs.ultralytics.com/
- NCNN: https://github.com/Tencent/ncnn
- Raspberry Pi: https://www.raspberrypi.com/documentation/

**Repository:** https://github.com/ridhammishra/pcbfaultdetector

*Our objective is to make PCB quality inspection more accessible by combining deep learning with affordable edge-computing hardware, helping identify manufacturing defects more efficiently.*
