# PCB Fault Detector

**AI-Based PCB Defect Detection Using YOLO26n, NCNN and ExecuTorch on Raspberry Pi 4**

## 1. The Problem We Are Solving

Printed Circuit Boards (PCBs) are essential components of electronic devices. During manufacturing, defects such as missing holes, broken connections, unwanted copper formations and short circuits can affect the reliability of the final product.

Manual PCB inspection can be time-consuming and inconsistent, particularly when production volumes are high. Small defects may go unnoticed, resulting in material wastage, increased costs and defective products.

Our project aims to address this problem through an AI-powered visual inspection system. We use computer vision to automatically identify and locate PCB defects, making quality inspection faster and more consistent.

Our approach focuses on running the detection system directly on affordable edge hardware rather than depending on cloud-based processing or expensive computing infrastructure.

## 2. Hardware Setup

**Raspberry Pi 4**

We use Raspberry Pi 4 as the main computing device. It is powered by the Broadcom BCM2711 processor, which uses a quad-core 64-bit ARM Cortex-A72 architecture.

Since Raspberry Pi 4 is ARM-based, it provides a compact and low-power platform for deploying AI applications. Its camera interface and Linux support make it suitable for our edge-based inspection system.

**Raspberry Pi Camera Module**

The camera captures PCB images using the Picamera2 library. These images are passed to the detection pipeline for processing and defect identification.

## 3. PCB Defect Detection Using YOLO26n

We use YOLO26n from Ultralytics for object detection.

YOLO (You Only Look Once) identifies objects and their locations within an image. Unlike image classification, it can detect multiple defects in a single PCB image and mark each one using a bounding box.

At a high level, its architecture consists of three components:

- **Backbone:** Extracts visual features such as edges, textures and patterns.
- **Neck:** Combines features at different resolutions to help detect defects of different sizes.
- **Detection Head:** Predicts defect categories, bounding boxes and confidence scores.

Our model is trained to detect six PCB defect categories.

## 4. PCB Defect Categories

The following images illustrate the six types of manufacturing defects considered in our project.

![PCB Defect Categories](assets/pcb-defect-categories.png)

The model detects:

1. Missing Hole
2. MouseBite
3. Open Circuit
4. Short Circuit
5. Spur
6. Spurious Copper

## 5. Dataset and Model Training

We use the PCB Defect Detection Ultra dataset from Roboflow Universe.

**Dataset:** https://universe.roboflow.com/cnn-pcb-defect-detection/pcb-defect-detection-ultra

The model is trained using Python and the Ultralytics YOLO framework. We initialize YOLO26n using its model configuration and train it on the annotated PCB images.

Our training configuration:

| Parameter | Value |
|---|---|
| Model | YOLO26n |
| Epochs | 100 |
| Image Size | 640 × 640 |
| Batch Size | 4 |
| Training Device | CPU |
| Pretrained Weights | Disabled |

The trained model is prepared for deployment using two edge inference frameworks: NCNN and ExecuTorch.

## 6. Edge Deployment Using NCNN and ExecuTorch

Running deep learning models on Raspberry Pi 4 requires an inference runtime suitable for resource-constrained ARM-based hardware.

We explore two deployment frameworks:

**NCNN**

NCNN is a lightweight neural network inference framework optimized for mobile and embedded devices. It supports CPU-based inference and provides an efficient way to deploy trained models on ARM hardware.

Our NCNN deployment uses exported `.param` and `.bin` model files and performs inference using four CPU threads.

Documentation: https://github.com/Tencent/ncnn

**ExecuTorch**

ExecuTorch is Meta's open-source solution for running AI models on edge devices. It is built around the PyTorch ecosystem and provides a lightweight runtime for deploying models across different hardware platforms.

We also evaluate ExecuTorch as an alternative inference runtime for our Raspberry Pi-based PCB inspection system.

Documentation: https://docs.pytorch.org/executorch/stable/

The trained model is prepared for both NCNN and ExecuTorch deployment, allowing us to compare their performance at different input resolutions.

## 7. Python Inference Pipeline

The live inference implementation uses Raspberry Pi Camera, Picamera2, OpenCV, NumPy and the selected inference runtime.

The process involves:

1. Capturing PCB images through the Raspberry Pi Camera.
2. Cropping and resizing images according to the model input resolution.
3. Converting and normalizing images for model inference.
4. Running the trained YOLO26n model using NCNN or ExecuTorch.
5. Applying confidence filtering and Non-Maximum Suppression (NMS).
6. Drawing bounding boxes and displaying defect labels and confidence scores.

**Training and Deployment Flow**

```text
PCB Defect Dataset
        |
        v
Dataset Preparation
        |
        v
YOLO26n Training
        |
        v
Trained Model Weights
        |
        v
Model Export
        |
        +------------------+
        |                  |
        v                  v
      NCNN             ExecuTorch
        |                  |
        v                  v
   Edge Inference     Edge Inference
        |                  |
        +--------+---------+
                 |
                 v
          Defect Detection
```

## 8. Performance Benchmark

We evaluated both inference frameworks on Raspberry Pi 4 at two different input resolutions.

| S.No. | Runtime | Resolution | Framerate (FPS) | Latency (ms) |
|---:|---|---:|---:|---:|
| 1 | NCNN | 640 × 640 | 1.7 | 588.2 |
| 2 | NCNN | 480 × 480 | 3.6 | 277.8 |
| 3 | ExecuTorch | 640 × 640 | 1.9 | 526.3 |
| 4 | ExecuTorch | 480 × 480 | 3.7 | 270.3 |

Latency is calculated approximately as:

`Latency (ms) = 1000 / FPS`

The results show that reducing the input resolution from 640 × 640 to 480 × 480 improves the processing speed for both frameworks.

ExecuTorch achieves slightly higher FPS in both comparisons based on our reported measurements.

**Note:** These are the supplied benchmark results. Latency represents approximate frame time derived from FPS, not independently measured model-only execution latency.

## 9. Limitations and Future Improvements

The current system achieves relatively low FPS because Raspberry Pi 4 has limited computational resources and our implementation does not use dedicated AI acceleration hardware.

Future improvements include model quantization, inference optimization, hardware acceleration and better camera positioning and lighting conditions.

Our goal is to develop a practical and affordable edge-AI inspection system that can assist quality-control operators in identifying manufacturing defects.

## 10. Technologies Used

- Python
- YOLO26n
- Ultralytics
- PyTorch
- NCNN
- ExecuTorch
- OpenCV
- NumPy
- Picamera2
- Raspberry Pi 4
- Roboflow Universe

## 11. References

- PCB Defect Detection Dataset: https://universe.roboflow.com/cnn-pcb-defect-detection/pcb-defect-detection-ultra
- Ultralytics Documentation: https://docs.ultralytics.com/
- ExecuTorch Documentation: https://docs.pytorch.org/executorch/stable/
- NCNN Documentation: https://github.com/Tencent/ncnn
- Raspberry Pi Documentation: https://www.raspberrypi.com/documentation/
- YOLO Research Paper: https://arxiv.org/abs/1506.02640

**Repository:** https://github.com/ridhammishra/pcbfaultdetector

Developed under the Manufacturing and Industry 4.0 theme.

Our objective is to bring affordable AI-powered visual inspection closer to the manufacturing floor, helping make PCB quality control more efficient and reliable.
