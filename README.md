PCB FAULT DETECTOR

Real-Time PCB Defect Detection Using YOLO26n, NCNN and Raspberry Pi 4

1. INTRODUCTION

Printed Circuit Boards (PCBs) are used in almost every electronic product, from smartphones and computers to automotive systems and industrial equipment. The quality of a PCB directly affects the reliability and performance of the final product.
During manufacturing, defects such as broken connections, unwanted copper formations, missing holes and short circuits can occur. Even a small defect can make an entire board unusable.
In many manufacturing environments, PCB inspection still involves manual checking or expensive automated inspection systems. Manual inspection becomes difficult when production volumes are high, and small defects can easily go unnoticed. This can lead to defective products, increased production costs and unnecessary material wastage.
Our project aims to address this problem using computer vision and edge AI. We have developed a PCB defect detection system that captures images through a Raspberry Pi Camera and uses a trained YOLO26n model to identify and locate defects.
The main idea is to perform inspection directly on a compact computing device rather than depending on a cloud server or a high-end computer. This makes the approach suitable for exploring affordable AI-assisted quality control in manufacturing.

2. PROJECT OVERVIEW

The system captures PCB images using a Raspberry Pi Camera Module. These images are processed by a trained YOLO26n object detection model, which identifies defects and marks their locations using bounding boxes.
We use NCNN as the inference framework to run the exported model on Raspberry Pi 4.
The current implementation detects six types of PCB defects:
1. Missing Hole
2. MouseBite
3. Open Circuit
4. Short Circuit
5. Spur
6. Spurious Copper
For every detected defect, the system displays its predicted category and confidence score.

3. HARDWARE SETUP
Raspberry Pi 4
We use Raspberry Pi 4 as the main computing device for our project. It is a compact single-board computer powered by the Broadcom BCM2711 processor, which uses a quad-core 64-bit ARM Cortex-A72 architecture.
The ARM-based processor allows us to explore the deployment of deep learning models on relatively low-power hardware.
We selected Raspberry Pi 4 because it is affordable, compact and supports Linux-based operating systems. It also provides the necessary interfaces to connect a camera and run Python-based computer vision applications.
One of the main challenges is that Raspberry Pi 4 has limited computational resources compared with systems equipped with dedicated GPUs. Therefore, choosing an appropriate model and inference framework is important for running the application efficiently.

Raspberry Pi Camera Module
The Raspberry Pi Camera Module is used to capture images of the PCB during inspection.
We use the Picamera2 library to initialize the camera and continuously capture frames. These frames are then passed to our Python inference pipeline for defect detection.
The camera is configured to capture images at 640 × 480 resolution.

4. OBJECT DETECTION USING YOLO26n

For detecting PCB defects, we use YOLO26n, a lightweight object detection model from Ultralytics.
YOLO stands for You Only Look Once. Unlike image classification models, which assign a label to an entire image, object detection models can identify multiple objects and their locations within the same image.
This is particularly useful for PCB inspection because a single board can contain multiple defects at different positions.
We selected YOLO26n because we wanted to experiment with a lightweight detection model that could eventually run on resource-constrained hardware such as Raspberry Pi 4.

How YOLO Works

At a high level, an object detection network consists of three major components.
Backbone
The backbone extracts useful visual features from the input image, such as edges, textures and shapes. These features help the model distinguish between normal PCB structures and defective regions.
Neck
The neck combines features from different stages of the network. This helps the model identify objects of different sizes, which is important because PCB defects can be very small.
Detection Head
The detection head generates predictions for the input image. These predictions contain bounding-box coordinates, class scores and confidence information.
The final predictions are processed to remove overlapping detections and obtain the defect locations.

5. DATASET PREPARATION
We use the PCB Defect Detection Ultra dataset available on Roboflow Universe.

Dataset link:
https://universe.roboflow.com/cnn-pcb-defect-detection/pcb-defect-detection-ultra

The dataset contains PCB images annotated with different types of manufacturing defects.

6. MODEL TRAINING

We train our model using Python and the Ultralytics YOLO framework. The training implementation is available in trainmodel.ipynb.

We initialize the YOLO26n architecture using its YAML configuration and train it on the PCB defect dataset.

Our current training configuration is:

Model: YOLO26n
Training epochs: 100
Image size: 640 × 640
Batch size: 4
Device: CPU
Pretrained weights: Disabled
Workers: 4

We use pretrained=False, which means the model is trained from scratch instead of starting with pretrained weights.

The training process involves preparing the dataset, initializing the model, training it on annotated images and evaluating its performance using validation data.

After training, the model weights are saved and prepared for deployment.

7. MODEL DEPLOYMENT USING NCNN

Training a deep learning model and running it on an edge device are two different challenges. A model that works well during training may require a more suitable inference runtime for deployment on resource-constrained hardware.

To address this, we use NCNN, a neural network inference framework designed for mobile and edge devices.

We export the trained YOLO26n model into NCNN format. The exported model contains two important files:

- model.ncnn.param, which stores the model structure and layer configuration.
- model.ncnn.bin, which contains the model weights.

We also use a metadata.yaml file to store information such as the model input dimensions and defect class names.
NCNN allows us to load and execute the exported model directly on Raspberry Pi 4 without running the original training pipeline.
In our current implementation, NCNN is configured to use four CPU threads, while Vulkan acceleration is disabled.
This allows the model to run using the ARM-based processor of Raspberry Pi 4.
8. PYTHON IMPLEMENTATION

Our implementation has two main stages: model training and live inference.
Training Pipeline
The training pipeline starts with the PCB defect dataset. After downloading and preparing the images and annotations, we configure the YOLO26n model and train it using the Ultralytics framework.
The trained model is then exported into NCNN format so that it can be used on Raspberry Pi 4.

Training flow:
PCB Defect Dataset
        |
Dataset Preparation
        |
YOLO26n Initialization
        |
Model Training
        |
Model Validation
        |
Trained Model Weights
        |
NCNN Model Export
        |
NCNN Model Files
Live Inference Pipeline
The live inference implementation is available in liveinference.py.
This script connects the Raspberry Pi Camera with the NCNN model and performs continuous defect detection.
The process begins by initializing the camera using Picamera2. Each captured frame is passed through preprocessing before being sent to the NCNN inference engine.
The main steps are explained below.

Image Capture
The Raspberry Pi Camera continuously captures images of the PCB.
Image Preprocessing

The captured frame is center-cropped into a square region and resized to the model input dimensions. The image is then converted into the NCNN matrix format and normalized by dividing pixel values by 255.
Model Inference
The preprocessed image is passed to the trained NCNN model. The model processes the image and generates predictions containing bounding-box information and class scores.

Defect Classification
The predicted class scores are examined to identify the most likely defect category. Predictions below the confidence threshold are discarded.
Non-Maximum Suppression

We use Non-Maximum Suppression (NMS) to remove overlapping duplicate bounding boxes. NMS is applied separately to each defect category.

Result Visualization
The final bounding boxes are mapped back to the original camera frame. OpenCV is used to draw rectangles around detected defects and display their class names and confidence scores.
The application also displays the current FPS, inference time and total number of detections.

Inference flow:
Raspberry Pi Camera
        |
Capture Image Frame
        |
Image Preprocessing
        |
NCNN Input Conversion
        |
Pixel Normalization
        |
YOLO26n Inference
        |
Prediction Processing
        |
Confidence Filtering
        |
Non-Maximum Suppression
        |
Defect Localization
        |
Display Detection Results

9. EDGE DEPLOYMENT ON RASPBERRY PI 4
One of the main objectives of this project is to run the trained model directly on Raspberry Pi 4.
The complete inference process takes place locally. Images captured by the camera are processed on the device, and the detection results are displayed without requiring a cloud server.
We use Picamera2 for image acquisition, OpenCV for image processing, NumPy for numerical operations and NCNN for model inference.
This approach reduces dependence on external computing infrastructure and allows the system to operate without continuous internet connectivity.
However, Raspberry Pi 4 has limited processing capabilities compared with dedicated GPU-based systems.
In our current implementation, the FPS is relatively low because inference is performed using the onboard ARM CPU without dedicated AI acceleration hardware.
This is an important limitation of the current prototype. Improving inference speed while maintaining detection accuracy is one of the areas we would like to explore further.

10. CURRENT LIMITATIONS AND FUTURE WORK

The current implementation demonstrates how a trained object detection model can be deployed on an ARM-based edge device for PCB inspection.
However, there are still some areas that can be improved.
The most noticeable limitation is the relatively low inference FPS on Raspberry Pi 4. Since the current setup relies on CPU-based inference, processing every frame in real time is computationally demanding.
In future versions, we would like to explore model quantization, further inference optimization and compatible AI accelerators to improve performance.
Other improvements may include better camera positioning, controlled lighting conditions, improved defect detection for very small regions and an interface for maintaining inspection records.
Our long-term aim is to develop a practical PCB inspection system that can assist quality-control operators in identifying manufacturing defects more efficiently.

11. TECHNOLOGIES USED

- Python
- YOLO26n
- Ultralytics
- PyTorch
- NCNN
- OpenCV
- NumPy
- Picamera2
- Raspberry Pi 4
- Roboflow Universe

12. REFERENCES

Dataset

PCB Defect Detection Ultra
https://universe.roboflow.com/cnn-pcb-defect-detection/pcb-defect-detection-ultra

YOLO Research Paper

Joseph Redmon, Santosh Divvala, Ross Girshick and Ali Farhadi.

You Only Look Once: Unified, Real-Time Object Detection. CVPR, 2016.

https://arxiv.org/abs/1506.02640

Ultralytics Documentation
https://docs.ultralytics.com/

NCNN Framework
https://github.com/Tencent/ncnn

Raspberry Pi Documentation
https://www.raspberrypi.com/documentation/

PROJECT REPOSITORY

https://github.com/ridhammishra/pcbfaultdetector

Developed as a prototype for AI-assisted industrial quality inspection under the Manufacturing and Industry 4.0 theme.

Our goal is to explore how computer vision and affordable edge hardware can make PCB quality inspection more accessible, efficient and reliable.
