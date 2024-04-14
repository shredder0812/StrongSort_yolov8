# Object Detection and Tracking with YOLOv8 and StrongSORT

## Introduction

This project implements a real-time object detection and tracking system using YOLOv8 and StrongSORT. Object detection is a fundamental task in computer vision that involves identifying and locating objects of interest within images or video frames. Object tracking, on the other hand, involves following the movement of objects over time in video streams.

## Objectives

The primary objective of this project is to develop a robust and efficient system for detecting and tracking objects in real-time video streams. By combining the YOLOv8 object detection model with the StrongSORT object tracking algorithm, we aim to achieve high accuracy and real-time performance.

## Features

- **Real-time Object Detection:** The YOLOv8 model is capable of detecting objects in real-time with high accuracy, making it suitable for applications such as surveillance, traffic monitoring, and more.

- **Object Tracking:** The StrongSORT algorithm provides robust object tracking capabilities, allowing the system to track objects across multiple frames even in challenging scenarios such as occlusions and cluttered backgrounds.

- **Customizable Configuration:** The system's parameters can be easily customized to adapt to different environments and scenarios. Users can adjust parameters such as detection and tracking thresholds, maximum object age, and more.

## Use Cases

This project has various potential applications across different domains:

- **Surveillance:** The system can be deployed for real-time surveillance applications, such as monitoring public spaces, detecting suspicious behavior, and identifying unauthorized objects.

- **Traffic Monitoring:** In transportation systems, the system can be used for real-time traffic monitoring, vehicle counting, and detecting traffic violations.

- **Retail Analytics:** Retailers can utilize the system for tracking customer behavior, analyzing foot traffic patterns, and optimizing store layouts.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/object_detection.git
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the file:
   ```bash
   python yolov8.py

Make sure you have a webcam connected to your computer or adjust the capture_index variable in ObjectDetection class constructor to use a different video source.
