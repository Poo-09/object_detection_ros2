# Object Detection and Tracking in ROS 2 with TurtleBot4

This project demonstrates an object detection and tracking pipeline in ROS 2 using YOLOv8 and OpenCV. The system processes a live camera feed from a simulated TurtleBot4 running in Gazebo (or Ignition), detects objects (e.g., persons, bottles, etc.), tracks them over time using a simple centroid tracker, and publishes detection and tracking information as ROS 2 topics.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Workspace Structure](#workspace-structure)
- [Building the Workspace](#building-the-workspace)
- [Running the Simulation](#running-the-simulation)
  - [Launching TurtleBot4 in Gazebo Classic](#launching-turtlebot4-in-gazebo-classic)
- [Running Object Detection & Tracking](#running-object-detection--tracking)
- [ROS 2 Integration](#ros-2-integration)
- [Teleoperation (Optional)](#teleoperation-optional)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Object Detection:** Uses a pre-trained YOLOv8 (nano) model to detect objects in a live camera feed.
- **Object Tracking:** Implements a simple centroid-based tracker to maintain consistent IDs for detected objects.
- **ROS 2 Integration:** Publishes detected object data (class labels and bounding boxes) and tracking information as ROS 2 topics.
- **Simulation Environment:** Demonstrates simulation with TurtleBot4 in Gazebo Classic (or Ignition) with camera feed.
- **Teleoperation Support (Optional):** Allows for manual control of TurtleBot4 via keyboard or joystick teleop nodes.

## Prerequisites

- **Operating System:** Ubuntu 22.04 or later
- **ROS 2:** Humble Hawksbill (or a compatible ROS 2 distribution)
- **Gazebo Classic:** (e.g., Gazebo 11) – if using Gazebo Classic simulation
- **TurtleBot4 Packages:**  
  - `ros-humble-turtlebot4-description`  
  - (Optionally) `ros-humble-turtlebot4-simulator` if using provided simulation packages
- **Python Dependencies:**  
  - `opencv-python`
  - `numpy`
  - `ultralytics`  
  Install them using pip:
  ```bash
  pip install opencv-python numpy ultralytics
