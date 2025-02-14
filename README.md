# Object Detection and Tracking in ROS 2 with TurtleBot4

This project demonstrates an object detection and tracking pipeline in ROS 2 using YOLOv8 and OpenCV. The system processes a live camera feed from a simulated TurtleBot4 running in Gazebo (or Ignition), detects objects (e.g., persons, bottles, etc.), tracks them over time using a simple centroid tracker, and publishes detection and tracking information as ROS 2 topics.


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


## Running the Simulation
   ```bash
   ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py
   ros2 run object_detection detect


**Use rqt_image_view to see the annotated image:**
 ```bash
ros2 run rqt_image_view rqt_image_view

