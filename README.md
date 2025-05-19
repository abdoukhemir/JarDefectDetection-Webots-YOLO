# **Jar Defect Detection Simulation in Webots with YOLO (TFLite)**

## **Project Description**

This project simulates an industrial system for detecting and sorting defective jars on a production line. The simulation is carried out in Webots, using a YOLO model in TensorFlow Lite (TFLite) format for visual inspection.

## **Technologies Used**

* **Webots:** Simulation of the physical environment (conveyor, sensor, barrier, camera, robotic arm, jars).  
* **YOLO (TFLite):** Object detection model to identify intact vs. defective jars from the simulated camera images.

## **Modeled Industrial Problem**

Automation of visual inspection and sorting of jars for quality control, a process often manual and prone to errors.

## **Approach and Workflow**

The simulation models the following flow:

1. Jars move forward on a conveyor belt.  
2. A distance sensor detects the arrival of a jar at the inspection zone.  
3. A barrier lifts to allow the detected jar to pass, then closes again.  
4. A camera captures an image of the stationary jar.  
5. The YOLO TFLite model analyzes the image to detect defects.  
6. A robotic arm sorts the jar to the "intact" or "defective" position based on the detection result.  
7. The flow resumes for the next jar.

## **Configuration and Execution**

### **Prerequisites**

* Webots installed.  
* Python environment with necessary libraries for TFLite (tensorflow-lite or tflite\_runtime), OpenCV (opencv-python), and NumPy (numpy). A requirements.txt is recommended.  
* The YOLO model .tflite file and the class names file.

### **Project Structure**

The repository should include the Webots scene .wbt file, the controller code (often Python), the .tflite file and class names, as well as the 3D models used.

## **Defect Modeling**

Defects are simulated in Webots, potentially via different 3D models or textures applied to the jars.

## **YOLO Model Training and Conversion**

The YOLO model must be trained on jar data (intact and defective) and then converted to TFLite format. These steps are generally not included in this simulation repository.

## **Potential Extensions**

Improve detection, add defect types, optimize sorting, add statistics, etc.

## **Author**

Khemir Abderrahmen

