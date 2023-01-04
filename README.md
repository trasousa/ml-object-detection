# Tensorflow Object Detection 
Object detection is a computer vision technique in which a software system can detect, locate, and trace the object from a given image or video. The special attribute of object detection algorithms is that they identify the class of objects (person, table, chair, etc.) and their location-specific coordinates in the given image. 

These object detection algorithms might be pre-trained or can be trained from scratch. In most use cases, we use pre-trained weights from pre-trained models and then fine-tune them as per our requirements and different use cases.

In the use case, we will be using Tensorflow, and Tensorflow Object Detection API.

- Tensorflow is an open-source library for numerical computation and large-scale machine learning that ease Google Brain TensorFlow, the process of acquiring data, training models, serving predictions, and refining future results. 

- The TensorFlow Object Detection API is an open-source framework built on top of TensorFlow that makes it easy to construct, train and deploy object detection models. There are already pre-trained models in their framework which are referred to as [Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). For more information about all the state of the art models using TensorFlow check [TensorFlow model Garden](https://github.com/tensorflow/models).

![example](img/readme_example.jpeg)

## Goal

The goal of this project is to introduce you to Image recognition/object detection
This is one of the main challenges that Deep Neural Networks tackle nowadays, and in this Learning path, you will have the opportunity to train and leverage your own custom object detection model using the Tensorflow Object Detection API.

## Data

The Data used in this use case will be retrieved by you using your computer camera. For that, we will use OpenCv and 

- **OpenCV** is a cross-platform library using which we can develop real-time computer vision applications. It mainly focuses on image processing, video capture, and analysis including features like face detection and object detection. 

- **LabelImg** is a graphical image annotation tool. It is written in Python and uses Qt for its graphical interface.
Annotations are saved as XML files in PASCAL VOC format, the format used by ImageNet. Besides, it also supports YOLO and CreateML formats.
