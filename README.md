# CV_vehicle_counter :car:

## Introduction
In an era where urbanization and population growth are accelerating, efficient transportation systems have
become crucial for the sustainability and livability of cities. As the number of vehicles on the roads continues
to rise, managing traffic congestion, improving road safety, and planning future infrastructure developments
are pressing challenges. Accurate and real-time data on vehicle movement is essential to address these issues
effectively. Our Car Counting Project aims to provide a robust solution for real-time vehicle detection and
counting, offering critical insights for transportation planning and traffic management

# Vehicle Detection using Computer Vision and Machine Learning

## Methods

Our dataset contains images of highways with labels of motorcycles, Sedans, SUVs, trucks, vans, and pickups. In our vehicle detection algorithms, we ignored all motorcycles and summed up the instances of Sedans, SUVs, vans, and pickups to be the number of cars.

### Computer Vision Techniques

We initially attempted to implement vehicle detection using techniques purely in computer vision. Many of these algorithms have limitations and require specific conditions to work correctly. Among them, we tried edge detection and background subtraction algorithms.

#### Edge Detection

- **Process:** Gray → Gaussian blur (remove noises) → Get edges (Canny Edge) → Find contour
- **Description:** We used the Canny Edge detection algorithm to identify the boundaries of an image. After detecting edges, we used a contour detector to find continuous edges that form shapes, which might be vehicles. Vehicle detection using edge detection performed poorly because of the complexity of vehicle shapes, ambiguity in edge formation, and inadequate contour formation. The edge detection algorithm struggles to accurately delineate vehicle boundaries, which leads to inaccurate contours.

![Canny Edge](report1/edge_detection.png) <!-- Ensure the image path is correct or provide the actual path -->

#### Background Subtraction

- **Process:** Gray → Gaussian blur (remove noises) → Threshold to remove shadows → Apply some morphological operations to ensure a good mask (getStructuringElement, erode, dilate) → Find contour
- **Description:** Background subtraction can be limited in its effectiveness for vehicle detection in images. One major challenge arises from the need for a consistent and accurate background model. Background subtraction techniques rely on the assumption that the scene’s background remains relatively static over time. However, in real-world scenarios, backgrounds can vary due to changes in lighting conditions, moving objects, or environmental factors. Analyzing a single image lacks the temporal context necessary for accurate background modeling. Without prior knowledge of the background or a reference image captured from the same angle without any vehicles present, background subtraction may yield inaccurate results.

![Background Subtraction](report1/background_detection.png) <!-- Ensure the image path is correct or provide the actual path -->

#### Summary

Background subtraction and edge detection are fundamental techniques used for vehicle detection in images, each with its own set of challenges and limitations. Vehicle detection using edge detection performs poorly due to the complexity of vehicle shapes, ambiguity in edge formation, and inadequate contour formation. Similarly, the background subtraction technique is significantly limited when applied to single images due to the need for a consistent and accurate background model. Without an updated background model or a reference image without vehicles, background subtraction often yields inaccurate results. In contrast, `cv2.createBackgroundSubtractorMOG()`, a Gaussian Mixture-based Background/Foreground Segmentation Algorithm in CV2, performs well with video data by continuously updating the background model.

### Detection using Machine Learning

To find a better vehicle detector, we tried machine learning-based methods like You-Only-Look-Once (YOLO). YOLO is a popular object detection algorithm known for its speed and accuracy. The architecture of YOLO involves dividing the input image into a grid and predicting bounding boxes and class probabilities for each grid cell simultaneously. This allows YOLO to detect multiple objects in an image in real-time.

#### YOLOv3

Initially, we chose to use YOLOv3. The YOLOv3 object detection model runs a deep learning convolutional neural network (CNN) on an input image to produce network predictions from multiple feature maps. Then, the object detector gathers and decodes predictions to generate the bounding boxes. From these bounding boxes, YOLOv3 uses anchor boxes, where the model predicts these three attributes for each anchor box:
- **Intersection over union (IoU):** Predicts the objectness score of each anchor box.
- **Anchor box offsets:** Refine the anchor box position.
- **Class probability:** Predicts the class label assigned to each anchor box.

We compared the performance of YOLOv3 to other YOLO models, such as YOLOv2 and YOLOv4. From version two to four, improvements to object detection were made in both speed and accuracy through various machine learning techniques and changes in architecture. Since these are pretrained models, we combined all the images into one directory and tested on that.

### Experimental Results

| Ground Truth or YOLO model | Detection Label Instances | Accuracy | Precision, Recall, and F1-Score |
|----------------------------|---------------------------|----------|--------------------------------|
| **Ground Truth**           | 6 cars, 0 buses, 2 trucks | N/A      | N/A                            |
| **YOLO v2**                | 1 car, 0 buses, 0 trucks  | 0% for car, 100% for buses, 0% for trucks | Precision = <br>Recall = <br>F1-Score = |
| **YOLO v3**                | 10 cars, 0 buses, 3 trucks | 0% for car, 100% for buses, 0% for trucks | Precision = <br>Recall = <br>F1-Score = |
| **YOLO v4**                | 12 cars, 0 buses, 3 trucks | 0% for car, 100% for buses, 0% for trucks | Precision = <br>Recall = <br>F1-Score = |

**Table 1: Comparison of YOLO Models**

---

Ensure to replace the image paths with the actual paths of your images before using this README.

