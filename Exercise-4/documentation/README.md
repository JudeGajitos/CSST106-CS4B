# Exercise 1: HOG (Histogram of Oriented Gradients) Object Detection
**Approach:**

*   Loaded a grayscale image and applied the HOG descriptor to extract features.
*   Visualized the gradient orientations using matplotlib.

**Observations:**
*   HOG is used for object detection by highlighting the structure in the image through gradient orientation.
*   Provides a human-readable visualization of features like edges and texture.

**Results:**
*   Displayed the HOG representation of the image.

# Exercise 2: YOLOv3 Object Detection
**Approach:**

*   Downloaded the YOLOv3 pre-trained model and configuration.
*  Used OpenCV to preprocess the image, then passed it through the YOLO network for detection.
*   Drew bounding boxes for objects detected with confidence greater than 50%.

**Observations:**

*   YOLO provides real-time object detection by splitting the image into regions and predicting bounding boxes.
*   The model outputs multiple detections, and confidence filtering ensures accuracy.

**Results:**

*   Displayed the image with bounding boxes around detected objects.

# Exercise 3: SSD (Single Shot MultiBox Detector) with TensorFlow
**Approach:**

*   Downloaded the pre-trained SSD model and loaded an image using OpenCV.
*   Ran the TensorFlow object detection model and visualized bounding boxes on the image for objects with a confidence score greater than 50%.

**Observations:**

*   SSD, like YOLO, is used for real-time detection, but it uses a slightly different approach by predicting both the bounding boxes and classes directly.

Results:

*   Displayed the detected objects with bounding boxes and labels.

# Exercise 4: Traditional vs. Deep Learning Object Detection Comparison
**Approach:**

*   Compared HOG-SVM (traditional) and YOLOv5 (deep learning) object detection methods.
*   HOG-SVM: Extracted HOG features from positive (object) and negative (random background) samples, then trained an SVM.
*   YOLOv5: Trained using a dataset with images and labels, followed by evaluation using a pre-trained YOLOv5 model.

**Observations:**

*   HOG-SVM: This traditional method depends on manually engineered features like HOG and uses a linear SVM for classification. It is generally slower and less accurate than deep learning models.
*   YOLOv5: This method, based on deep learning, processes images much faster and provides higher accuracy due to end-to-end feature learning.


**Results:**
*   HOG-SVM:
Accuracy: 77.5%, Average inference time per image: 0.123 seconds

*   YOLOv5:
Inference time for two images: 1.4 seconds, Average inference time per image: 0.7 seconds

Overall, the comparison demonstrated the superiority of deep learning methods like YOLOv5 in terms of both speed and accuracy for object detection tasks. Traditional methods like HOG-SVM, while still useful, are typically slower and less efficient when compared to modern deep learning techniques.
