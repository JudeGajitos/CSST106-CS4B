**Exercise 1: Harris Corner Detection**

Grayscale Conversion: The input image is converted to grayscale as Harris Corner Detection works on intensity values.

Corner Detection: The cv2.cornerHarris function detects corners in the grayscale image, highlighting areas where pixel intensity changes sharply.
Corner Marking: Detected corners are dilated for visibility and then marked in red on the original image.

Display: The result is displayed using matplotlib after converting the image from BGR to RGB.

**Exercise 2: HOG (Histogram of Oriented Gradients) Feature Extraction**

Grayscale Conversion: The image is converted to grayscale.

HOG Descriptor: The hog() function extracts Histogram of Oriented Gradients (HOG) features, which capture edge direction and intensity for object recognition.

Normalization: The HOG image is rescaled for better visualization.

Visualization: The original image and the HOG feature image are displayed side by side.

**Exercise 3: FAST Keypoint Detection**

Grayscale Conversion: Similar to the previous exercises, the image is converted to grayscale.

FAST Algorithm: The FAST (Features from Accelerated Segment Test) algorithm detects keypoints in the image.

Visualization: The detected keypoints are drawn on the original image and displayed with green circles to mark the keypoints.

**Exercise 4: ORB (Oriented FAST and Rotated BRIEF) Keypoints and FLANN Matching**

Grayscale Conversion: Two images are loaded and converted to grayscale for processing.

ORB Detector: The ORB algorithm is used to detect keypoints and compute descriptors for both images.

FLANN Matching: The FLANN (Fast Library for Approximate Nearest Neighbors) matcher finds and matches keypoints between the two images.

Visualization: The matched features are drawn and displayed, showing correspondences between the two images.

**Exercise 5: Image Segmentation using the Watershed Algorithm**

Grayscale Conversion and Thresholding: The image is converted to grayscale and a binary threshold is applied to create a binary image.

Morphological Operations: Morphological transformations clean the binary image to separate objects from the background.

Watershed Algorithm: The Watershed algorithm is applied to segment the image, identifying different regions and marking boundaries in red.

Visualization: The segmented image and the original image are displayed side by side.
