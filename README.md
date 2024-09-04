# CSST106-CS4B

https://github.com/user-attachments/assets/f3612311-4681-46a7-a2d4-d202b969e373

# **Introduction to Computer Vision**

The ability of computers to comprehend and evaluate visual information similarly to humans is known as computer vision. This covers activities like reading text, identifying faces and objects, and figuring out the context of a picture or video. 

Artificial intelligence (AI) and computer vision are closely related fields that frequently employ AI methods like machine learning to evaluate and comprehend visual data. A computer can be "trained" to identify patterns and features in visual data, such as edges, forms, and colors, by using machine learning techniques. After being educated, the computer can recognize and categorize items in fresh pictures and videos. As these classifiers are trained more and exposed to more data, their accuracy can be increased over time.

# **Role of Image Processing in Artificial Intelligence (AI)**

The term "image processing" describes techniques for enhancing, analyzing, and extracting information from digital images. Image segmentation, feature extraction, geometric modifications, image enhancement, image compression, and picture reconstruction are the primary tasks in image processing. With the help of these methods, we may enhance the quality of photos, eliminate noise and artifacts, extract useful data and metadata, and identify patterns and objects. 

Scientific image analysis, medical imaging systems, industrial inspection systems, surveillance and security systems, photo editing software, and image to text converter are among the devices that use image processing techniques. Machine learning and deep learning models are supplementing traditional image processing approaches with the advent of artificial intelligence.

*   Image classification – Using machine learning and deep learning models to classify images or labels. This enables applications like automated sorting, recognition, and scene understanding.
*   Object detection – Locating and identifying objects within images. Requires training models on labeled image datasets with object bounding boxes. Used for applications like product detection, face recognition, and content filtering.
*   Semantic segmentation – Assigning a semantic label (like road, building, person, etc) to every pixel in an image. Requires powerful convolutional neural networks. Used for autonomous driving, medical imaging, and drone analysis.
*   Image enhancement – Applying techniques like super-resolution, noise reduction, contrast adjustment, and color correction to images using trained AI models. This improves image quality for downstream tasks.
*   Image generation – Generating new realistic images using generative adversarial networks (GANs) and autoencoders. Used to augment training datasets or for creative applications.
*   Anomaly detection – Identifying abnormal or outlier images that differ from a “normal” class using unsupervised learning. Used for defect detection, medical diagnosis, and security.
*   Visual recommendation – Providing personalized image and video recommendations based on a user’s viewed content using convolutional neural nets and reinforcement learning.
*   Content moderation – Detecting inappropriate or offensive visual content like nudity, violence, hate speech, etc. Requires training on labeled datasets of multiple classes.

# **Types of Image Processing Techniques**

*   Filtering - The pixels in an image are directly subjected to the filtering technique. Generally speaking, a mask is added in size to have a certain center pixel. The mask is positioned on the image so that its center crosses every pixel in the image.
<img width="599" alt="filtering" src="https://github.com/user-attachments/assets/b7e024cd-cbb9-45c0-b1ef-bc39ffe7528a">


*   Segmentation - In order assist with object detection and related tasks, image segmentation is a computer vision approach that divides a digital image into distinct groupings of pixels, or image segments.
![segmentaion](https://github.com/user-attachments/assets/62ad5dd0-481e-492b-ad52-3f3aede9b8c2)


*   Edge Detection - An essential method for locating and recognizing the borders or edges of objects in an image is edge detection. It is employed to extract the outlines of objects that are present in an image as well as to recognize and detect discontinuities in the image intensity.
![edgeDetection](https://github.com/user-attachments/assets/5f70c118-86c1-4e58-9eca-fb8a66098786)


# **Case Study Selection: BIOMETRICS**
Image processing is needed in biometrics to identify a person whose biometric image has already been stored in the database. Biometrics based on images, like fingerprints, iris scans, and facial features, require the use of image processing and pattern recognition techniques. An image-based biometric system requires an extremely clear and pure sample image of the user's biometric in order to function correctly. 

**Key Techniques Used In Biometrics**


*   Image Restoration - Lowering the noise that was added to the image during sample acquisition. And erasing the distortions that surfaced during biometric enrollment. 
*   Image Enhancement - Techniques for enhancing images make any area or feature more visible while hiding information in other areas. Only after the restoration is finished is it done. In order to make the image suitable for additional processing, it involves brightness, sharpness, contrast adjustment, and other adjustments. 
*   Feature Extraction:
  *   General features − The features such as shape, texture, color, etc., which are used to describe content of the image.
  *   Domain-specific features − They are application dependent features such as face, iris, fingerprint, etc. Gabor filters are used to extract features.

# **Implementation Creation**

**Edge Detection**

Edge detection is essential to biometric systems, especially in fingerprint and face recognition. It aids in identifying the unique features that make each person unique. The image is processed by edge detection techniques to determine the boundaries between various regions. These boundaries often line up with significant features, such as the ridges and valleys of fingerprints or the edges of facial features. 

**Edge Detection using OpenCV's Canny edge detection algorithm**

**1. Importing Libraries**

The cv2 library (OpenCV) provides functions for image processing and computer vision tasks. The matplotlib.pyplot (imported as plt) library is used for creating visualizations like plots. Amd google.colab.patches (specifically cv2_imshow) library patch is used for displaying images in Google Colab (optional, not used for core functionality).
```py
import cv2
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow
```

**2. Loading the Image**

Load the image named "face.jpg" in img variable.
```py
# Load the image
img = cv2.imread("face.jpg")
```
**3. Converting to Grayscale**

Converts the loaded image stored in img varible to grayscale and stores the output in gray variable.
```py
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

**4. Applying Gaussian Blur**

Applies a Gaussian blur filter to the grayscale image stored in gray variable and stored the output in blurred variable.
```py
# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
```

**5. Canny Edge Detection**

Performs the Edge Detection using the Canny edge detection algorithm.
```py
# Detect edges using Canny edge detection
edges = cv2.Canny(blurred, 100, 200)
```

**6. Displaying Images**

Display the Original and the Edge Detected Image usuing plt.
```py
# Display the original image and the edge-detected image
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Detected Image'), plt.xticks([]), plt.yticks([])
plt.show()
```
![EdgeDetection](https://github.com/user-attachments/assets/c6176f35-c5b9-4d79-bf81-228dde635898)


# **Diagram/Flowchart**
![EdgeDetectionDiagram drawio](https://github.com/user-attachments/assets/a808377f-505e-4114-aa37-7a7f8f6c2a71)

# **Conclusion**

Artificial intelligence (AI) requires image processing, particularly in areas like computer vision, robotics, and medical imaging. In order to extract relevant information, digital images must be manipulated and analyzed. AI systems can better comprehend and interact with the visual environment by creating effective image processing algorithms, which will promote various kinds of industries and applications.

In this activity I have learned different things about computer vision especially in image processing. The image processing techniques that helps the image to enhance its quality, extract its important features, and analyze it. 

# **Emerging Form of Image Processing**

**Edge Computing**

Edge computing is a network model where data processing occurs at the network's edge, near the data source, eliminating the need to send image data to a central server or cloud service. For instance, a low-power single board computer near the camera can be used for license plate recognition in road toll collection, broadcasting only the license plate numbers. Applications can be delivered to embedded systems and cross-platform development is made simple with FLIR's Spinnaker SDK, which supports X64 and ARM hardware. 

**Key Benefits of Edge Computing**

By reducing the number of switches and hosts between the data's source and destination, processing images on the edge reduces jitter and latency inside the system. The time interval between acquiring an image and taking action grows with each network node that data packets pass through. By reducing the delay brought on by the time it takes to upload image data, edge computing significantly reduces system latency. Edge nodes have the ability to anonymize data that is transferred to the cloud for additional analysis, hence improving system security and reducing privacy issues. 
