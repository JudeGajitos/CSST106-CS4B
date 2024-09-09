# **Machine Problem No. 2: Applying Image Processing Techniques**

# **Objective:**

Understand and apply various image processing techniques, including image transformations and filtering, using tools like OpenCV. Gain hands-on experience in implementing these techniques and solving common image processing tasks.

# **Hands-On Exploration:**
*   **Lab Session 1: Image Transformations**
  *   Scaling and Rotation: Learn how to apply scaling and rotation transformations to images using OpenCV.

  *   Implementation:
    
  **1. Install the OpenCV**
```py
!pip install opencv-python-headless
```
  **2. Import Libraries and Create a display_image function**
```py
import cv2
import numpy as np
from matplotlib import pyplot as plt

def display_image(img, title = "Image"):
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.title(title)
  plt.axis("off")
  plt.show()
```
  **3. Upload and Load Image**
```py
from google.colab import files
from io import BytesIO
from PIL import Image

uploaded = files.upload()
image_path = next(iter(uploaded))
image = Image.open(BytesIO(uploaded[image_path]))
image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

display_image(image, "Original Image")
```
![Original Image](https://github.com/user-attachments/assets/3ec49f93-73ff-4161-ac62-401fdcb2d7ef)

  **4. Scaling and Rotation**
```py
def scale_image(image, scale_factor):
  height, width = image.shape[:2]
  scaled_img = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)), interpolation = cv2.INTER_LINEAR)
  return scaled_img

def rotate_image(image, angle):
  height, width = image.shape[:2]
  center = (width//2, height//2)
  matrix = cv2.getRotationMatrix2D(center, angle, 1)
  rotated_image = cv2.warpAffine(image, matrix, (width, height))
  return rotated_image

scaled_image = scale_image(image, 0.5)
display_image(scaled_image, "Scaled Image")

rotated_image = rotate_image(image, 45)
display_image(rotated_image, "Rotated Image")
```
![Scaled](https://github.com/user-attachments/assets/d7802440-ddf8-4da2-9a35-a8ede21fbf35)
![Rotated](https://github.com/user-attachments/assets/813f8505-09c2-47d8-9aa8-fbffdf1cfbea)

*   **Lab Session 2: Filtering Techniques**
  *   Blurring and Edge Detection: Explore how to apply blurring filters and edge detection algorithms to images using OpenCV.
  
  *   Implementation:
  **1. Blurring using Gaussian Blur, Median Blur, and Bilateral Blur.**
```py
#from the same uploaded and loaded image on the previous Lab Session.
gaussian_blur = cv2.GaussianBlur(image, (11, 11), 0)
display_image(gaussian_blur, "Gaussian Blur")

median_blur = cv2.medianBlur(image, 7)
display_image(median_blur, "Median Blur")

bilateral_blur = cv2.bilateralFilter(image, 111, 65, 65)
display_image(bilateral_blur, "Bilateral Blur")
```
![Gaussian](https://github.com/user-attachments/assets/cc7bcfad-a337-44e1-bd0f-97307134bb29)
![Median](https://github.com/user-attachments/assets/bd3bad65-1638-42ff-8089-feb2371d49d9)
![Bilateral](https://github.com/user-attachments/assets/8e10ae07-b7f7-4083-880a-c9fc5d853b9f)

  **2. Edge Detection using Canny**
```py
edge_detection = cv2.Canny(image, 80, 130)
display_image(edge_detection, "Edge Detection")
```
![Edge Detection](https://github.com/user-attachments/assets/9b0d8e83-a9cf-4be4-b376-a362e97f1801)
  
# **Problem-Solving Session:**

**Common Image Processing Tasks:** Engage in a problem-solving session focused on common challenges encountered in image processing tasks.
  *   Challenge: Image Noise
  *   Problem: Random fluctuations in pixel intensities, often caused by factors like low light, sensor imperfections, or transmission errors.
  *   Solution: Filtering: Apply filters like Gaussian blur, median filter, or bilateral filter to reduce noise while preserving edges.

**Scenario-Based Problems:** 
  *   Scenario: I have image that is taken in low quality, blurry, and difficult to see. Now, I want to enhance the quality of the image and improve its clarity and details.
  *   Solution: Apply a Gaussian blur or bilateral filter to reduce noise caused by low light conditions. These filters can help smooth out random pixel variations while preserving edges.

# **Implementing Image Transformations and Filtering:**
**Google Colab Link:**
https://colab.research.google.com/drive/1Y2Rbhrnyo9hfzTpA3wZ_GrWrjViS0xrt?usp=sharing

**Documentation Link:**
https://docs.google.com/document/d/1s49Qewt3QDPQHFJZK-DITZN8VV_oRPu4s6Xx1Qh6vAY/edit?usp=sharing

