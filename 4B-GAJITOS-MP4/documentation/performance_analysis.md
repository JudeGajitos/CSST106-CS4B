# Harris Corner Detection:

*   Function: Detects corners in an image, marking them in red.
*   Performance: Efficient for basic corner detection, but may struggle with complex textures or high-noise images.

# HOG (Histogram of Oriented Gradients) Feature Extraction:

*   Function: Extracts edge and gradient features from an image, commonly used for object detection.
*   Performance: Requires grayscale conversion and gradient computation, which is efficient but may require high memory for large images.

# ORB (Oriented FAST and Rotated BRIEF) Feature Matching:

*   Function: Detects and matches keypoints between two images using ORB and FLANN-based matching.
*   Performance: ORB is fast and suitable for real-time applications but may have reduced accuracy compared to SIFT/SURF, especially in scenes with affine transformations.

# SIFT and SURF Feature Extraction:

*   Function: Detects and matches features using SIFT and SURF.
*   Performance: High accuracy, especially with transformations and scale variations, but computationally intensive, especially without GPU support.

# Brute-Force Matcher with ORB:

*   Function: Matches features between two images using brute-force search on ORB descriptors.
*   Performance: Slower than FLANN-based matching but provides reliable matches with simple distance calculations.

# Watershed Segmentation:

*   Function: Segments an image based on the watershed algorithm, marking object boundaries.
*   Performance: Computationally heavy, especially for large images. Works well for images with distinct foreground and background.
