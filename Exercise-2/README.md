# Task 1: SIFT Feature Extraction
**Approach:**


*   The notebook begins by loading and converting an image of "eminem.jpg" to grayscale.
*   SIFT (Scale-Invariant Feature Transform) is used to detect keypoints and compute descriptors for these keypoints, highlighting key features of the image.

**Observations:**
*   Keypoints represent significant patterns or edges in the image, providing unique identifiers for different parts.

**Results:**

*   Keypoints and descriptors are visualized, showing the detected SIFT features superimposed on the grayscale image.

# Task 2: SURF Feature Extraction
**Approach:**
*   The image "eminem.jpg" is loaded and converted to grayscale.
*   SURF (Speeded-Up Robust Features) is used to detect and compute descriptors of keypoints, similar to SIFT but optimized for speed.
*   Detected keypoints are visualized by overlaying them on the original image.

**Observations:**
*   SURF is computationally efficient and often faster than SIFT, making it suitable for real-time applications.
*   Keypoints detected by SURF focus on areas of the image with high contrast and edges, providing robust features for matching.

**Results:**
*   The displayed image shows keypoints detected by SURF, illustrating the prominent features in "eminem.jpg" that SURF successfully identified.

# Task 3: ORB Feature Extraction
**Approach:**
*   The image "eminem2.jpg" is loaded and converted to grayscale.
*   ORB (Oriented FAST and Rotated BRIEF) is applied to detect and compute descriptors, offering a lightweight, free alternative to SIFT and SURF.
*   Keypoints are visualized by overlaying them on the original image.

**Observations:**
*   ORB is highly efficient and suited for applications needing fast computation, though it might be less precise than SIFT and SURF.
*   The detected keypoints focus on key areas, though ORB’s performance can vary with image scale and rotation.

**Results:**
*   The displayed image shows the keypoints detected by ORB, highlighting key features in "eminem2.jpg" that are useful for matching and analysis.

# Task 4. Feature Matching
**Approach:**

*   The code reads two images (eminem.jpg, and eminem2.jpg), resizes them to match dimensions, and applies the SIFT (Scale-Invariant Feature Transform) algorithm to detect keypoints and compute descriptors.
*   A brute-force matcher is used to find and sort the best feature matches between the two images.
*   The top 10 matches are visualized, with matched keypoints overlaid on the images for comparison.

**Observations:**
*   SIFT is effective in identifying distinctive keypoints, offering robust feature matching even with variations in image details.
*   Sorting by match distance prioritizes more accurate matches, though some mismatches may still occur.

**Results:**
*   The resulting image highlights the top feature matches between the two images, demonstrating SIFT’s capability in detecting consistent features across images.

# Task 5: Applications of Feature Matching
**Approach:**
*   Two images ("building.jpg" and "building1.jpg") are loaded and converted to grayscale.
*   SIFT is applied to detect keypoints and compute descriptors in both images.
*   The BFMatcher (Brute-Force Matcher) with K-Nearest Neighbors is used to find good matches between the images based on descriptor similarity.
*   A homography transformation aligns the first image to the second, allowing for perspective alignment using RANSAC (Random Sample Consensus).

**Observations:**
*   Homography effectively matches keypoints across the two images despite minor differences.
*   Proper alignment is achieved by filtering good matches with RANSAC, reducing the influence of outliers.

**Results:**
*   The resulting image shows the alignment of the two buildings, demonstrating successful image matching and alignment.

# Task 6: Combining Feature Extraction Methods
**Approach:**
*   Two images ("eminem.jpg" and "eminem2.jpg") are loaded in grayscale, with resizing applied if necessary to match dimensions.
*   Both SIFT and ORB (Oriented FAST and Rotated BRIEF) are used to extract keypoints and descriptors.
*   SIFT matches are obtained with BFMatcher (L2 norm), while ORB matches use BFMatcher (Hamming distance).
*   Matches from both SIFT and ORB are visualized side-by-side for comparison.

**Observations:**
*   SIFT and ORB show different matching patterns. SIFT is more accurate for detailed matching, while ORB is faster but might produce more false positives.

**Results:**
*   Side-by-side visualization of SIFT and ORB matches showcases the relative performance of these feature extraction methods, illustrating their strengths and weaknesses in feature matching.
