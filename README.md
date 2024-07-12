# HOG-feature-extraction-and-visualization

This repository demonstrates the extraction of Histogram of Oriented Gradients (HOG) features from an image using Python. The code reads an image, extracts the HOG features, and visualizes both the original image and the HOG image side by side.

## Requirements

To run the code, you need to install the following Python packages:

- `scikit-image`
- `matplotlib`

You can install these packages using pip:

```bash
pip install scikit-image matplotlib
```
## Usage

1. Place the image you want to process in the same directory as the script and name it image.png.

2. Run the script:
```bash
python hog_feature_extraction.py
```

The script will read the image, extract the HOG features, and display the original image and the HOG image side by side.

## Code Explanation

```python

from skimage import feature, io
import matplotlib.pyplot as plt

# Read the image
image = io.imread('image.png', as_gray=True)

# Extraction of HOG feature
hog_features, hog_image = feature.hog(image, orientations=8, pixels_per_cell=(16, 16),
                                      cells_per_block=(1, 1), visualize=True)

# Plot the original image and the HOG image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Original Image')

ax2.axis('off')
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')

plt.show()
```
Image Reading: The image is read in grayscale using io.imread.

HOG Feature Extraction: HOG features are extracted using feature.hog with the following parameters:
- orientations=8: Number of orientation bins.
- pixels_per_cell=(16, 16): Size of the cells.
- cells_per_block=(1, 1): Number of cells per block.
- visualize=True: Returns the HOG image.

Visualization: The original and HOG images are displayed side by side using matplotlib.

License
-------
This project is licensed under the MIT License. See the LICENSE file for details.

Contributing
------------
Feel free to open issues or submit pull requests if you have any suggestions or improvements.

Acknowledgements
----------------
- This project uses the scikit-image (https://scikit-image.org/) library for image processing.
- Visualization is done using matplotlib (https://matplotlib.org/).

Contact
-------
For any questions or comments, please open an issue on this repository.

