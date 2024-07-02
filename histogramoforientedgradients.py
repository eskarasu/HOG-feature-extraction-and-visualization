from skimage import feature, io
import matplotlib.pyplot as plt

# Görüntüyü okuyalım
image = io.imread('image.png', as_gray=True)

# HOG özelliklerini çıkaralım
hog_features, hog_image = feature.hog(image, orientations=8, pixels_per_cell=(16, 16),
                                      cells_per_block=(1, 1), visualize=True)

# Orijinal görüntüyü ve HOG görüntüsünü çizdirelim
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Orijinal Görüntü')

ax2.axis('off')
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')

plt.show()
