import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.util import invert

img_path = '1.png'
image = cv2.imread(img_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(rgb_image)
plt.title("Original Image")
plt.axis('off')
dimensions = image.shape
height = image.shape[0]
width = image.shape[1]
channels = image.shape[2]
size = image.size
print('Image Dimensions:', dimensions)
print('Image Height:', height)
print('Image Width:', width)
print('Number of Channels:', channels)
print('Image Size:', size)
plt.subplot(2, 3, 2)
plt.hist(image.ravel(), 256, [0, 256])
plt.title("Histogram")

negative = 255 - image
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB))
plt.title("Negative Image")
plt.axis('off')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.subplot(2, 3, 4)
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

inverted_gray_image = invert(gray_image)
skeleton = skeletonize(inverted_gray_image / 255.0)
plt.subplot(2, 3, 5)
plt.imshow(skeleton, cmap='gray')
plt.title("Skeleton Image")
plt.axis('off')
plt.tight_layout()
plt.show()