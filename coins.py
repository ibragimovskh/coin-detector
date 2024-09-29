import cv2 
import numpy as np 
import matplotlib.pyplot as plt

# image = input("Input an image: ")


def binarize(img):
    # Apply Gaussian blur to reduce noise before binarization
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Otsu's adaptive thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return thresh

def morphling(binary):
    kernel = np.ones((11,11), np.uint8)
    
    # Just closing
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Closing followed by opening
    dilation = cv2.dilate(closing, np.ones((3,3), np.uint8), iterations=1)
    
    return dilation



# Apply morphological operations with different kernel sizes

def distransform(img): 
    # Distance transform
    dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    dist_max = dist_transform.max()
    dist_mean = dist_transform.mean() 

    dist_thresh = (dist_max + dist_mean) / 2
    return dist_transform, dist_thresh


# cv2.findContours() 
def detect_coins(img): 
    # Binarization
    thresh = binarize(img)
    # Morphological operations
    morphed = morphling(thresh)

    dist_transform, dist_thresh = distransform(morphed)
    
    # Thresholding to get sure foreground area
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 1.0, 0)
    sure_fg = np.uint8(sure_fg)

    # expand the background to have clear separation between bg and fg
    sure_bg = cv2.dilate(morphed, None, iterations=3)

    # Finding unknown region (doesn't belong neither to fg or bg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    num_labels, labels = cv2.connectedComponents(sure_fg)




    # shifting by 1 is needed for watershed algorithm, since it uses 0 for unknown regions 
    markers = labels + 1
    markers[unknown == 255] = 0 

    
    # img must be 3-channel for watershed, markers - grayscale
    markers = cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), markers)
    
    # binary mask for coins to be used in findContours
    coin_mask = np.zeros(img.shape, dtype=np.uint8)
    coin_mask[markers > 1] = 255
    
    # Find contours of the coins
    contours, _ = cv2.findContours(coin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    return thresh, morphed, dist_transform, sure_fg, markers, result, labels, contours


def filter_coins(contours, reference_area=None):
    areas = [cv2.contourArea(cnt) for cnt in contours]
    print(areas)

    
# Load and process the image
img = cv2.imread("./samples/26.png", cv2.IMREAD_GRAYSCALE)
thresh, morphed, dist_transform, sure_fg, markers, result, labels, contours = detect_coins(img)

filter_coins(contours)



plt.figure(figsize=(20, 20))

plt.subplot(331)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(332)
plt.imshow(thresh, cmap='gray')
plt.title('Binarized')
plt.axis('off')

plt.subplot(333)
plt.imshow(morphed, cmap='gray')
plt.title('Morphological Operations')
plt.axis('off')

plt.subplot(334)
plt.imshow(dist_transform, cmap='jet')
plt.title('Distance Transform')
plt.axis('off')

plt.subplot(335)
plt.imshow(sure_fg, cmap='gray')
plt.title('Sure Foreground')
plt.axis('off')

plt.subplot(336)
plt.imshow(markers, cmap='nipy_spectral')
plt.title('Watershed Markers')
plt.axis('off')

plt.subplot(337)
plt.imshow(labels, cmap='nipy_spectral')
plt.title(f'Connected Components (Count: {labels.max()})')
plt.axis('off')

plt.subplot(338)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Final Result')
plt.axis('off')

# Leave the last subplot (339) empty or use it for additional information if needed

plt.tight_layout()
plt.show()

print(f"Number of connected components: {labels.max()}")
