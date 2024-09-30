import cv2
import numpy as np
import matplotlib.pyplot as plt
image = input("")


def binarize(img, kernel=(3,3)):

    # Apply Gaussian blur to reduce noise before binarization
    blurred = cv2.GaussianBlur(img, kernel, 0)
    # Otsu's adaptive thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def morphling(binary):
    kernel = np.ones((9,9), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)
    return opening

def distransform(img): 
    dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 3)
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

    # markers = cv2.dilate(markers, np.ones((29,29), np.uint8), iterations=1)
    # img must be 3-channel for watershed, markers - grayscale
    markers = cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), markers)
    
    # binary mask for coins to be used in findContours
    coin_mask = np.zeros(img.shape, dtype=np.uint8)
    coin_mask[markers > 1] = 255
    coin_mask = cv2.dilate(coin_mask, np.ones((3,3), np.uint8), iterations=3)
    coin_mask = cv2.bitwise_and(morphed, coin_mask)
    # Find contours of the coins
    contours, _ = cv2.findContours(coin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    return thresh, morphed, dist_transform, sure_fg, markers, result, labels, contours, coin_mask

def categorize_coin(area):
    print(f"AREA {area}\n")
    if 44000 <= area:
        return 25   
    elif 40000 <= area < 44000:
        return 10 
    elif 30000 <= area < 40000:
        return 5  
    elif area < 30000:
        return 1  
    return 0 

def coin_stats(contours):

    coin_data = []
    for contour in contours:
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        if area < 10000:  
            continue
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            coin_value = categorize_coin(area)
            if coin_value > 0:
                coin_data.append({
                    'x': cx,
                    'y': cy,
                    'value': coin_value,
                    'area': area
                })
    return coin_data



# Load and process the image
img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
thresh, morphed, dist_transform, sure_fg, markers, result, labels, contours, coin_mask = detect_coins(img)

coin_data = coin_stats(contours)
print(len(coin_data))
for coin in coin_data:
    print(f" AREA {coin['area']} {coin['x']} {coin['y']} {coin['value']}")



    
def visualize_results(img, thresh, morphed, dist_transform, sure_fg, markers, labels, result, coin_mask, coin_data):
    """Visualize the processing steps and annotate detected coins."""
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
    
    # Annotate each coin with its type before displaying
    annotated_result = result.copy()
    for coin in coin_data:
        center = (int(coin['x']), int(coin['y']))
        cv2.circle(annotated_result, center, 10, (255, 0, 0), -1)  # Blue circles
        label = str(coin['value']) if coin['value'] != 0 else 'Unknown'
        cv2.putText(annotated_result, label, (center[0] + 10, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    plt.subplot(338)
    plt.imshow(cv2.cvtColor(annotated_result, cv2.COLOR_BGR2RGB))
    plt.title('Final Result with Classified Coins')
    plt.axis('off')
    
    # Display the coin mask correctly using grayscale colormap
    plt.subplot(339)
    plt.imshow(coin_mask, cmap='gray')
    plt.title('Coin Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_results(img, thresh, morphed, dist_transform, sure_fg, markers, labels, result, coin_mask, coin_data)