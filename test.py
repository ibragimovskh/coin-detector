import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

def binarize(img):
    # Apply Gaussian blur to reduce noise before binarization
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Otsu's adaptive thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return thresh

def morphling(binary):
    kernel = np.ones((5,5), np.uint8)  # Reduced kernel size
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilation = cv2.dilate(closing, kernel, iterations=1)
    return dilation

def distransform(img): 
    # Distance transform
    dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    dist_max = dist_transform.max()
    dist_mean = dist_transform.mean() 

    dist_thresh = (dist_max + dist_mean) / 2
    return dist_transform, dist_thresh

def detect_coins(img): 
    # Binarization
    thresh = binarize(img)
    # Morphological operations
    morphed = morphling(thresh)

    dist_transform, dist_thresh = distransform(morphed)
    
    # Thresholding to get sure foreground area
    _, sure_fg = cv2.threshold(dist_transform, dist_thresh, 1, 0)  # 0 and 1
    sure_fg = np.uint8(sure_fg)

    # Scale sure_fg to 0 and 255
    sure_fg_scaled = sure_fg * 255

    # Expand the background to have clear separation between bg and fg
    sure_bg = cv2.dilate(morphed, None, iterations=3)  # 0 and 255

    # Finding unknown region (doesn't belong neither to fg or bg)
    unknown = cv2.subtract(sure_bg, sure_fg_scaled)

    # Connected components for sure foreground
    num_labels, labels = cv2.connectedComponents(sure_fg)

    # Shift labels by 1 to ensure background is not 0
    markers = labels + 1
    markers[unknown > 0] = 0  # Mark unknown regions as 0

    # img must be 3-channel for watershed
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Apply watershed
    markers = cv2.watershed(img_color, markers)
    
    # Create a binary mask for coins to be used in findContours
    coin_mask = np.zeros(img.shape, dtype=np.uint8)
    coin_mask[markers > 1] = 255  # Coins are labeled >1
    
    # Find contours of the coins
    contours, _ = cv2.findContours(coin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image for visualization
    result = img_color.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    return thresh, morphed, dist_transform, sure_fg, markers, result, labels, contours

def categorize_coins(contours):
    if not contours:
        return 0, []
    
    areas = [cv2.contourArea(cnt) for cnt in contours]
    areas_sorted = sorted(areas)
    
    # Define percentiles for dynamic thresholding
    p25 = np.percentile(areas_sorted, 25)
    p50 = np.percentile(areas_sorted, 50)
    p75 = np.percentile(areas_sorted, 75)
    
    coin_types = []
    for area in areas:
        if area < p25:
            coin_types.append('1 cent')
        elif p25 <= area < p50:
            coin_types.append('5 cents')
        elif p50 <= area < p75:
            coin_types.append('10 cents')
        else:
            coin_types.append('25 cents')
    
    return len(coin_types), list(zip(coin_types, areas))

def find_centers(contours):
    centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
        else:
            centers.append((0, 0))
    return centers

def annotate_image(result, contours, coin_info, centers):
    for i, cnt in enumerate(contours):
        cX, cY = centers[i]
        coin_type = coin_info[i][0] if i < len(coin_info) else "Unknown"
        cv2.circle(result, (cX, cY), 5, (255, 0, 0), -1)  # Blue dot at center
        cv2.putText(result, coin_type, (cX - 40, cY - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red text
    return result

def plot_results(img, thresh, morphed, dist_transform, sure_fg, markers, labels, result):
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
    plt.title('Final Result with Annotations')
    plt.axis('off')
    
    # Leave the last subplot (339) empty or use it for additional information if needed
    
    plt.tight_layout()
    plt.show()

def main():
    # Load and process the image
    img_path = "./samples/6.png"  # Update with your image path
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}. Please check the path.")
    
    thresh, morphed, dist_transform, sure_fg, markers, result, labels, contours = detect_coins(img)
    
    # Categorize coins
    num_coins, coin_info = categorize_coins(contours)
    
    # Find centers
    centers = find_centers(contours)
    
    # Annotate the result image with centers and labels
    result = annotate_image(result, contours, coin_info, centers)
    
    # Plotting the results
    plot_results(img, thresh, morphed, dist_transform, sure_fg, markers, labels, result)
    
    # Output the results
    if contours:
        print(f"Number of coins detected: {num_coins}")
        for idx, (coin, area) in enumerate(coin_info):
            print(f"Coin {idx+1}: {coin}, Area: {area:.2f}, Center: {centers[idx]}")
    else:
        print(0)

if __name__ == "__main__":
    main()
