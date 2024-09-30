def preprocess_image(image_file):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applying Otsu's thresholding to create a binarized form
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Applying morphological operations to fill gaps inside the coins
    kernel = np.ones((15, 15), np.uint8)
    morph_img = cv2.dilate(binary, kernel, iterations=1)
    morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    morph_img = cv2.erode(morph_img, kernel, iterations=1)
    return img, gray, morph_img


def apply_watershed(img, morph_img):
    # Distance transformation to separate touching coins
    distance_transform = cv2.distanceTransform(morph_img, cv2.DIST_L2, 5)
    _, sure_foreground = cv2.threshold(distance_transform, 0.7 * distance_transform.max(), 255, 0)
    sure_foreground = np.uint8(sure_foreground)
    small_kernel = np.ones((5, 5), np.uint8)
    sure_foreground = cv2.erode(sure_foreground, small_kernel, iterations=1)
    sure_background = cv2.dilate(morph_img, small_kernel, iterations=2)

    # Subtracting the sure foreground from the background to get the unknown region
    unknown = cv2.subtract(sure_background, sure_foreground)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_foreground)
    # Adding 1 to all markers so that sure background is not labeled as zero
    markers = markers + 1
    # Marking the region of the unknown with zero
    markers[unknown == 255] = 0
    # Applying watershed
    markers = cv2.watershed(img, markers)
    return markers