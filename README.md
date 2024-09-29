### Steps

1. Thresholding (adaptive thresholding using Otsu), coin pixels are white, bg pixels are black
2. Morphing - process shapes in the image. Erosion - shrink white areas (coins), dilation - expand white areas. Opening - erosion followed by dilation, closing - vice versa.
3. Distance transform - calculates the distance from every foreground pixels to the nearest background pixel. The pixels in the centre of the coin have heavier""weight"".
4. gaussian blur -> thresholding -> morphology -> distance transform -> connected components -> watershed -> contours

### Definitions

kernel - convolution matrix or mask, basically a small matrix that gets applied on the sections of the image to apply various effects (blurring, sharpening, etc.)

Opening (Erosion followed by Dilation):

Removes small objects and noise while preserving the shape and size of larger objects.

Closing (Dilation followed by Erosion):

Fills small holes and connects nearby objects while preserving overall shapes.

Morphological operations - use closing, that works the best.

Basically, the way distance transform works is you get the binary image for the input, then you calculate the distance from each non-bg pixel(white) to the nearest bg pixel(black). And the further the distance is, the closer your number is to 1(if normalized), and the edges will have really low, close to 0 number.
