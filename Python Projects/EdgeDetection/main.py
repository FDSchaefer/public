import cv2
import matplotlib.pyplot as plt
import numpy as np
 
# Read the original image
img = cv2.imread(r'inputImages\20tex_alpha120_10.tif')

def resize4display(img):
    return(cv2.resize(img,
                    (int(img.shape[1]/3), int(img.shape[0]/3))
                    )
    )

# Display original image
#cv2.imshow('Original',resize4display(img) )
#cv2.waitKey(0)
 
#plt.hist(img.ravel(),256,[0,256]); plt.show()
#cv2.waitKey(0)

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (7,7), 0) 
ret,grounded = cv2.threshold(img_blur,60,255,cv2.THRESH_TOZERO)
cv2.imwrite("outputImages\Cleaned.png", grounded)

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

image_sharp = grounded.copy()
for i in range(0,0):
    image_sharp = cv2.filter2D(src=image_sharp, ddepth=-1, kernel=kernel)

#cv2.imshow('Grounded', resize4display(grounded))
#cv2.imshow('Sharp', resize4display(image_sharp))
#cv2.waitKey(0)

cv2.imwrite("outputImages\Sharpened.png", image_sharp)

# Sobel Edge Detection
sobelx = cv2.Sobel(src=image_sharp, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=image_sharp, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=image_sharp, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection


# Display Sobel Edge Detection Images
#cv2.imshow('Sobel X',  resize4display(sobelx))
#cv2.imshow('Sobel Y', resize4display(sobely))
#cv2.imshow('Sobel X Y', resize4display(sobelxy))
#cv2.waitKey(0)
cv2.imwrite("outputImages\Edges.png", sobely)

# Remove processing edges. 
sobelClean = (sobely <= (sobely.mean()+sobely.std()*2))*sobely
sobelClean = (sobely >= (sobely.mean()-sobely.std()*2))*sobelClean
cv2.imwrite("outputImages\CleanEdges.png", sobelClean)


from skimage.transform import probabilistic_hough_line
from matplotlib import cm
lines = probabilistic_hough_line(sobelClean, threshold=500, line_length=200,
                                 line_gap=10)


# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(sobelClean, cmap=cm.gray)
ax[1].set_title('Edges')

ax[2].imshow(img_blur * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_xlim((0, img_blur.shape[1]))
ax[2].set_ylim((img_blur.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()


cv2.destroyAllWindows()