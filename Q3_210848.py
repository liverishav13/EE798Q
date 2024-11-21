import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform as tf
# import math
# from statistics import mode

# Read image
image = cv2.imread('3_c.jpg')

image_c = image

# plt.imshow(image_c)

# Convert image to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Use canny edge detection
edges = cv2.Canny(gray,50,150,apertureSize=3)

# Apply HoughLinesP method to
# to directly obtain line end points
lines_list =[]
lines = cv2.HoughLinesP(
            edges, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=100, # Min number of votes for valid line
            minLineLength=0, # Min allowed length of line
            maxLineGap=5 # Max allowed gap between line for joining them
            )

# Iterate over points

angles = []

for points in lines:
    # Extracted points nested in the list
    x1,y1,x2,y2=points[0]
    # Draw the lines joing the points
    # On the original image
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    
    if (angle < 0):
        angle  = angle + 90
    else:
        angle = angle - 90
        
    angles.append(angle)
    
#     cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
    # Maintain a simples lookup list for points
    lines_list.append([(x1,y1),(x2,y2)])


# for i in range(0,len(angles)):
#     if (angles[i] < 0):
#         angles[i] = angles[i]*-1 

# angles.sort()
        
# angle1 = 0

unique_values, counts = np.unique(angles, return_counts=True)
mode_index = np.argmax(counts)
mode = unique_values[mode_index]

median = angles[int(len(angles)/2)]

# angle2 = 2*sum(angles)/len(angles) - median - mode(angles) 

std_dev = np.std(angles)
var = np.var(angles)

print(median - mode)

if var > 5:
    angle1 = median + 90
else:
    angle1 = mode + 90 + var

print(angle1)
# Save the result image
plt.imshow(image)
# cv2.imwrite('detectedLines.jpg',image)
height, width = image_c.shape[:2]

# Calculate the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle1, 1.0)

# Calculate the rotated image dimensions
rotated_width = int(np.ceil(height * np.abs(np.sin(np.radians(angle1))) + width * np.abs(np.cos(np.radians(angle1)))))
rotated_height = int(np.ceil(height * np.abs(np.cos(np.radians(angle1))) + width * np.abs(np.sin(np.radians(angle1)))))

# Update the rotation matrix to include translation to keep the entire rotated image within bounds
rotation_matrix[0, 2] += (rotated_width - width) / 2
rotation_matrix[1, 2] += (rotated_height - height) / 2

# Perform the rotation using cv2.warpAffine
rotated_image = cv2.warpAffine(image_c, rotation_matrix, (rotated_width, rotated_height))

# print(rotated_image)

plt.imshow(rotated_image)
