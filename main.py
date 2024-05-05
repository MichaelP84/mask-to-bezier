import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from fitCurves import fitCurve
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import pandas as pd
from PIL import Image

def transform_segments(segments):
    # This will hold the transformed segments
    transformed = []
    
    # Iterate over the segments list except the last one
    for i in range(len(segments) - 1):
        # Get the current segment and the next segment
        current_segment = segments[i]
        next_segment = segments[i + 1]
        
        # Take the last point from the current segment
        last_point_current = current_segment[-1]
        
        # Take the first point from the next segment (which should be the same as last_point_current)
        first_point_next = next_segment[0]
        
        # Take the point before the last point in the current segment
        before_last_point_current = current_segment[-2] if len(current_segment) > 1 else last_point_current
        
        # Take the point after the first point in the next segment
        after_first_point_next = next_segment[1] if len(next_segment) > 1 else first_point_next
        
        # Add the new segment to the transformed list
        transformed.append([before_last_point_current, last_point_current, after_first_point_next])
    
    return transformed


mask = cv2.imread('./mask_to_bez/mask.png', cv2.IMREAD_GRAYSCALE)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(contours) == 0:
    print('No contours found')
    exit()
    
boundary = contours[0]  # Assuming there's only one contour

if len(contours) > 1:
    print('Multiple contours found. Using the largest one')
    for contour in contours:
        if len(contour) > len(boundary):
            boundary = contour
    
points = []

# pick every nth point
for i, point in enumerate(boundary):
    points.append(point[0])

res = fitCurve(points, 10.0)

formatted = transform_segments(res)

print("\noriginal, ", res[:2])
print("\n\nformatted, ", formatted[:2])

exit()

# Function to draw a Bezier curve given four control points
def bezier_curve(points, num=200):
    t = np.linspace(0, 1, num)
    t = t[:, np.newaxis]  # Reshape t for broadcasting
    curve = (1-t)**3*points[0] + 3*(1-t)**2*t*points[1] + 3*(1-t)*t**2*points[2] + t**3*points[3]
    return curve.T

# Load an image
img_path = './mask_to_bez/hair.png'  # Replace with your actual image path
image = Image.open(img_path)

# Define Bezier segments (example, replace with your actual data)
bezier_segments = res

# Plotting
fig, ax = plt.subplots()
ax.imshow(image)

# Draw each Bezier segment on the image
for segment in bezier_segments:
    curve = bezier_curve(segment)
    ax.plot(curve[0], curve[1], 'r')  # Draw the Bezier curve in red

    # Draw the control points
    for point in segment:
        ax.plot(point[0], point[1], 'o', color='orange')  # Draw control points in orange
        break

# Display the plot with the image and the Bezier curves
plt.axis('off')  # Turn off axis numbers and ticks
plt.show()