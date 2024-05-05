import modal
from modal import Image, web_endpoint
import numpy as np
import cv2
from fitCurves import fitCurve
import base64

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

bezier_image = Image.debian_slim().pip_install(
    "numpy==1.24.3",
    "opencv-python-headless",
)

stub = modal.Stub("bezier")

@stub.function(
    image=bezier_image,
    container_idle_timeout=120,
)
@web_endpoint(method="POST")
def process_mask(file: dict):
    base64_image_data = file["image"]
    error = file["error"]
    if "," in base64_image_data:
        base64_image_data = base64_image_data.split(",")[1]
    image_data = base64.b64decode(base64_image_data)
    # print(image_data)
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)

    mask = image

    # Step 1: Find boundary points of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))

    if len(contours) == 0:
        print('No contours found')
        return None

    boundary = contours[0]  # Assuming there's only one contour
    if len(contours) > 1:
        print('Multiple contours found. Using the largest one')
        for contour in contours:
            if len(contour) > len(boundary):
                boundary = contour

    points = []
    # pick every nth point
    for point in boundary:
        points.append(point[0])
                
    res = fitCurve(points, error)
    # res is a list of lists of points
    serializable_points = [[list(map(int, point)) for point in bezier] for bezier in res]
    
    # convert from segments being 4 points to intuitive 3 points (left handle, anchor, right handle)    
    formatted = transform_segments(serializable_points)
    
    return {"bezier": formatted}