from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import shutil
from .detection import BboxDetection, estimate_text_orientation, group_boxes_by_line
from .models import BoundingBox
from typing import List
from fastapi.responses import StreamingResponse
import io
import cv2
from utils import imgproc
import numpy as np
import colorsys

app = FastAPI()

UPLOAD_DIR = "data"
bbox_detector = BboxDetection()

@app.on_event("startup")
async def startup_event():
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a file to the UPLOAD_DIR directory.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Pre-process the image
    angle = rotate_image(file_path)
    
    return JSONResponse(status_code=200, content={"message": f"File '{file.filename}' uploaded successfully and rotated by {angle} degrees"})

def rotate_image(image_path):
    """
    Pre-processes the image by rotating it to the correct orientation.
    """
    boxes, _, _ = bbox_detector.get_bbox(image_path)

    if boxes.shape[0] == 0:
        return []

    angle, _ = estimate_text_orientation(boxes, plot=False)
    angle = np.rad2deg(angle)

    imgproc.rotateImage(image_path, -angle)

    return angle

@app.post("/get-bbox", response_model=List[BoundingBox])
async def get_bbox(file_name: str):
    """
    Computes the text bounding-boxes in an image.
    """
    image_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(image_path):
        return JSONResponse(status_code=404, content={"message": "File not found."})
    
    _, polys, _ = bbox_detector.get_bbox(image_path)
    
    # Convert polygons to BoundingBox objects
    bounding_boxes = []
    for poly in polys:
        bounding_boxes.append(BoundingBox(points=poly.tolist()))

    return bounding_boxes 

@app.post("/get-bbox-by-line", response_model=List[List[BoundingBox]])
async def get_bbox_by_line(file_name: str):
    """
    Computes the text bounding-boxes in an image, grouped by lines.
    """
    image_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(image_path):
        return JSONResponse(status_code=404, content={"message": "File not found."})
    
    boxes, _, _ = bbox_detector.get_bbox(image_path)

    if boxes.shape[0] == 0:
        return []

    angle, _ = estimate_text_orientation(boxes, plot=False)
    lines = group_boxes_by_line(boxes, angle)

    result_lines = []
    for line in lines:
        line_boxes = []
        for box in line:
            line_boxes.append(BoundingBox(points=box.tolist()))
        result_lines.append(line_boxes)

    print(f"Number of lines: {len(result_lines)}")

    return result_lines

@app.post("/show-bbox")
async def show_bbox(file_name: str):
    """
    Computes the text bounding-boxes in an image and returns the image with the boxes drawn.
    Also draws a line at the center showing the text orientation angle.
    """
    image_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(image_path):
        return JSONResponse(status_code=404, content={"message": "File not found."})
    
    # Get bounding boxes
    boxes, polys, _ = bbox_detector.get_bbox(image_path)
    
    if boxes.shape[0] == 0:
        return JSONResponse(status_code=404, content={"message": "No text detected in the image."})
    
    # Calculate orientation angle
    angle, _ = estimate_text_orientation(boxes, plot=False)
    
    # Load image
    image = imgproc.loadImage(image_path)
    
    # Draw polygons
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for poly in polys:
        poly = np.array(poly).astype(np.int32).reshape(-1, 2)
        cv2.polylines(img_bgr, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
    
    # Draw orientation line at the center of the image
    height, width = img_bgr.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Calculate line endpoints based on orientation angle
    line_length = min(width, height) // 4  # Make line 1/4 of the smaller dimension
    end_x = int(center_x + line_length * np.cos(angle))
    end_y = int(center_y + line_length * np.sin(angle))
    
    # Draw the orientation line in green with thickness 3
    cv2.line(img_bgr, (center_x, center_y), (end_x, end_y), (0, 255, 0),10)
    
    # Draw a small circle at the center point
    cv2.circle(img_bgr, (center_x, center_y), 5, (0, 255, 0), -1)
    
    # Encode image to memory
    is_success, buffer = cv2.imencode(".jpg", img_bgr)
    if not is_success:
        return JSONResponse(status_code=500, content={"message": "Failed to encode image."})
    
    # Return image as a streaming response
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg") 

@app.post("/show-bbox-by-line")
async def show_bbox_by_line(file_name: str):
    """
    Computes the text bounding-boxes in an image, groups them by lines, and returns the image with the boxes drawn in different colors for each line.
    Each line is also labeled with a number for clear identification.
    Also draws a line at the center showing the text orientation angle.
    """
    image_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(image_path):
        return JSONResponse(status_code=404, content={"message": "File not found."})
    
    # Get bounding boxes
    boxes, _, _ = bbox_detector.get_bbox(image_path)
    
    if boxes.shape[0] == 0:
        return JSONResponse(status_code=404, content={"message": "No text detected in the image."})
    
    # Group boxes by line and calculate orientation angle
    angle, _ = estimate_text_orientation(boxes, plot=False)
    lines = group_boxes_by_line(boxes, angle)
    
    # Load image
    image = imgproc.loadImage(image_path)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Generate distinct colors for each line
    num_lines = len(lines)
    colors = []
    for i in range(num_lines):
        # Use HSV color space to generate distinct colors
        hue = i / num_lines
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        # Convert to BGR for OpenCV (values 0-255)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    
    # Draw polygons with different colors for each line and add line numbers
    for line_idx, line in enumerate(lines):
        color = colors[line_idx]
        
        # Draw bounding boxes for this line
        for box in line:
            poly = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(img_bgr, [poly.reshape((-1, 1, 2))], True, color=color, thickness=2)
        
    # Draw orientation line at the center of the image
    height, width = img_bgr.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Calculate line endpoints based on orientation angle
    line_length = min(width, height) // 4  # Make line 1/4 of the smaller dimension
    end_x = int(center_x + line_length * np.cos(angle))
    end_y = int(center_y + line_length * np.sin(angle))
    
    # Draw the orientation line in green with thickness 3
    cv2.line(img_bgr, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3)
    
    # Draw a small circle at the center point
    cv2.circle(img_bgr, (center_x, center_y), 5, (0, 255, 0), -1)
    
    # Encode image to memory
    is_success, buffer = cv2.imencode(".jpg", img_bgr)
    if not is_success:
        return JSONResponse(status_code=500, content={"message": "Failed to encode image."})
    
    # Return image as a streaming response
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg") 