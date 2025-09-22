# API Endpoint Explanation

This document provides a detailed explanation of the API endpoints defined in `app/main.py`, including the methods, classes, and utilities used in their implementation.

## 1. Upload File

- **Endpoint:** `POST /upload`
- **Function:** `upload_file(file: UploadFile = File(...))`
- **Description:** This endpoint handles file uploads. It receives a file and saves it to the `data/` directory on the server.

### Utilities and Classes Used:

- **`fastapi.UploadFile`**: This class is used to handle the uploaded file data. It provides file-like async methods.
- **`fastapi.File`**: Used as a dependency to declare that the endpoint expects a file upload.
- **`os.path.join`**: This function is used to construct a valid file path for storing the uploaded file in the `UPLOAD_DIR` directory.
- **`shutil.copyfileobj`**: Efficiently copies the content of the uploaded file (from `file.file`) to a destination file opened in write-binary mode.
- **`fastapi.responses.JSONResponse`**: Returns a JSON response to the client, indicating whether the upload was successful.

## 2. Get Bounding Boxes

- **Endpoint:** `POST /get-bbox`
- **Function:** `get_bbox(file_name: str)`
- **Description:** This endpoint takes a file name as input, processes the corresponding image to detect text, and returns a list of bounding boxes for the detected text.

### Utilities and Classes Used:

- **`os.path.join` and `os.path.exists`**: Used to check if the requested image file exists in the `UPLOAD_DIR` before processing.
- **`BboxDetection.get_bbox()`**: This method from the `detection.py` module is the core of the text detection. It takes the image path, runs the CRAFT model, and returns the polygons (bounding boxes) for the detected text.
- **`models.BoundingBox`**: A Pydantic model defined in `models.py`. It's used to structure the response data, ensuring that each bounding box is a list of points. The `response_model` argument in the decorator (`@app.post("/get-bbox", response_model=List[BoundingBox])`) uses this model to validate and serialize the output.
- **`numpy.ndarray.tolist()`**: The polygons returned by the detection model are NumPy arrays. This method is called to convert them into a standard Python list format for JSON serialization.
- **`fastapi.responses.JSONResponse`**: Used to return an error message if the file is not found.

## 3. Get Bounding Boxes by Line

- **Endpoint:** `POST /get-bbox-by-line`
- **Function:** `get_bbox_by_line(file_name: str)`
- **Description:** This endpoint computes the text bounding-boxes in an image and groups them by line.

### Utilities and Classes Used:

- **`BboxDetection.get_bbox()`**: Returns the bounding boxes for the detected text.
- **`estimate_text_orientation()`**: Estimates the dominant text orientation.
- **`group_boxes_by_line()`**: Groups the bounding boxes by line based on their orientation.
- **`models.BoundingBox`**: The Pydantic model for a bounding box. The response model is `List[List[BoundingBox]]`.

## 4. Show Bounding Boxes on Image

- **Endpoint:** `POST /show-bbox`
- **Function:** `show_bbox(file_name: str)`
- **Description:** This endpoint detects text in an image and returns a new image with the bounding boxes drawn on it.

### Utilities and Classes Used:

- **`BboxDetection.get_bbox()`**: Similar to the `/get-bbox` endpoint, this is used to get the coordinates of the text bounding boxes.
- **`utils.imgproc.loadImage()`**: A helper function from `utils/imgproc.py` to load the image from the specified path.
- **`cv2` (OpenCV)**: This library is extensively used for image manipulation:
    - **`cv2.cvtColor`**: Converts the image from RGB (as loaded by `loadImage`) to BGR, which is the format expected by other OpenCV functions for drawing.
    - **`cv2.polylines`**: Draws the polygons (bounding boxes) onto the BGR image.
    - **`cv2.imencode`**: Encodes the final image (with bounding boxes) into a JPEG format in memory, preparing it to be sent in the HTTP response.
- **`numpy`**: Used to convert the polygon lists into a NumPy array with the specific shape and data type (`np.int32`) required by `cv2.polylines`.
- **`fastapi.responses.StreamingResponse`**: This response class is used to stream the generated image back to the client. This is more efficient for sending binary data like images.
- **`io.BytesIO`**: Creates an in-memory binary stream from the encoded image buffer (`buffer.tobytes()`). This stream is then used by `StreamingResponse`.

## 5. Show Bounding Boxes by Line on Image

- **Endpoint:** `POST /show-bbox-by-line`
- **Function:** `show_bbox_by_line(file_name: str)`
- **Description:** This endpoint detects text in an image, groups the bounding boxes by line, and returns a new image with each line drawn in a different color.

### Utilities and Classes Used:

- **`BboxDetection.get_bbox()`**: Returns the bounding boxes for the detected text.
- **`estimate_text_orientation()`**: Estimates the dominant text orientation from the detected bounding boxes.
- **`group_boxes_by_line()`**: Groups the bounding boxes by line based on their orientation and spatial proximity.
- **`utils.imgproc.loadImage()`**: A helper function from `utils/imgproc.py` to load the image from the specified path.
- **`cv2` (OpenCV)**: This library is used for image manipulation:
    - **`cv2.cvtColor`**: Converts the image from RGB (as loaded by `loadImage`) to BGR, which is the format expected by other OpenCV functions for drawing.
    - **`cv2.polylines`**: Draws the polygons (bounding boxes) onto the BGR image with different colors for each line.
    - **`cv2.imencode`**: Encodes the final image (with colored bounding boxes) into a JPEG format in memory.
- **`colorsys`**: Python's built-in color space conversion library:
    - **`colorsys.hsv_to_rgb()`**: Converts HSV (Hue, Saturation, Value) color values to RGB format, allowing for the generation of distinct, evenly distributed colors for each line.
- **`numpy`**: Used to convert the polygon lists into a NumPy array with the specific shape and data type (`np.int32`) required by `cv2.polylines`.
- **`fastapi.responses.StreamingResponse`**: This response class is used to stream the generated image back to the client.
- **`io.BytesIO`**: Creates an in-memory binary stream from the encoded image buffer.
- **`fastapi.responses.JSONResponse`**: Used to return error messages if the file is not found or no text is detected in the image.

### Color Generation Logic:

The endpoint generates distinct colors for each line using the HSV color space:
1. **Hue Distribution**: Each line gets a hue value evenly distributed across the color spectrum (0 to 1)
2. **Saturation and Value**: Set to maximum (1.0) for vibrant, distinct colors
3. **BGR Conversion**: The RGB values are converted to BGR format and scaled to 0-255 range for OpenCV compatibility
4. **Color Assignment**: Each line of bounding boxes is drawn with its assigned color, making it easy to visually distinguish between different lines of text 