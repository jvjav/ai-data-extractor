# Backend Documentation

This document provides an overview of the backend structure, API endpoints, and instructions on how to run the application.

## Folder Structure

The backend is organized into several directories, each with a specific purpose.

- `app/`: Contains the core FastAPI application.
  - `main.py`: Defines the API endpoints (`/upload`, `/get-bbox`) and manages the application lifecycle.
  - `detection.py`: Handles the text detection logic using the CRAFT model. It loads the model and processes images to find text bounding boxes.
  - `models.py`: Defines the Pydantic data models used for API requests and responses, such as `BoundingBox`.
- `ml/`: Contains the implementation of the machine learning models.
  - `craft.py`: The core CRAFT text detection model.
  - `craft_utils.py`: Utility functions for processing the model's output.
  - `refinenet.py`: An optional model to refine the bounding box predictions.
- `weights/`: Stores the pre-trained model weights required by the ML models.
  - `craft_mlt_25k.pth`: The default CRAFT model weights.
- `utils/`: Contains utility functions used across the application.
  - `imgproc.py`: Functions for image processing, such as resizing and normalization.
  - `file_utils.py`: Helper functions for file operations.
- `data/`: The default directory where uploaded images are stored.
- `result/`: A directory intended for storing processing results (currently not used by the API).
- `requirements.txt`: A list of all Python dependencies required to run the backend.

## How to use the API

### 1. Installation

To set up the backend, you need to install the required Python dependencies.

```bash
pip install -r requirements.txt
```

### 2. Running the Server

You can run the FastAPI application using `uvicorn`.

```bash
uvicorn app.main:app --reload
```

The server will be available at `http://127.0.0.1:8000`.

### 3. API Endpoints

The API provides three endpoints for handling image uploads and text detection.

#### Upload File

- **Endpoint:** `POST /upload`
- **Description:** Uploads an image file to the server and rotates it by the orientation of the text detected. The file will be stored in the `data/` directory.
- **Request:** `multipart/form-data` with a `file` field containing the image.
- **Response:** A JSON message indicating success or failure.

**Example using cURL:**

```bash
curl -X POST "http://127.0.0.1:8000/upload" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/image.jpg"
```

#### Get Bounding Boxes

- **Endpoint:** `POST /get-bbox`
- **Description:** Detects text in a previously uploaded image and returns the bounding boxes.
- **Request:** A JSON object with the `file_name` of the image.
- **Response:** A JSON array of bounding boxes. Each bounding box is a list of points `[[x1, y1], [x2, y2], ..., [xn, yn]]`.

**Example using cURL:**

```bash
curl -X POST "http://127.0.0.1:8000/get-bbox?file_name=your_image.jpg" -H "accept: application/json"
```

**Example Response:**

```json
[
  {
    "points": [
      [100.0, 50.0],
      [300.0, 50.0],
      [300.0, 100.0],
      [100.0, 100.0]
    ]
  },
  {
    "points": [
      [120.0, 150.0],
      [320.0, 150.0],
      [320.0, 200.0],
      [120.0, 200.0]
    ]
  }
]
```

#### Get Bounding Boxes by Line

- **Endpoint:** `POST /get-bbox-by-line`
- **Description:** Detects text in a previously uploaded image and returns the bounding boxes grouped by line.
- **Request:** A JSON object with the `file_name` of the image.
- **Response:** A JSON array of lines, where each line is an array of bounding boxes.

**Example using cURL:**

```bash
curl -X POST "http://127.0.0.1:8000/get-bbox-by-line?file_name=your_image.jpg" -H "accept: application/json"
```

**Example Response:**

```json
[
  [
    {
      "points": [
        [100.0, 50.0],
        [300.0, 50.0],
        [300.0, 100.0],
        [100.0, 100.0]
      ]
    }
  ],
  [
    {
      "points": [
        [120.0, 150.0],
        [320.0, 150.0],
        [320.0, 200.0],
        [120.0, 200.0]
      ]
    }
  ]
]
```

#### Show Bounding Boxes on Image

- **Endpoint:** `POST /show-bbox`
- **Description:** Detects text in a previously uploaded image and returns an image with the bounding boxes drawn on it.
- **Request:** A query parameter `file_name` with the name of the image you want to process.
- **Response:** An image in JPEG format with the bounding boxes visualized.

**Example using cURL:**

```bash
curl -X POST "http://127.0.0.1:8000/show-bbox?file_name=your_image.jpg" -o result_image.jpg
```

#### Show Bounding Boxes by Line on Image (TO BE IMPROVED)

This endpoint should be improved

- **Endpoint:** `POST /show-bbox-by-line`
- **Description:** Detects text in a previously uploaded image, groups the bounding boxes by line, and returns an image with each line drawn in a different color.
- **Request:** A query parameter `file_name` with the name of the image you want to process.
- **Response:** An image in JPEG format with the bounding boxes visualized, where each line of text is outlined in a distinct color.

**Example using cURL:**

```bash
curl -X POST "http://127.0.0.1:8000/show-bbox-by-line?file_name=your_image.jpg" -o result_image_colored.jpg
```