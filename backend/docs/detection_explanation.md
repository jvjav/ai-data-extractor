# Explanation of `detection.py`

This document explains the functions and classes within the `backend/app/detection.py` file. This file is responsible for detecting text in images using the CRAFT (Character-Region Awareness for Text) model.

## `copyStateDict(state_dict)`

This is a utility function for loading PyTorch model weights.

-   **Purpose**: To load model state dictionaries (`state_dict`) that may have been saved with `torch.nn.DataParallel`. When a model is wrapped in `DataParallel`, the keys in its `state_dict` are prefixed with `module.`. This function removes that prefix to allow the model to be loaded correctly, whether it's in a parallelized environment or not.
-   **Parameters**:
    -   `state_dict`: The input PyTorch model state dictionary.
-   **Returns**: A new `OrderedDict` with the corrected keys.

## `class BboxDetection`

This class is the main interface for performing text detection.

### `__init__(self, ...)`

The constructor initializes the detection pipeline.

-   **Purpose**: To set up the CRAFT model and optionally a RefineNet model for more accurate bounding box prediction. It loads the pre-trained weights and configures various parameters for the detection process.
-   **Key Parameters**:
    -   `trained_model`: The filename of the pre-trained CRAFT model weights.
    -   `text_threshold`: Threshold for the text region score.
    -   `low_text`: Lower text confidence threshold for expanding bounding boxes.
    -   `link_threshold`: Threshold for the affinity (link) score between characters.
    -   `cuda`: A boolean flag to enable/disable GPU (CUDA) usage.
    -   `canvas_size`: The maximum size of the input image's longer side during inference.
    -   `mag_ratio`: Image magnification ratio.
    -   `poly`: A boolean flag to output polygonal bounding boxes instead of simple rectangles.
    -   `refine`: A boolean flag to enable/disable the use of the RefineNet model.
    -   `refiner_model`: The filename of the pre-trained RefineNet model weights.

### `get_bbox(self, image_path)`

This method performs text detection on a given image.

-   **Purpose**: To take an image file path, load the image, and use the configured CRAFT model to detect text regions.
-   **Parameters**:
    -   `image_path`: The path to the input image file.
-   **Returns**: A tuple containing:
    -   `bboxes`: A list of rectangular bounding boxes for detected text.
    -   `polys`: A list of polygonal bounding boxes for detected text.
    -   `score_text`: A heatmap image representing the text detection scores.

## `test_net(...)`

This function contains the core logic for running a single image through the text detection network.

-   **Purpose**: It handles the full pipeline of pre-processing, inference, and post-processing for text detection. It is called by `BboxDetection.get_bbox()`.
-   **Parameters**:
    -   `net`: The CRAFT model instance.
    -   `image`: The input image as a NumPy array.
    -   `text_threshold`, `link_threshold`, `low_text`: Confidence thresholds for detection.
    -   `cuda`: Boolean for GPU usage.
    -   `poly`: Boolean for polygonal output.
    -   `canvas_size`, `mag_ratio`: Image resizing parameters.
    -   `refine_net`: An optional RefineNet model instance.
-   **Process**:
    1.  **Resize & Pre-process**: The image is resized and normalized.
    2.  **Forward Pass**: The image is passed through the CRAFT network to generate two score maps: one for text regions (`score_text`) and one for character affinity (`score_link`).
    3.  **Refinement (Optional)**: If a `refine_net` is provided, it's used to refine the affinity score map for more accurate linking of characters.
    4.  **Post-processing**: The score maps are processed to extract the final bounding boxes/polygons. This involves thresholding and connecting text regions.
    5.  **Coordinate Adjustment**: The coordinates of the detected boxes are scaled back to match the original image dimensions.
-   **Returns**: The final bounding boxes (`boxes`), polygons (`polys`), and a score heatmap (`ret_score_text`). 