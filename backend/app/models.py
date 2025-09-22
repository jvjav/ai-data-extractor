from pydantic import BaseModel
from typing import List

class BoundingBox(BaseModel):
    """
    Represents a single bounding box with a list of points.
    """
    points: List[List[float]]