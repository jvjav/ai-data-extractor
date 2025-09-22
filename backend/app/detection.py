import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import cv2
import numpy as np
from ml import craft_utils
from utils import imgproc, file_utils
from ml.craft import CRAFT
from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


class BboxDetection:
    def __init__(self, trained_model='craft_mlt_25k.pth', text_threshold=0.7, low_text=0.4, link_threshold=0.4, cuda=False, canvas_size=1280, mag_ratio=1.5, poly=False, refine=False, refiner_model='craft_refiner_CTW1500.pth'):
        weights_dir = Path(__file__).resolve().parents[1] / 'weights'
        trained_model_path = weights_dir / trained_model
        refiner_model_path = weights_dir / refiner_model

        self.text_threshold = text_threshold
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.cuda = cuda
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.poly = poly
        
        # load net
        self.net = CRAFT()

        if self.cuda:
            self.net.load_state_dict(copyStateDict(torch.load(trained_model_path)))
        else:
            self.net.load_state_dict(copyStateDict(torch.load(trained_model_path, map_location='cpu')))

        if self.cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False

        self.net.eval()

        # LinkRefiner
        self.refine_net = None
        if refine:
            from ml.refinenet import RefineNet
            self.refine_net = RefineNet()
            if self.cuda:
                self.refine_net.load_state_dict(copyStateDict(torch.load(refiner_model_path)))
                self.refine_net = self.refine_net.cuda()
                self.refine_net = torch.nn.DataParallel(self.refine_net)
            else:
                self.refine_net.load_state_dict(copyStateDict(torch.load(refiner_model_path, map_location='cpu')))

            self.refine_net.eval()
            self.poly = True

    def get_bbox(self, image_path):
        image = imgproc.loadImage(image_path)
        
        bboxes, polys, score_text = test_net(self.net, image, self.text_threshold, self.link_threshold, self.low_text, self.cuda, self.poly, self.canvas_size, self.mag_ratio, self.refine_net)
        
        return bboxes, polys, score_text

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, canvas_size, mag_ratio, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text

def estimate_text_orientation(boxes, plot=True):
    """
    Estimate the dominant text orientation from bounding boxes using PCA.

    Args:
        boxes: list of (4,2) arrays, each = word bounding box
        plot: if True, visualize boxes + orientation axis
    Returns:
        angle (float): orientation angle in radians (relative to x-axis)
        direction (np.array): 2D unit vector along the text direction
    """
    # collect centers of all boxes
    centers = []
    for b in boxes:
        pts = np.array(b)
        cx, cy = np.mean(pts, axis=0)
        centers.append([cx, cy])
    centers = np.array(centers)

    # PCA on centers
    mean = np.mean(centers, axis=0)
    cov = np.cov((centers - mean).T)
    eigvals, eigvecs = np.linalg.eig(cov)

    # principal direction = eigenvector with largest eigenvalue
    main_dir = eigvecs[:, np.argmax(eigvals)]
    main_dir = main_dir / np.linalg.norm(main_dir)  # normalize
    angle = np.arctan2(main_dir[1], main_dir[0])    # radians

    return angle, main_dir

def get_horizontal_separation(boxes):
    """
    Get the horizontal separation between boxes.

    Args:
        boxes: list of (4,2) arrays, each = word bounding box

    Returns:
        float: horizontal separation between boxes
    """

    # Get the centers: Nx2
    centers = np.array([np.mean(box, axis=0) for box in boxes])
    
    # Compute differences between x coordinates (first dimension)
    x_diffs = centers[:, 0][:, None] - centers[:, 0]
    
    # Compute differences between y coordinates (second dimension) 
    y_diffs = centers[:, 1][:, None] - centers[:, 1]

    # Combine into overall distances
    distances = np.sqrt(x_diffs**2 + y_diffs**2)

    
    return distances
    
def group_boxes_by_line(boxes, angle, distance_threshold=30):
    """
    Groups bounding boxes by line based on their estimated orientation.

    Args:
        boxes: list of (4,2) arrays, each = word bounding box
        angle: orientation angle in radians (relative to x-axis)
        distance_threshold: maximum distance for boxes to be considered on the same line

    Returns:
        list of lists: each inner list contains boxes belonging to the same line
    """
    centers = np.array([np.mean(box, axis=0) for box in boxes])

    # Calculate the projection of each center onto a vector perpendicular to the orientation
    # This essentially gives the "height" of each box relative to the slanted orientation
    perp_angle = angle + np.pi/2
    perp_vector = np.array([np.cos(perp_angle), np.sin(perp_angle)])
    projections = np.dot(centers, perp_vector)

    # Sort the boxes based on their projection (height)
    sorted_indices = np.argsort(projections)
    sorted_boxes = [boxes[i] for i in sorted_indices]
    sorted_projections = projections[sorted_indices]

    # Group boxes by line based on the distance between projections
    lines = []
    current_line = []

    if len(sorted_boxes) > 0:
        current_line.append(sorted_boxes[0])

        for i in range(1, len(sorted_boxes)):
            if sorted_projections[i] - sorted_projections[i-1] < distance_threshold:
                current_line.append(sorted_boxes[i])
            else:
                lines.append(current_line)
                current_line = [sorted_boxes[i]]

        # Add the last line
        if current_line:
            lines.append(current_line)

    return lines