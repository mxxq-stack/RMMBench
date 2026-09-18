import glob
import os
import re

import cv2
import numpy as np


def save_bev(bev, resolution, min_points, path):
    os.makedirs(path, exist_ok=True)

    # Match trailing _<number> on all png files in the directory
    existing = glob.glob(os.path.join(path, "*.png"))
    max_idx = 0
    for f in existing:
        basename = os.path.basename(f)
        match = re.search(r'_(\d+)\.png$', basename)
        if match:
            max_idx = max(max_idx, int(match.group(1)))

    i = max_idx + 1
    filename = f"bev_r{resolution}_m{min_points}_{i}.png"
    filepath = os.path.join(path, filename)

    bev_vis = (bev * 255).astype(np.uint8)
    bev_vis = cv2.resize(bev_vis, (bev_vis.shape[1] * 12, bev_vis.shape[0] * 12), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filepath, bev_vis)
    return filepath


def smooth_and_rectangularize(free_mask):
    # Input free_mask: 1 = free space, 0 = obstacle

    # Extract contours
    contours, _ = cv2.findContours(free_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Take the largest contour (there should be only one)
    contour = max(contours, key=cv2.contourArea)

    # Polygon approximation
    # epsilon is the key parameter: larger means more simplification, smaller keeps more detail
    # Here 2% of the perimeter is used as epsilon; you can tune it
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)

    # Draw the approximated polygon
    result = np.zeros_like(free_mask)
    cv2.fillPoly(result, [approx], 1)


    smoothed=result
    return smoothed